/* cvf-live.js - the computing engine for learn-computer-vision-with-phoebe
 *
 * Real Fashion-MNIST images (cvf-sample.js): 2,000 training images as the unlabelled pool,
 * 1,000 official test images as the fixed holdout, all pooled to 14x14 and 16 grey levels.
 * Everything below is computed here, in the browser, from those pixels. Nothing is asserted.
 *
 * What is REAL: every pixel, every convolution, every feature, every trained weight, every
 *   accuracy, error and confusion count, every picked image.
 * What is MODELLED, and labelled so on the widget: the label noise rung (a stated rule: three
 *   labels in ten replaced by the class a rushed labeller confuses it with), and the camera
 *   shifts on session 5 (stated array operations standing in for a dimmer lamp, a turned item).
 *
 * The ladder, all at an equal budget of 300 labels drawn from the same pool:
 *   majority     predict the most common labelled class
 *   raw          196 pixel values, softmax regression
 *   hand         33 hand-made measurements (row and column profiles, ink, box, symmetry)
 *   conv         8 fixed 3x3 kernels, ReLU, 2x2 max pool: 288 features
 *   uncertainty  conv, labels chosen 50 at a time by smallest margin (Settles 2009)
 *   diversity    conv, labels chosen by k-centre greedy (Sener and Savarese 2018)
 *   noisy        ANTI: conv on the random 300, three labels in ten wrong
 * materials/cvf_reference.py recomputes every number with numpy; compare-ladder.py checks both.
 */
(function (root) {
  "use strict";

  var STEPS = 150, LR = 0.5, L2 = 1e-3;
  var BUDGET = 300, SEED_SET = 50, BATCH = 50, SEED = 7;
  var CURVE = [50, 100, 150, 200, 300, 400, 500];
  var NOISE = 0.30;
  var CONFUSE = [6, 3, 4, 0, 2, 7, 0, 5, 2, 7];
  var KERNELS = {
    vertical:   [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
    horizontal: [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
    diag_down:  [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],
    diag_up:    [[2, 1, 0], [1, 0, -1], [0, -1, -2]]
  };
  var SHORT = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Boot"];

  /* ---------- seeded generator (mulberry32), identical in the Python reference ---------- */
  function mulberry32(a) {
    return function () {
      a |= 0; a = a + 0x6D2B79F5 | 0;
      var t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }
  function permutation(n, seed) {
    var rnd = mulberry32(seed), a = [];
    for (var i = 0; i < n; i++) a.push(i);
    for (var k = n - 1; k > 0; k--) {
      var j = Math.floor(rnd() * (k + 1));
      var t = a[k]; a[k] = a[j]; a[j] = t;
    }
    return a;
  }

  /* ---------- decoding ---------- */
  function b64bytes(s) {
    if (typeof atob === "function") {
      var bin = atob(s), out = new Uint8Array(bin.length);
      for (var i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
      return out;
    }
    return new Uint8Array(Buffer.from(s, "base64"));
  }
  /* returns an array of n images, each a Float64Array(196) row-major, values level/15 */
  function unpack(b64, n) {
    var b = b64bytes(b64), imgs = [];
    for (var i = 0; i < n; i++) {
      var x = new Float64Array(196);
      for (var p = 0; p < 98; p++) {
        var v = b[i * 98 + p];
        x[2 * p] = (v >> 4) / 15; x[2 * p + 1] = (v & 15) / 15;
      }
      imgs.push(x);
    }
    return imgs;
  }
  function prepare(sample) {
    var g = b64bytes(sample.gallery.px), gal = [];
    for (var i = 0; i < sample.gallery.y.length; i++) gal.push(g.subarray(i * 784, (i + 1) * 784));
    return {
      classes: sample.classes,
      xp: unpack(sample.pool.px, sample.pool.n), yp: sample.pool.y.slice(),
      xh: unpack(sample.holdout.px, sample.holdout.n), yh: sample.holdout.y.slice(),
      gallery: gal, gy: sample.gallery.y.slice()
    };
  }

  /* ---------- features ---------- */
  function featRaw(x) { return Float64Array.from(x); }
  function featHand(x) {
    var f = new Float64Array(33), r, c, s;
    for (r = 0; r < 14; r++) { s = 0; for (c = 0; c < 14; c++) s += x[r * 14 + c]; f[r] = s / 14; }
    for (c = 0; c < 14; c++) { s = 0; for (r = 0; r < 14; r++) s += x[r * 14 + c]; f[14 + c] = s / 14; }
    s = 0; for (var i = 0; i < 196; i++) s += x[i]; f[28] = s / 196;
    var r0 = -1, r1 = -1, c0 = 99, c1 = -1;
    for (r = 0; r < 14; r++) for (c = 0; c < 14; c++) if (x[r * 14 + c] >= 0.2) {
      if (r0 < 0) r0 = r; r1 = r; if (c < c0) c0 = c; if (c > c1) c1 = c;
    }
    f[29] = r0 >= 0 ? (r1 - r0 + 1) / 14 : 0;
    f[30] = r0 >= 0 ? (c1 - c0 + 1) / 14 : 0;
    s = 0; for (r = 0; r < 14; r++) for (c = 0; c < 14; c++) s += Math.abs(x[r * 14 + c] - x[r * 14 + 13 - c]);
    f[31] = s / 196;
    var top = 0, bot = 0;
    for (i = 0; i < 98; i++) top += x[i];
    for (i = 98; i < 196; i++) bot += x[i];
    f[32] = top / 98 - bot / 98;
    return f;
  }
  /* cross-correlation, no padding, stride 1 (what every library calls convolution) */
  function convValid(x, side, k) {
    var o = side - 2, out = new Float64Array(o * o);
    for (var a = 0; a < 3; a++) for (var b = 0; b < 3; b++) {
      var w = k[a][b];
      if (!w) continue;
      for (var r = 0; r < o; r++) for (var c = 0; c < o; c++) out[r * o + c] += w * x[(r + a) * side + c + b];
    }
    return out;
  }
  function maxpool2(m, side) {
    var h = side >> 1, out = new Float64Array(h * h);
    for (var r = 0; r < h; r++) for (var c = 0; c < h; c++) {
      var i = 2 * r * side + 2 * c;
      out[r * h + c] = Math.max(m[i], m[i + 1], m[i + side], m[i + side + 1]);
    }
    return out;
  }
  function featConv(x) {
    var f = new Float64Array(288), o = 0;
    Object.keys(KERNELS).forEach(function (name) {
      var r = convValid(x, 14, KERNELS[name]);
      var pos = new Float64Array(144), neg = new Float64Array(144);
      for (var i = 0; i < 144; i++) { var v = r[i] / 4; pos[i] = v > 0 ? v : 0; neg[i] = -v > 0 ? -v : 0; }
      var a = maxpool2(pos, 12), b = maxpool2(neg, 12);
      f.set(a, o); f.set(b, o + 36); o += 72;
    });
    return f;
  }
  var FEATS = { raw: featRaw, hand: featHand, conv: featConv };

  /* ---------- standardise on the pool, then softmax regression ---------- */
  function standardiser(F) {
    var d = F[0].length, n = F.length, mu = new Float64Array(d), sd = new Float64Array(d), i, j;
    for (i = 0; i < n; i++) for (j = 0; j < d; j++) mu[j] += F[i][j];
    for (j = 0; j < d; j++) mu[j] /= n;
    for (i = 0; i < n; i++) for (j = 0; j < d; j++) { var e = F[i][j] - mu[j]; sd[j] += e * e; }
    for (j = 0; j < d; j++) { sd[j] = Math.sqrt(sd[j] / n); if (sd[j] === 0) sd[j] = 1; }
    return { mu: mu, sd: sd };
  }
  function design(F, st) {
    return F.map(function (f) {
      var d = f.length, z = new Float64Array(d + 1);
      for (var j = 0; j < d; j++) z[j] = (f[j] - st.mu[j]) / st.sd[j];
      z[d] = 1;
      return z;
    });
  }
  function softmaxRow(L) {
    var m = -Infinity, k, s = 0, P = new Float64Array(10);
    for (k = 0; k < 10; k++) if (L[k] > m) m = L[k];
    for (k = 0; k < 10; k++) { P[k] = Math.exp(L[k] - m); s += P[k]; }
    for (k = 0; k < 10; k++) P[k] /= s;
    return P;
  }
  function probs(W, z) {
    var d = z.length, L = new Float64Array(10);
    for (var j = 0; j < d; j++) { var v = z[j]; if (!v) continue; var o = j * 10; for (var k = 0; k < 10; k++) L[k] += v * W[o + k]; }
    return softmaxRow(L);
  }
  function train(Z, y, steps) {
    steps = steps || STEPS;
    var n = Z.length, d = Z[0].length, W = new Float64Array(d * 10), G = new Float64Array(d * 10);
    for (var t = 0; t < steps; t++) {
      G.fill(0);
      for (var i = 0; i < n; i++) {
        var z = Z[i], P = probs(W, z);
        P[y[i]] -= 1;
        for (var j = 0; j < d; j++) { var v = z[j]; if (!v) continue; var o = j * 10; for (var k = 0; k < 10; k++) G[o + k] += v * P[k]; }
      }
      for (var q = 0; q < d * 10; q++) {
        var g = G[q] / n;
        if (q < (d - 1) * 10) g += L2 * W[q];
        W[q] -= LR * g;
      }
    }
    return W;
  }
  function evaluate(W, Zh, yh) {
    var correct = 0, conf = [], pred = [], i, c;
    for (c = 0; c < 10; c++) conf.push([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    for (i = 0; i < Zh.length; i++) {
      var P = probs(W, Zh[i]), best = 0;
      for (c = 1; c < 10; c++) if (P[c] > P[best]) best = c;
      pred.push(best); conf[yh[i]][best]++;
      if (best === yh[i]) correct++;
    }
    var per = [];
    for (c = 0; c < 10; c++) {
      var tot = 0; for (var k = 0; k < 10; k++) tot += conf[c][k];
      per.push(round6((tot - conf[c][c]) / tot));
    }
    return { correct: correct, acc: correct / yh.length, perClassError: per, confusion: conf, pred: pred };
  }
  function round6(v) { return Math.round(v * 1e6) / 1e6; }
  function round9(v) { return Math.round(v * 1e9) / 1e9; }
  function pick(arr, idx) { return idx.map(function (i) { return arr[i]; }); }

  /* ---------- choosing which images to label ---------- */
  function pickRandom(n, budget, seed) { return permutation(n, seed).slice(0, budget); }
  function pickUncertainty(Zp, yp, budget, seed) {
    var chosen = permutation(Zp.length, seed).slice(0, SEED_SET);
    while (chosen.length < budget) {
      var W = train(pick(Zp, chosen), pick(yp, chosen));
      var taken = {}; chosen.forEach(function (i) { taken[i] = 1; });
      var cand = [];
      for (var i = 0; i < Zp.length; i++) {
        if (taken[i]) continue;
        var P = probs(W, Zp[i]), a = -1, b = -1;
        for (var k = 0; k < 10; k++) { if (P[k] > a) { b = a; a = P[k]; } else if (P[k] > b) b = P[k]; }
        cand.push([round9(a - b), i]);
      }
      cand.sort(function (u, v) { return u[0] - v[0] || u[1] - v[1]; });
      var need = Math.min(BATCH, budget - chosen.length);
      for (var c = 0; c < need; c++) chosen.push(cand[c][1]);
    }
    return chosen;
  }
  function pickDiversity(Zp, budget, seed) {
    var n = Zp.length, d = Zp[0].length - 1, sq = new Float64Array(n), dmin = new Float64Array(n).fill(Infinity), i, j;
    for (i = 0; i < n; i++) { var s = 0; for (j = 0; j < d; j++) s += Zp[i][j] * Zp[i][j]; sq[i] = s; }
    function relax(c) {
      var xc = Zp[c];
      for (var i = 0; i < n; i++) {
        var dot = 0, xi = Zp[i];
        for (var j = 0; j < d; j++) dot += xi[j] * xc[j];
        var v = Math.round((sq[i] + sq[c] - 2 * dot) * 1e6) / 1e6;
        if (v < dmin[i]) dmin[i] = v;
      }
    }
    var chosen = permutation(n, seed).slice(0, SEED_SET), taken = {};
    chosen.forEach(function (c) { taken[c] = 1; relax(c); });
    while (chosen.length < budget) {
      var best = -1, bv = -Infinity;
      for (i = 0; i < n; i++) { var v = taken[i] ? -1 : dmin[i]; if (v > bv) { bv = v; best = i; } }
      chosen.push(best); taken[best] = 1; relax(best);
    }
    return chosen;
  }
  function noisy(labels, seed) {
    var rnd = mulberry32(seed);
    return labels.map(function (y) { return rnd() < NOISE ? CONFUSE[y] : y; });
  }

  /* ---------- session 5: shifts, calibration, IoU ---------- */
  var SHIFTS = {
    clean: function (x) { return x; },
    dim: function (x) { return x.map(function (v) { return v * 0.6; }); },
    flip: function (x) { var o = new Float64Array(196); for (var r = 0; r < 14; r++) for (var c = 0; c < 14; c++) o[r * 14 + c] = x[r * 14 + 13 - c]; return o; },
    shift: function (x) { var o = new Float64Array(196); for (var r = 0; r < 14; r++) for (var c = 2; c < 14; c++) o[r * 14 + c] = x[r * 14 + c - 2]; return o; },
    rot90: function (x) { var o = new Float64Array(196); for (var r = 0; r < 14; r++) for (var c = 0; c < 14; c++) o[r * 14 + c] = x[(13 - c) * 14 + r]; return o; }
  };
  function ece(Ps, y, bins) {
    bins = bins || 10;
    var tot = 0, nb = [], cb = [], ab = [], k;
    for (k = 0; k < bins; k++) { nb.push(0); cb.push(0); ab.push(0); }
    Ps.forEach(function (P, i) {
      var best = 0; for (var c = 1; c < 10; c++) if (P[c] > P[best]) best = c;
      var b = Math.min(bins - 1, Math.floor(P[best] * bins));
      nb[b]++; cb[b] += P[best]; ab[b] += best === y[i] ? 1 : 0;
    });
    for (k = 0; k < bins; k++) if (nb[k]) tot += nb[k] / y.length * Math.abs(ab[k] / nb[k] - cb[k] / nb[k]);
    return { ece: tot, bins: nb.map(function (n, k) { return { n: n, acc: n ? ab[k] / n : 0, conf: n ? cb[k] / n : 0 }; }) };
  }
  function iou(a, b) {
    var ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
    var iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
    var inter = ix * iy;
    var union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
    return union ? inter / union : 0;
  }
  var IOU_PRESETS = {
    tight:   [[40, 30, 160, 150], [46, 34, 164, 152]],
    shifted: [[40, 30, 160, 150], [80, 30, 200, 150]],
    loose:   [[40, 30, 160, 150], [20, 10, 190, 180]],
    small:   [[40, 30, 160, 150], [70, 60, 130, 120]]
  };

  /* ---------- the engine ---------- */
  function Engine(sample) {
    this.d = prepare(sample);
    this.cache = {};
  }
  Engine.prototype.Z = function (key) {
    if (this.cache[key]) return this.cache[key];
    var fn = FEATS[key], d = this.d;
    var fp = d.xp.map(fn), fh = d.xh.map(fn), st = standardiser(fp);
    this.cache[key] = { p: design(fp, st), h: design(fh, st), st: st };
    return this.cache[key];
  };
  Engine.prototype.rung = function (id) {
    var d = this.d, n = d.yp.length, t0 = Date.now(), idx, labels, fkey = "conv";
    var rand = pickRandom(n, BUDGET, SEED);
    if (id === "majority") {
      var cnt = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]; rand.forEach(function (i) { cnt[d.yp[i]]++; });
      var major = 0; for (var c = 1; c < 10; c++) if (cnt[c] > cnt[major]) major = c;
      var cor = 0; d.yh.forEach(function (y) { if (y === major) cor++; });
      var conf = []; for (c = 0; c < 10; c++) { var row = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]; conf.push(row); }
      d.yh.forEach(function (y) { conf[y][major]++; });
      var per = []; for (c = 0; c < 10; c++) per.push(c === major ? 0 : 1);
      return { id: id, correct: cor, acc: cor / d.yh.length, predicts: major, labels: BUDGET, chosen: rand,
               perClassError: per, confusion: conf, ms: Date.now() - t0 };
    }
    if (id === "raw" || id === "hand") fkey = id;
    var Z = this.Z(fkey);
    if (id === "uncertainty") idx = pickUncertainty(Z.p, d.yp, BUDGET, SEED);
    else if (id === "diversity") idx = pickDiversity(Z.p, BUDGET, SEED);
    else if (id === "all") { idx = []; for (var i = 0; i < n; i++) idx.push(i); }
    else idx = rand;
    labels = pick(d.yp, idx);
    if (id === "noisy") labels = noisy(labels, SEED + 1);
    var W = train(pick(Z.p, idx), labels);
    var e = evaluate(W, Z.h, d.yh);
    e.id = id; e.labels = idx.length; e.chosen = idx; e.W = W; e.ms = Date.now() - t0;
    if (id === "noisy") { var wrong = 0; labels.forEach(function (l, k) { if (l !== d.yp[idx[k]]) wrong++; }); e.wrongLabels = wrong; }
    return e;
  };
  Engine.prototype.curvePoint = function (strategy, budget) {
    var Z = this.Z("conv"), d = this.d, idx;
    if (strategy === "random") idx = pickRandom(d.yp.length, budget, SEED);
    else if (strategy === "uncertainty") idx = pickUncertainty(Z.p, d.yp, budget, SEED);
    else idx = pickDiversity(Z.p, budget, SEED);
    var W = train(pick(Z.p, idx), pick(d.yp, idx));
    return evaluate(W, Z.h, d.yh).correct;
  };
  Engine.prototype.randomSeed = function (seed) {
    var Z = this.Z("conv"), d = this.d, idx = pickRandom(d.yp.length, BUDGET, seed);
    return evaluate(train(pick(Z.p, idx), pick(d.yp, idx)), Z.h, d.yh).correct;
  };
  Engine.prototype.shift = function (name, W) {
    var Z = this.Z("conv"), d = this.d, fn = SHIFTS[name];
    var Zs = design(d.xh.map(function (x) { return featConv(fn(x)); }), Z.st);
    var Ps = Zs.map(function (z) { return probs(W, z); });
    var cor = 0, conf = 0, wrong = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], tot = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    Ps.forEach(function (P, i) {
      var best = 0; for (var c = 1; c < 10; c++) if (P[c] > P[best]) best = c;
      conf += P[best]; tot[d.yh[i]]++;
      if (best === d.yh[i]) cor++; else wrong[d.yh[i]]++;
    });
    var e = ece(Ps, d.yh);
    return { correct: cor, meanConf: round6(conf / Ps.length), ece: round6(e.ece), bins: e.bins,
             perClassError: wrong.map(function (w, c) { return round6(w / tot[c]); }) };
  };
  Engine.prototype.inkSlice = function (W) {
    var Z = this.Z("conv"), d = this.d;
    var ink = d.xh.map(function (x) { var s = 0; for (var i = 0; i < 196; i++) s += x[i]; return s / 196; });
    var srt = ink.slice().sort(function (a, b) { return a - b; }), cut = srt[(ink.length >> 2) - 1];
    var o = { faintN: 0, faintCorrect: 0, restN: 0, restCorrect: 0, cut: round6(cut) };
    d.xh.forEach(function (x, i) {
      var P = probs(W, Z.h[i]), best = 0; for (var c = 1; c < 10; c++) if (P[c] > P[best]) best = c;
      var ok = best === d.yh[i] ? 1 : 0;
      if (ink[i] <= cut) { o.faintN++; o.faintCorrect += ok; } else { o.restN++; o.restCorrect += ok; }
    });
    return o;
  };

  /* the whole canon, in the same shape as cvf_reference.py prints */
  function canon(sample) {
    var E = new Engine(sample), out = { budget: BUDGET, rungs: {} };
    ["majority", "raw", "hand", "conv", "uncertainty", "diversity", "noisy", "all"].forEach(function (id) {
      var r = E.rung(id);
      out.rungs[id] = id === "majority" ? { correct: r.correct, acc: r.acc, predicts: r.predicts }
        : { correct: r.correct, acc: r.acc, perClassError: r.perClassError, confusion: r.confusion };
      if (id === "all") out._W = r.W;
    });
    out.curve = { budgets: CURVE, random: [], uncertainty: [], diversity: [] };
    CURVE.forEach(function (b) {
      ["random", "uncertainty", "diversity"].forEach(function (s) { out.curve[s].push(E.curvePoint(s, b)); });
    });
    out.randomSeeds = [7, 8, 9, 10, 11].map(function (s) { return E.randomSeed(s); });
    out.shift = {};
    Object.keys(SHIFTS).forEach(function (k) { var s = E.shift(k, out._W); delete s.bins; out.shift[k] = s; });
    out.inkSlice = E.inkSlice(out._W);
    out.iou = {};
    Object.keys(IOU_PRESETS).forEach(function (k) { out.iou[k] = round6(iou(IOU_PRESETS[k][0], IOU_PRESETS[k][1])); });
    var cc = function (idx) { var c = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]; idx.forEach(function (i) { c[E.d.yp[i]]++; }); return c; };
    out.labelledClassCounts = {
      random: cc(pickRandom(E.d.yp.length, BUDGET, SEED)),
      uncertainty: cc(pickUncertainty(E.Z("conv").p, E.d.yp, BUDGET, SEED)),
      diversity: cc(pickDiversity(E.Z("conv").p, BUDGET, SEED))
    };
    out.gallery = E.d.gallery.map(function (px, i) {
      var g = { label: E.d.gy[i] };
      ["none", "bright", "contrast", "flip", "invert"].forEach(function (op) { var st = pixelStats(applyOp(px, op)); g[op] = [round6(st.mean), st.zeros, st.sat]; });
      var x = new Float64Array(784); for (var k = 0; k < 784; k++) x[k] = px[k];
      Object.keys(DEMO_KERNELS).forEach(function (kn) { var cs = convStats(x, kn); g["k_" + kn] = [round6(cs.max), cs.above40]; });
      return g;
    });
    delete out._W;
    return out;
  }

  /* =====================================================================================
   * UI. Four mount points, each optional:
   *   #cvf-pixels  session 1: a real 28x28 image as numbers; brightness, contrast, flip
   *   #cvf-conv    session 2: pick a kernel, watch it slide over a real image
   *   #cvf-bench   session 4: the label budget bench (the whole ladder)
   *   #cvf-shift   session 5: the all-labels model under camera shift, plus IoU boxes
   * ===================================================================================== */
  function esc(s) {
    return String(s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; });
  }
  function pct1(x) { return (x * 100).toFixed(1) + "%"; }
  function comma(n) { return String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ","); }
  function el(tag, cls, html) { var e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; }
  function drawGrey(canvas, px, side, scale) {
    canvas.width = side * scale; canvas.height = side * scale;
    var g = canvas.getContext("2d");
    for (var r = 0; r < side; r++) for (var c = 0; c < side; c++) {
      var v = Math.max(0, Math.min(255, Math.round(px[r * side + c])));
      var ink = 255 - v;                      /* dark ink on light paper: 0 is paper, 255 is ink */
      g.fillStyle = "rgb(" + ink + "," + ink + "," + ink + ")";
      g.fillRect(c * scale, r * scale, scale, scale);
    }
  }
  function metric(label, value, unit, kind) {
    return '<div class="mb-metric"><span class="mb-mlabel">' + esc(label) + "</span>" +
      '<span class="mb-mvalue">' + esc(value) + "</span>" +
      '<span class="mb-munit">' + esc(unit) + "</span>" +
      '<span class="mb-mkind is-' + kind + '">' + kind + "</span></div>";
  }

  /* ---------- helpers shared by the widgets and the canon ---------- */
  function applyOp(px, op) {
    var o = new Float64Array(784), r, c, i;
    for (i = 0; i < 784; i++) o[i] = px[i];
    if (op === "bright") for (i = 0; i < 784; i++) o[i] = Math.min(255, o[i] + 60);
    if (op === "contrast") for (i = 0; i < 784; i++) o[i] = Math.max(0, Math.min(255, Math.floor((o[i] - 128) * 1.5 + 128 + 0.5)));
    if (op === "invert") for (i = 0; i < 784; i++) o[i] = 255 - o[i];
    if (op === "flip") { var f = new Float64Array(784); for (r = 0; r < 28; r++) for (c = 0; c < 28; c++) f[r * 28 + c] = o[r * 28 + 27 - c]; o = f; }
    return o;
  }
  function pixelStats(px) {
    var s = 0, zeros = 0, sat = 0;
    for (var i = 0; i < 784; i++) { s += px[i]; if (px[i] === 0) zeros++; if (px[i] === 255) sat++; }
    return { sum: s, mean: s / 784, zeros: zeros, sat: sat };
  }
  function convStats(x, kname) {
    var K = DEMO_KERNELS[kname], r = convValid(x, 28, K.k), out = new Float64Array(r.length), mx = 0, nz = 0;
    for (var i = 0; i < r.length; i++) { out[i] = Math.abs(r[i] / K.div); if (out[i] > mx) mx = out[i]; if (out[i] > 40) nz++; }
    return { out: out, max: mx, above40: nz };
  }

  /* ---------- session 1: pixels ---------- */
  function mountPixels(E, rootEl) {
    var d = E.d, cur = 0, op = "none";
    var bar = el("div", "mb-presets cvf-pickrow");
    var ops = [["none", "as stored"], ["bright", "brighter (+60)"], ["contrast", "contrast x1.5"], ["flip", "flip left-right"], ["invert", "invert"]];
    var picks = el("div", "cvf-thumbs");
    d.gallery.forEach(function (px, i) {
      if (i % 2) return;
      var b = el("button", "cvf-thumb"); b.type = "button";
      b.title = d.classes[d.gy[i]];
      var cv = document.createElement("canvas"); drawGrey(cv, px, 28, 2); b.appendChild(cv);
      b.appendChild(el("span", "", esc(SHORT[d.gy[i]])));
      b.addEventListener("click", function () { cur = i; render(); });
      picks.appendChild(b);
    });
    ops.forEach(function (o) {
      var b = el("button", "rqs-preset", esc(o[1])); b.type = "button"; b.dataset.op = o[0];
      b.addEventListener("click", function () { op = o[0]; render(); });
      bar.appendChild(b);
    });
    var view = el("div", "cvf-pixview");
    var big = document.createElement("canvas"); big.className = "cvf-big";
    var nums = el("div", "cvf-nums");
    var read = el("div", "cvf-read");
    view.appendChild(big); view.appendChild(nums);
    rootEl.appendChild(picks); rootEl.appendChild(bar); rootEl.appendChild(view); rootEl.appendChild(read);
    function apply(px) { return applyOp(px, op); }
    function render() {
      [].forEach.call(bar.children, function (b) { b.classList.toggle("is-on", b.dataset.op === op); });
      [].forEach.call(picks.children, function (b, k) { b.classList.toggle("is-on", k * 2 === cur); });
      var px = apply(d.gallery[cur]);
      drawGrey(big, px, 28, 8);
      var rows = "";
      for (var r = 8; r < 16; r++) {
        rows += "<tr>";
        for (var c = 10; c < 18; c++) rows += "<td>" + Math.round(px[r * 28 + c]) + "</td>";
        rows += "</tr>";
      }
      nums.innerHTML = '<p class="cvf-cap">Rows 8 to 15, columns 10 to 17: the middle of the image, as the model receives it</p><table class="cvf-numtab">' + rows + "</table>";
      var st = pixelStats(px), s = st.sum, zeros = st.zeros, sat = st.sat;
      read.innerHTML = '<div class="mb-metrics">' +
        metric("Label", d.classes[d.gy[cur]], "Fashion-MNIST class " + d.gy[cur], "measured") +
        metric("Numbers in the image", "784", "28 rows x 28 columns, one channel", "measured") +
        metric("Mean value", (s / 784).toFixed(1), "0 is background, 255 is full ink", "measured") +
        metric("Background pixels", zeros, "exactly 0", "measured") +
        metric("Clipped at 255", sat, "information lost to the ceiling", "measured") +
        "</div>";
    }
    render();
    return { show: function (i, o) { cur = i; op = o || "none"; render(); } };
  }

  /* ---------- session 2: convolution ---------- */
  var DEMO_KERNELS = {
    vertical: { label: "vertical edge", k: KERNELS.vertical, div: 4 },
    horizontal: { label: "horizontal edge", k: KERNELS.horizontal, div: 4 },
    blur: { label: "blur (box)", k: [[1, 1, 1], [1, 1, 1], [1, 1, 1]], div: 9 },
    sharpen: { label: "sharpen", k: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]], div: 1 }
  };
  function mountConv(E, rootEl) {
    var d = E.d, cur = 2, kname = "vertical", pool = false;
    var picks = el("div", "cvf-thumbs");
    d.gallery.forEach(function (px, i) {
      if (i % 2) return;
      var b = el("button", "cvf-thumb"); b.type = "button";
      var cv = document.createElement("canvas"); drawGrey(cv, px, 28, 2); b.appendChild(cv);
      b.appendChild(el("span", "", esc(SHORT[d.gy[i]])));
      b.addEventListener("click", function () { cur = i; render(); });
      picks.appendChild(b);
    });
    var bar = el("div", "mb-presets cvf-pickrow");
    Object.keys(DEMO_KERNELS).forEach(function (k) {
      var b = el("button", "rqs-preset", esc(DEMO_KERNELS[k].label)); b.type = "button"; b.dataset.k = k;
      b.addEventListener("click", function () { kname = k; render(); });
      bar.appendChild(b);
    });
    var pb = el("button", "rqs-pad", "2x2 max pool: off"); pb.type = "button";
    pb.addEventListener("click", function () { pool = !pool; render(); });
    bar.appendChild(pb);
    var view = el("div", "cvf-convview");
    var a = document.createElement("canvas"), b2 = document.createElement("canvas");
    var kt = el("div", "cvf-kernel");
    var fa = el("figure", "cvf-panel"); fa.appendChild(a); fa.appendChild(el("figcaption", "", "input, 28 x 28"));
    var fk = el("figure", "cvf-panel"); fk.appendChild(kt); fk.appendChild(el("figcaption", "", "the 3 x 3 kernel"));
    var fb = el("figure", "cvf-panel"); fb.appendChild(b2); var cap = el("figcaption", "", ""); fb.appendChild(cap);
    view.appendChild(fa); view.appendChild(fk); view.appendChild(fb);
    var read = el("div", "cvf-read");
    rootEl.appendChild(picks); rootEl.appendChild(bar); rootEl.appendChild(view); rootEl.appendChild(read);
    function render() {
      [].forEach.call(bar.children, function (b) { if (b.dataset.k) b.classList.toggle("is-on", b.dataset.k === kname); });
      [].forEach.call(picks.children, function (b, k) { b.classList.toggle("is-on", k * 2 === cur); });
      pb.textContent = "2x2 max pool: " + (pool ? "on" : "off"); pb.classList.toggle("is-on", pool);
      var K = DEMO_KERNELS[kname], src = d.gallery[cur], x = new Float64Array(784);
      for (var i = 0; i < 784; i++) x[i] = src[i];
      drawGrey(a, x, 28, 5);
      var cs = convStats(x, kname), out = cs.out, side = 26, mx = cs.max, nz = cs.above40;
      if (pool) { out = maxpool2(out, 26); side = 13; }
      var show = new Float64Array(out.length);
      for (i = 0; i < out.length; i++) show[i] = Math.min(255, out[i]);
      drawGrey(b2, show, side, side === 26 ? 5 : 10);
      cap.textContent = "output, " + side + " x " + side + (pool ? " after pooling" : "");
      kt.innerHTML = '<table class="cvf-numtab cvf-ktab">' + K.k.map(function (row) {
        return "<tr>" + row.map(function (v) { return "<td>" + v + "</td>"; }).join("") + "</tr>";
      }).join("") + "</table><p class=\"cvf-cap\">then divide by " + K.div + "</p>";
      read.innerHTML = '<div class="mb-metrics">' +
        metric("Output size", side + " x " + side, pool ? "26 x 26 halved by the pool" : "28 - 3 + 1, no padding", "measured") +
        metric("Strongest response", mx.toFixed(1), "input runs 0 to 255; the picture caps at 255", "measured") +
        metric("Positions above 40", nz, "of 676 before pooling", "measured") +
        metric("Weights in the kernel", "9", "reused at every position", "measured") +
        "</div>";
    }
    render();
    return { show: function (i, k, p) { cur = i; kname = k; pool = !!p; render(); } };
  }

  /* ---------- session 4: the bench ---------- */
  var RUNGS = [
    { id: "majority", label: "1 · Majority class", note: "Predict the most common of the 300 labels. The floor." },
    { id: "raw", label: "2 · Raw pixels", note: "196 pixel values into softmax regression." },
    { id: "hand", label: "3 · Hand-made features", note: "33 measurements: row and column profiles, ink, box, symmetry." },
    { id: "conv", label: "4 · Fixed convolution features", note: "8 kernels, ReLU, 2x2 max pool: 288 features." },
    { id: "uncertainty", label: "5 · + uncertainty sampling", note: "Same 300 labels, picked 50 at a time where the model is least sure." },
    { id: "diversity", label: "6 · + diversity sampling", note: "Same 300 labels, picked to spread across the feature space." },
    { id: "noisy", label: "7 · Cheap labels", note: "The random 300, three in ten written as the look-alike class.", anti: true }
  ];
  function mountBench(E, rootEl) {
    var cur = "conv", results = {}, base = null;
    var panel = el("div", "mb-presets");
    var btns = {};
    RUNGS.forEach(function (p) {
      var lab = el("label", "mb-preset" + (p.anti ? " is-anti" : ""));
      lab.innerHTML = '<input type="radio" name="cvf-p" value="' + p.id + '">' +
        '<span class="mb-pname">' + esc(p.label) + (p.anti ? ' <em class="mb-anti">the break button</em>' : "") + "</span>" +
        '<span class="mb-pnote">' + esc(p.note) + "</span>";
      lab.querySelector("input").addEventListener("change", function () { setRung(p.id); });
      panel.appendChild(lab); btns[p.id] = lab;
    });
    var hint = el("p", "mb-hint");
    hint.textContent = "Pool: " + comma(E.d.yp.length) + " real Fashion-MNIST training images. Holdout: " + comma(E.d.yh.length) +
      " official test images, never labelled, never picked. Every rung buys 300 labels from the same pool (seed " + SEED +
      ") and trains softmax regression for " + STEPS + " fixed steps. Rung 7 is modelled: the label noise rule is stated, the training and scoring are real.";
    var readout = el("div", "mb-readout"), detail = el("div", "cvf-detail"), strip = el("div", "cvf-strip");
    var curveBox = el("div", "cvf-curvebox");
    rootEl.appendChild(panel); rootEl.appendChild(hint); rootEl.appendChild(readout); rootEl.appendChild(detail);
    rootEl.appendChild(strip); rootEl.appendChild(curveBox);

    function get(id) { if (!results[id]) results[id] = E.rung(id); return results[id]; }
    function verdict(r) {
      var b = base;
      if (r.id === "noisy") return ["bad", "Same images as rung 4, " + r.wrongLabels + " of 300 labels wrong: " + pct1(r.acc) + " on the holdout, " + ((b.acc - r.acc) * 100).toFixed(1) + " points worse than clean labels"];
      if (r.id === "majority") return ["bad", "Always answers " + E.d.classes[r.predicts] + ". " + pct1(r.acc) + " is the floor any model must clear"];
      var dlt = (r.acc - b.acc) * 100;
      if (r.id === "conv") return ["ok", "The reference rung: " + pct1(r.acc) + " on 1,000 unseen images with 300 random labels"];
      return [dlt > 0 ? "good" : "ok", pct1(r.acc) + " on the holdout, " + (dlt >= 0 ? "+" : "") + dlt.toFixed(1) + " points against rung 4 at the same 300 labels"];
    }
    function render() {
      var r = get(cur); base = base || get("conv");
      var g = verdict(r);
      var errs = r.perClassError.map(function (e, c) { return [e, c]; }).sort(function (a, b) { return b[0] - a[0] || a[1] - b[1]; });
      var pairs = [];
      for (var t = 0; t < 10; t++) for (var p = 0; p < 10; p++) if (t !== p && r.confusion[t][p]) pairs.push([r.confusion[t][p], t, p]);
      pairs.sort(function (a, b) { return b[0] - a[0] || a[1] - b[1] || a[2] - b[2]; });
      var top = pairs[0];
      readout.innerHTML = '<div class="mb-verdict is-' + g[0] + '">' + esc(g[1]) + ' <span class="mb-mkind is-measured">measured</span></div>' +
        '<div class="mb-metrics">' +
        metric("Holdout accuracy", pct1(r.acc), r.correct + " of " + comma(E.d.yh.length) + " test images", "measured") +
        metric("Labels paid for", comma(r.labels), "from a pool of " + comma(E.d.yp.length), r.id === "noisy" ? "modelled" : "measured") +
        metric("Worst class", SHORT[errs[0][1]], pct1(errs[0][0]) + " of them wrong", "measured") +
        metric("Biggest confusion", top ? SHORT[top[1]] + " as " + SHORT[top[2]] : "none", top ? top[0] + " test images" : "", "measured") +
        metric("Computed in", r.ms + " ms", "in this browser, this click", "measured") +
        "</div>";
      var rows = errs.map(function (e) {
        var w = Math.round(e[0] * 100);
        return "<tr><th>" + esc(E.d.classes[e[1]]) + "</th><td>" + pct1(e[0]) + '</td><td class="cvf-barcell"><span class="cvf-bar" style="width:' + w + '%"></span></td></tr>';
      }).join("");
      detail.innerHTML = '<table class="dt-table cvf-errtab"><thead><tr><th>Class</th><th>Error</th><th>Share of the class wrong</th></tr></thead><tbody>' + rows + "</tbody></table>";
      strip.innerHTML = "";
      if (r.chosen && r.id !== "majority") {
        var cnt = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]; r.chosen.forEach(function (i) { cnt[E.d.yp[i]]++; });
        strip.appendChild(el("p", "cvf-cap", "The last 40 images this rung paid to label" + (r.id === "uncertainty" ? " (the final batch the model was least sure about)" : "") + ". Labels bought per class: " +
          cnt.map(function (n, c) { return SHORT[c] + " " + n; }).join(" · ")));
        var row = el("div", "cvf-chosen");
        r.chosen.slice(-40).forEach(function (i) {
          var cv = document.createElement("canvas"), x = E.d.xp[i], px = new Float64Array(196);
          for (var k = 0; k < 196; k++) px[k] = x[k] * 255;
          drawGrey(cv, px, 14, 3); cv.title = E.d.classes[E.d.yp[i]]; row.appendChild(cv);
        });
        strip.appendChild(row);
      }
    }
    function setRung(id) {
      cur = id;
      Object.keys(btns).forEach(function (k) { btns[k].classList.toggle("is-on", k === id); btns[k].querySelector("input").checked = k === id; });
      readout.innerHTML = '<p class="mb-hint">Training...</p>';
      setTimeout(render, 20);
    }
    /* the curve: computed on request, one point at a time so the page stays responsive */
    var cbtn = el("button", "rqs-run", "Draw accuracy against labels (21 trainings, a few seconds)"); cbtn.type = "button";
    var cout = el("div", "cvf-curve");
    var seedBtn = el("button", "rqs-pad", "Re-draw the random 300 with five seeds"); seedBtn.type = "button";
    var seedOut = el("p", "cvf-cap");
    curveBox.appendChild(cbtn); curveBox.appendChild(cout); curveBox.appendChild(seedBtn); curveBox.appendChild(seedOut);
    var curve = null;
    cbtn.addEventListener("click", function () {
      cbtn.disabled = true; curve = { random: [], uncertainty: [], diversity: [] };
      var jobs = [];
      CURVE.forEach(function (b) { ["random", "uncertainty", "diversity"].forEach(function (s) { jobs.push([s, b]); }); });
      (function next() {
        if (!jobs.length) { cbtn.disabled = false; cbtn.textContent = "Curve drawn: computed in this browser"; return; }
        var j = jobs.shift(); curve[j[0]].push(E.curvePoint(j[0], j[1])); drawCurve();
        setTimeout(next, 10);
      })();
    });
    seedBtn.addEventListener("click", function () {
      var vals = [7, 8, 9, 10, 11].map(function (s) { return E.randomSeed(s); });
      var lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals);
      seedOut.textContent = "Rung 4 with five different random draws of 300 labels: " + vals.map(function (v) { return (v / 10).toFixed(1) + "%"; }).join(", ") +
        ". Spread " + ((hi - lo) / 10).toFixed(1) + " points from luck alone. Any strategy's gain has to clear this before it means anything.";
    });
    function drawCurve() {
      var W = 640, H = 280, x0 = 56, x1 = 600, y0 = 236, y1 = 24, lo = 600, hi = 820;
      function X(b) { return x0 + (b - 50) / 450 * (x1 - x0); }
      function Y(c) { return y0 - (c - lo) / (hi - lo) * (y0 - y1); }
      var cols = { random: "var(--muted)", uncertainty: "var(--indigo)", diversity: "var(--amber-ink)" };
      var s = '<svg viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="Holdout accuracy against labels bought, for random, uncertainty and diversity sampling, computed in the browser">';
      [600, 650, 700, 750, 800].forEach(function (v) {
        s += '<line x1="' + x0 + '" x2="' + x1 + '" y1="' + Y(v) + '" y2="' + Y(v) + '" stroke="var(--hairline)"/>' +
          '<text x="' + (x0 - 8) + '" y="' + (Y(v) + 4) + '" text-anchor="end" class="cvf-ax">' + (v / 10) + "%</text>";
      });
      CURVE.forEach(function (b) { s += '<text x="' + X(b) + '" y="' + (y0 + 18) + '" text-anchor="middle" class="cvf-ax">' + b + "</text>"; });
      s += '<text x="' + ((x0 + x1) / 2) + '" y="' + (H - 4) + '" text-anchor="middle" class="cvf-ax">labels bought</text>';
      var ly = 40;
      Object.keys(curve).forEach(function (k) {
        var pts = curve[k].map(function (c, i) { return X(CURVE[i]) + "," + Y(c); }).join(" ");
        s += '<polyline points="' + pts + '" fill="none" stroke="' + cols[k] + '" stroke-width="2.5"' + (k === "diversity" ? ' stroke-dasharray="6 4"' : "") + "/>";
        curve[k].forEach(function (c, i) { s += '<circle cx="' + X(CURVE[i]) + '" cy="' + Y(c) + '" r="3.5" fill="' + cols[k] + '"/>'; });
        var last = curve[k].length ? (curve[k][curve[k].length - 1] / 10).toFixed(1) + "%" : "";
        s += '<rect x="' + (x0 + 14) + '" y="' + (ly - 9) + '" width="16" height="4" fill="' + cols[k] + '"/><text x="' + (x0 + 36) + '" y="' + ly + '" class="cvf-ax">' + k + " " + last + "</text>";
        ly += 18;
      });
      s += "</svg>";
      cout.innerHTML = s;
    }
    setRung("conv");
    return { setRung: setRung, rungs: RUNGS.map(function (r) { return r.id; }), result: get, get current() { return cur; } };
  }

  /* ---------- session 5: shift and IoU ---------- */
  var SHIFT_LABELS = [["clean", "as photographed"], ["dim", "lamp dims (x0.6)"], ["flip", "item faces the other way"], ["shift", "2 pixels to the right"], ["rot90", "turned a quarter"]];
  function mountShift(E, rootEl) {
    var W = null, cur = "clean", res = {};
    var bar = el("div", "mb-presets cvf-pickrow");
    SHIFT_LABELS.forEach(function (s) {
      var b = el("button", "rqs-preset", esc(s[1])); b.type = "button"; b.dataset.s = s[0];
      b.addEventListener("click", function () { cur = s[0]; go(); });
      bar.appendChild(b);
    });
    var hint = el("p", "mb-hint", "One model, trained once on all 2,000 pool labels with the fixed convolution features. The holdout images are then changed by the stated array operation and scored again. The shifts are modelled stand-ins for a camera problem; the scoring is real.");
    var read = el("div", "mb-readout"), rel = el("div", "cvf-detail");
    rootEl.appendChild(bar); rootEl.appendChild(hint); rootEl.appendChild(read); rootEl.appendChild(rel);
    function go() {
      [].forEach.call(bar.children, function (b) { b.classList.toggle("is-on", b.dataset.s === cur); });
      read.innerHTML = '<p class="mb-hint">Training on 2,000 labels, then scoring...</p>';
      setTimeout(function () {
        if (!W) W = E.rung("all").W;
        if (!res[cur]) res[cur] = E.shift(cur, W);
        var r = res[cur], acc = r.correct / E.d.yh.length;
        var worst = r.perClassError.map(function (e, c) { return [e, c]; }).sort(function (a, b) { return b[0] - a[0] || a[1] - b[1]; })[0];
        read.innerHTML = '<div class="mb-verdict is-' + (acc > 0.8 ? "good" : acc > 0.5 ? "ok" : "bad") + '">' +
          esc(pct1(acc) + " correct while the model reports " + pct1(r.meanConf) + " average confidence") + ' <span class="mb-mkind is-measured">measured</span></div>' +
          '<div class="mb-metrics">' +
          metric("Holdout accuracy", pct1(acc), r.correct + " of " + comma(E.d.yh.length), "measured") +
          metric("Average confidence", pct1(r.meanConf), "top class probability", "measured") +
          metric("Calibration error (ECE)", r.ece.toFixed(3), "10 bins; 0 is honest", "measured") +
          metric("Worst class", SHORT[worst[1]], pct1(worst[0]) + " wrong", "measured") +
          "</div>";
        var rows = r.bins.map(function (b, k) {
          if (!b.n) return "";
          return "<tr><th>" + (k / 10).toFixed(1) + " to " + ((k + 1) / 10).toFixed(1) + "</th><td>" + b.n + "</td><td>" + pct1(b.conf) + "</td><td>" + pct1(b.acc) + "</td></tr>";
        }).join("");
        rel.innerHTML = '<table class="dt-table"><thead><tr><th>Confidence bin</th><th>Images</th><th>Said</th><th>Was right</th></tr></thead><tbody>' + rows + "</tbody></table>";
      }, 20);
    }
    go();
    return { show: function (s) { cur = s; go(); }, result: function (s) { return res[s]; } };
  }
  function mountIoU(rootEl) {
    var cur = "tight";
    var bar = el("div", "mb-presets cvf-pickrow");
    var names = { tight: "a tight guess", shifted: "slid right by a third", loose: "too loose", small: "too small" };
    Object.keys(IOU_PRESETS).forEach(function (k) {
      var b = el("button", "rqs-preset", esc(names[k])); b.type = "button"; b.dataset.k = k;
      b.addEventListener("click", function () { cur = k; render(); });
      bar.appendChild(b);
    });
    var box = el("div", "cvf-ioubox"), read = el("div", "mb-readout");
    rootEl.appendChild(bar); rootEl.appendChild(box); rootEl.appendChild(read);
    function render() {
      [].forEach.call(bar.children, function (b) { b.classList.toggle("is-on", b.dataset.k === cur); });
      var t = IOU_PRESETS[cur][0], p = IOU_PRESETS[cur][1], v = iou(t, p);
      var ix0 = Math.max(t[0], p[0]), iy0 = Math.max(t[1], p[1]), ix1 = Math.min(t[2], p[2]), iy1 = Math.min(t[3], p[3]);
      var s = '<svg viewBox="0 0 240 200" role="img" aria-label="Truth box and predicted box with their overlap shaded; IoU ' + v.toFixed(3) + '">' +
        '<rect x="0" y="0" width="240" height="200" fill="var(--paper)"/>';
      if (ix1 > ix0 && iy1 > iy0) s += '<rect x="' + ix0 + '" y="' + iy0 + '" width="' + (ix1 - ix0) + '" height="' + (iy1 - iy0) + '" fill="var(--amber)" opacity=".55"/>';
      s += '<rect x="' + t[0] + '" y="' + t[1] + '" width="' + (t[2] - t[0]) + '" height="' + (t[3] - t[1]) + '" fill="none" stroke="var(--ink)" stroke-width="2.5"/>' +
        '<rect x="' + p[0] + '" y="' + p[1] + '" width="' + (p[2] - p[0]) + '" height="' + (p[3] - p[1]) + '" fill="none" stroke="var(--indigo)" stroke-width="2.5" stroke-dasharray="7 4"/></svg>';
      box.innerHTML = s + '<p class="cvf-cap">Solid: the box a person drew. Dashed: the model\'s box. Shaded: where they overlap.</p>';
      var inter = Math.max(0, ix1 - ix0) * Math.max(0, iy1 - iy0);
      var union = (t[2] - t[0]) * (t[3] - t[1]) + (p[2] - p[0]) * (p[3] - p[1]) - inter;
      var pass = v > 0.5;
      read.innerHTML = '<div class="mb-verdict is-' + (pass ? "good" : "bad") + '">IoU ' + v.toFixed(3) + (pass ? ": counts as a hit at the PASCAL VOC threshold (must exceed 0.5)" : ": a miss at the PASCAL VOC threshold, which must be exceeded, not met") + "</div>" +
        '<div class="mb-metrics">' +
        metric("Overlap area", inter, "square pixels", "measured") +
        metric("Union area", union, "both boxes, overlap counted once", "measured") +
        metric("IoU", v.toFixed(3), "overlap divided by union", "measured") +
        metric("Hit at 0.75?", v > 0.75 ? "yes" : "no", "the stricter COCO AP75 bar", "measured") +
        "</div>";
    }
    render();
    return { show: function (k) { cur = k; render(); } };
  }

  function init() {
    if (typeof document === "undefined" || !root.CVF_SAMPLE) return;
    var ids = ["cvf-pixels", "cvf-conv", "cvf-bench", "cvf-shift", "cvf-iou"];
    var any = ids.some(function (i) { return document.getElementById(i); });
    if (!any) return;
    var E = new Engine(root.CVF_SAMPLE), api = { engine: E };
    var m;
    if ((m = document.getElementById("cvf-pixels"))) api.pixels = mountPixels(E, m);
    if ((m = document.getElementById("cvf-conv"))) api.conv = mountConv(E, m);
    if ((m = document.getElementById("cvf-bench"))) api.bench = mountBench(E, m);
    if ((m = document.getElementById("cvf-shift"))) api.shift = mountShift(E, m);
    if ((m = document.getElementById("cvf-iou"))) api.iou = mountIoU(m);
    root.CVF_LIVE = api;
  }

  var exported = { Engine: Engine, canon: canon, iou: iou, mulberry32: mulberry32, permutation: permutation,
                   featConv: featConv, featHand: featHand, convValid: convValid, KERNELS: KERNELS };
  if (typeof module !== "undefined" && module.exports) module.exports = exported;
  if (typeof document !== "undefined") {
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
    else init();
  }
})(typeof window !== "undefined" ? window : this);
