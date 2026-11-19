"""cvf_reference.py - independent Python reference for the label budget bench.

Reads assets/cvf-sample.js, recomputes every number the bench prints (assets/cvf-live.js) with
numpy, and writes the canon as JSON. materials/compare-ladder.py then checks the browser engine
(run in node) against this file by VALUE.

    python3 -I materials/cvf_reference.py assets/cvf-sample.js > /path/to/canon-python.json

Every step mirrors cvf-live.js: same decoding, same features, same seeded generator (mulberry32),
same gradient descent (zero start, fixed steps), same tie rules. Only the arithmetic library differs.
"""
import base64
import json
import sys

import numpy as np

STEPS, LR, L2 = 150, 0.5, 1e-3
BUDGET, SEED_SET, BATCH = 300, 50, 50
CURVE = [50, 100, 150, 200, 300, 400, 500]
NOISE = 0.30
CONFUSE = [6, 3, 4, 0, 2, 7, 0, 5, 2, 7]      # the class a rushed labeller writes instead
KERNELS = {
    "vertical":   [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
    "horizontal": [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
    "diag_down":  [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],
    "diag_up":    [[2, 1, 0], [1, 0, -1], [0, -1, -2]],
}


# ---------- seeded generator, identical to the JS ----------
def mulberry32(seed):
    state = [seed & 0xFFFFFFFF]

    def rnd():
        state[0] = (state[0] + 0x6D2B79F5) & 0xFFFFFFFF
        t = state[0]
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0
    return rnd


def permutation(n, seed):
    rnd = mulberry32(seed)
    a = list(range(n))
    for i in range(n - 1, 0, -1):
        j = int(rnd() * (i + 1))
        a[i], a[j] = a[j], a[i]
    return a


# ---------- data ----------
def load(path):
    txt = open(path).read()
    s = json.loads(txt[txt.index("{"): txt.rindex("}") + 1])

    def unpack(b64, n):
        b = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
        q = np.empty(b.size * 2, dtype=np.uint8)
        q[0::2] = b >> 4
        q[1::2] = b & 15
        return q.reshape(n, 14, 14).astype(np.float64) / 15.0
    return (s, unpack(s["pool"]["px"], s["pool"]["n"]), np.array(s["pool"]["y"]),
            unpack(s["holdout"]["px"], s["holdout"]["n"]), np.array(s["holdout"]["y"]))


# ---------- features ----------
def feat_raw(x):
    return x.reshape(len(x), -1)


def feat_hand(x):
    n = len(x)
    rows = x.mean(axis=2)                     # 14 row profile
    cols = x.mean(axis=1)                     # 14 column profile
    ink = x.reshape(n, -1).mean(axis=1)[:, None]
    on = x >= 0.2
    h = np.zeros((n, 1)); w = np.zeros((n, 1))
    for i in range(n):
        r = np.where(on[i].any(axis=1))[0]; c = np.where(on[i].any(axis=0))[0]
        if r.size:
            h[i, 0] = (r[-1] - r[0] + 1) / 14.0
            w[i, 0] = (c[-1] - c[0] + 1) / 14.0
    sym = np.abs(x - x[:, :, ::-1]).reshape(n, -1).mean(axis=1)[:, None]
    topbot = (x[:, :7, :].reshape(n, -1).mean(axis=1) - x[:, 7:, :].reshape(n, -1).mean(axis=1))[:, None]
    return np.hstack([rows, cols, ink, h, w, sym, topbot])          # 33


def conv_valid(x, k):
    """Cross-correlation, no padding, stride 1: 14x14 -> 12x12 (what libraries call convolution)."""
    k = np.array(k, dtype=np.float64)
    out = np.zeros((len(x), 12, 12))
    for a in range(3):
        for b in range(3):
            out += k[a, b] * x[:, a:a + 12, b:b + 12]
    return out


def maxpool2(m):
    n = len(m)
    return m.reshape(n, 6, 2, 6, 2).max(axis=(2, 4))


def feat_conv(x):
    """Eight fixed kernels (four edge directions and their mirrors), ReLU, 2x2 max pool."""
    parts = []
    for k in KERNELS.values():
        r = conv_valid(x, k) / 4.0
        parts.append(maxpool2(np.maximum(r, 0)).reshape(len(x), -1))     # dark-to-light edges
        parts.append(maxpool2(np.maximum(-r, 0)).reshape(len(x), -1))    # light-to-dark edges
    return np.hstack(parts)                     # 8 x 36 = 288


# ---------- standardise on the unlabelled pool, then softmax regression ----------
def standardiser(fpool):
    mu = fpool.mean(axis=0)
    sd = fpool.std(axis=0)
    sd[sd == 0] = 1.0
    return mu, sd


def design(f, mu, sd):
    z = (f - mu) / sd
    return np.hstack([z, np.ones((len(z), 1))])


def train(Z, y, steps=STEPS):
    n, d = Z.shape
    W = np.zeros((d, 10))
    Y = np.zeros((n, 10)); Y[np.arange(n), y] = 1.0
    for _ in range(steps):
        P = softmax(Z @ W)
        G = Z.T @ (P - Y) / n
        G[:-1] += L2 * W[:-1]
        W -= LR * G
    return W


def softmax(L):
    L = L - L.max(axis=1, keepdims=True)
    E = np.exp(L)
    return E / E.sum(axis=1, keepdims=True)


def predict(W, Z):
    return softmax(Z @ W)


def evaluate(W, Zh, yh):
    P = predict(W, Zh)
    pred = P.argmax(axis=1)
    correct = int((pred == yh).sum())
    per = [round(float((pred[yh == c] != c).mean()), 6) for c in range(10)]
    conf = np.zeros((10, 10), dtype=int)
    for t, p in zip(yh, pred):
        conf[t, p] += 1
    return {"correct": correct, "acc": correct / len(yh), "perClassError": per,
            "confusion": conf.tolist(), "pred": pred}


# ---------- choosing which images to label ----------
def pick_random(n_pool, budget, seed):
    return permutation(n_pool, seed)[:budget]


def pick_uncertainty(Zp, yp, budget, seed):
    chosen = permutation(len(Zp), seed)[:SEED_SET]
    while len(chosen) < budget:
        W = train(Zp[chosen], yp[chosen])
        P = predict(W, Zp)
        s = np.sort(P, axis=1)
        margin = np.round(s[:, -1] - s[:, -2], 9)
        taken = set(chosen)
        cand = [i for i in range(len(Zp)) if i not in taken]
        cand.sort(key=lambda i: (margin[i], i))
        chosen = chosen + cand[:min(BATCH, budget - len(chosen))]
    return chosen


def pick_diversity(Zp, budget, seed):
    """k-centre greedy (Sener and Savarese 2018) from the same random seed set."""
    X = Zp[:, :-1]
    chosen = permutation(len(Zp), seed)[:SEED_SET]
    sq = (X * X).sum(axis=1)
    dmin = np.full(len(X), np.inf)
    for c in chosen:
        d = sq + sq[c] - 2 * X @ X[c]
        dmin = np.minimum(dmin, np.round(d, 6))
    while len(chosen) < budget:
        dm = dmin.copy(); dm[chosen] = -1
        best = int(np.argmax(dm))              # first index on a tie
        chosen.append(best)
        d = sq + sq[best] - 2 * X @ X[best]
        dmin = np.minimum(dmin, np.round(d, 6))
    return chosen


def noisy(labels, seed):
    rnd = mulberry32(seed)
    out = list(labels)
    for i in range(len(out)):
        if rnd() < NOISE:
            out[i] = CONFUSE[out[i]]
    return out


# ---------- session 5: shift, calibration, IoU ----------
SHIFTS = {
    "clean": lambda x: x,
    "dim": lambda x: x * 0.6,                                   # the lamp over the belt dims
    "flip": lambda x: x[:, :, ::-1],                            # items arrive facing the other way
    "shift": lambda x: np.concatenate([np.zeros((len(x), 14, 2)), x[:, :, :12]], axis=2),  # 2 px right
    "rot90": lambda x: np.rot90(x, k=-1, axes=(1, 2)),          # items arrive turned a quarter
}


def ece(P, y, bins=10):
    conf = P.max(axis=1); pred = P.argmax(axis=1)
    b = np.minimum(bins - 1, np.floor(conf * bins).astype(int))
    tot = 0.0
    for k in range(bins):
        m = b == k
        if m.any():
            tot += m.sum() / len(y) * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(tot)


def iou(a, b):
    """Boxes as (x0, y0, x1, y1). PASCAL VOC overlap: intersection over union."""
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


IOU_PRESETS = {                              # truth box, then a predicted box, in pixels
    "tight":   ((40, 30, 160, 150), (46, 34, 164, 152)),
    "shifted": ((40, 30, 160, 150), (80, 30, 200, 150)),
    "loose":   ((40, 30, 160, 150), (20, 10, 190, 180)),
    "small":   ((40, 30, 160, 150), (70, 60, 130, 120)),
}


# ---------- sessions 1 and 2: array operations and hand-made kernels on 28x28 ----------
DEMO_KERNELS = {
    "vertical": (KERNELS["vertical"], 4),
    "horizontal": (KERNELS["horizontal"], 4),
    "blur": ([[1, 1, 1], [1, 1, 1], [1, 1, 1]], 9),
    "sharpen": ([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], 1),
}


def apply_op(img, op):
    o = img.copy()
    if op == "bright":
        o = np.minimum(255, o + 60)
    if op == "contrast":
        o = np.clip(np.floor((o - 128) * 1.5 + 128 + 0.5), 0, 255)
    if op == "invert":
        o = 255 - o
    if op == "flip":
        o = o[:, ::-1]
    return o


def conv_valid28(img, k):
    k = np.array(k, dtype=np.float64)
    out = np.zeros((26, 26))
    for a in range(3):
        for b in range(3):
            out += k[a, b] * img[a:a + 26, b:b + 26]
    return out


# ---------- the ladder ----------
def run(sample_path):
    s, xp, yp, xh, yh = load(sample_path)
    feats = {"raw": feat_raw, "hand": feat_hand, "conv": feat_conv}
    Z = {}
    for k, fn in feats.items():
        fp, fh = fn(xp), fn(xh)
        mu, sd = standardiser(fp)
        Z[k] = (design(fp, mu, sd), design(fh, mu, sd))

    SEED = 7
    rand = pick_random(len(yp), BUDGET, SEED)
    out = {"budget": BUDGET, "rungs": {}}

    major = int(np.bincount(yp[rand], minlength=10).argmax())
    out["rungs"]["majority"] = {"correct": int((yh == major).sum()), "acc": float((yh == major).mean()),
                                "predicts": major}

    def rung(key, fkey, idx, labels):
        Zp, Zh = Z[fkey]
        W = train(Zp[idx], np.array(labels))
        e = evaluate(W, Zh, yh)
        e.pop("pred")
        out["rungs"][key] = e

    rung("raw", "raw", rand, yp[rand])
    rung("hand", "hand", rand, yp[rand])
    rung("conv", "conv", rand, yp[rand])
    unc = pick_uncertainty(Z["conv"][0], yp, BUDGET, SEED)
    rung("uncertainty", "conv", unc, yp[unc])
    div = pick_diversity(Z["conv"][0], BUDGET, SEED)
    rung("diversity", "conv", div, yp[div])
    rung("noisy", "conv", rand, noisy(yp[rand], SEED + 1))
    full = list(range(len(yp)))
    rung("all", "conv", full, yp[full])

    # learning curves on conv features, same seed
    curves = {"random": [], "uncertainty": [], "diversity": []}
    for b in CURVE:
        for name, idx in (("random", pick_random(len(yp), b, SEED)),
                          ("uncertainty", pick_uncertainty(Z["conv"][0], yp, b, SEED)),
                          ("diversity", pick_diversity(Z["conv"][0], b, SEED))):
            W = train(Z["conv"][0][idx], yp[idx])
            curves[name].append(evaluate(W, Z["conv"][1], yh)["correct"])
    out["curve"] = {"budgets": CURVE, **curves}

    # seed spread for random at the bench budget (how much is luck)
    out["randomSeeds"] = []
    for sd_ in (7, 8, 9, 10, 11):
        idx = pick_random(len(yp), BUDGET, sd_)
        W = train(Z["conv"][0][idx], yp[idx])
        out["randomSeeds"].append(evaluate(W, Z["conv"][1], yh)["correct"])
    # session 5: the all-labels model under shift, slices and calibration
    W = train(Z["conv"][0], yp)
    out["shift"] = {}
    for name, fn in SHIFTS.items():
        xs = fn(xh)
        Zs = design(feat_conv(xs), *standardiser(feat_conv(xp)))
        P = predict(W, Zs)
        pred = P.argmax(axis=1)
        out["shift"][name] = {"correct": int((pred == yh).sum()),
                              "meanConf": round(float(P.max(axis=1).mean()), 6),
                              "ece": round(ece(P, yh), 6),
                              "perClassError": [round(float((pred[yh == c] != c).mean()), 6) for c in range(10)]}
    ink = xh.reshape(len(xh), -1).mean(axis=1)
    q1 = np.sort(ink)[len(ink) // 4 - 1]
    P = predict(W, Z["conv"][1]); pred = P.argmax(axis=1)
    faint = ink <= q1
    out["inkSlice"] = {"faintN": int(faint.sum()), "faintCorrect": int((pred[faint] == yh[faint]).sum()),
                       "restN": int((~faint).sum()), "restCorrect": int((pred[~faint] == yh[~faint]).sum()),
                       "cut": round(float(q1), 6)}
    out["iou"] = {k: round(iou(*v), 6) for k, v in IOU_PRESETS.items()}
    # sessions 1 and 2: the 20 full-size gallery images, pixel and convolution readouts
    gal = np.frombuffer(base64.b64decode(s["gallery"]["px"]), dtype=np.uint8).reshape(-1, 28, 28).astype(np.float64)
    out["gallery"] = []
    for img, lab in zip(gal, s["gallery"]["y"]):
        g = {"label": lab}
        for op in ("none", "bright", "contrast", "flip", "invert"):
            o = apply_op(img, op)
            g[op] = [round(float(o.mean()), 6), int((o == 0).sum()), int((o == 255).sum())]
        for kn, (k, kdiv) in DEMO_KERNELS.items():
            r = np.abs(conv_valid28(img, k) / kdiv)
            g["k_" + kn] = [round(float(r.max()), 6), int((r > 40).sum())]
        out["gallery"].append(g)
    out["labelledClassCounts"] = {
        "random": np.bincount(yp[rand], minlength=10).tolist(),
        "uncertainty": np.bincount(yp[unc], minlength=10).tolist(),
        "diversity": np.bincount(yp[div], minlength=10).tolist(),
    }
    return out


if __name__ == "__main__":
    print(json.dumps(run(sys.argv[1]), indent=1))
