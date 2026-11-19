# Agent brief - shared by every fan-out page of learn-computer-vision-with-phoebe

Filled from course-builder `references/agent-brief.template.md` on 2026-11-19. Internal build document:
never link it from an audience page.

You are writing ONE static HTML session page. No servers, no npm. **If your target file already
exists on disk, do not write it; report that and stop.** Write the file, return its path and one
line of coverage. No HTML in your reply.

## Read first, in this order

1. The template pages. Copy their structure, classes, SVG grammar and quiz markup EXACTLY,
   including FOUR options per question (A to D):
   - Session template: `/Users/phoebe.fu/Documents/claude_work/github_repo/learn-computer-vision-with-phoebe/courses/01-an-image-is-a-grid-of-numbers.html`
   - Widget + bench page (read for how a widget mount sits on a page and how the scripts are
     loaded): `/Users/phoebe.fu/Documents/claude_work/github_repo/learn-computer-vision-with-phoebe/courses/04-the-label-budget-bench.html`
2. The source map: every verified number, its evidence tier, per-session coverage, the seams. Use
   ONLY its numbers; never invent a statistic; if a fact is missing, teach the uncertainty.
   `/Users/phoebe.fu/Documents/claude_work/github_repo/learn-computer-vision-with-phoebe/materials/official-course-map.md`
3. The stylesheet `:root` block for the palette tokens (first 30 lines only):
   `/Users/phoebe.fu/Documents/claude_work/github_repo/learn-computer-vision-with-phoebe/assets/style.css`
4. For widget behaviour only (what each mount point prints), skim the UI half of
   `/Users/phoebe.fu/Documents/claude_work/github_repo/learn-computer-vision-with-phoebe/assets/cvf-live.js`
   (search "mountConv", "mountShift", "mountIoU"). Do NOT edit any asset.

## Page skeleton (keep every component)

toolbar (crumb EXACTLY "learn-computer-vision-with-phoebe / Session N of 6" with the repo name linked to
`../index.html`, #toggle-all, #zoom-toggle) · masthead (eyebrow "Learn Computer Vision Foundations with
Phoebe · Session N of 6", h1 with one `<span class="accent">`, .sub, .chip-row with the level chip
(🟢 Foundations / 🟡 Core / 🟠 Applied / 🔴 Advanced as given in your outline), two .chip.audience, .chip.time "45 min",
.agenda a1-a4) · main.wrap · section#intro (Part 0: kicker, .lede, .legend pills copied from session 1,
.callout.win "★ What you walk out with tonight.") · 3 Parts, each `section.section#part-N` with
section-kicker (klabel "Part N · covers ...", h2, `.tag.concept "N min live"`), a `.lede`, ONE
figure, `details.card` accordions (summary: `.mode.live` or `.mode.self`, title, `.mini`,
`.caret ▶`), at least one `.callout.example` with `span.ex-pill` "Real world" on the page ·
section#demo-1 Build-along (kicker klabel "Build-along", `.tag.demo "★ 22 min · everyone builds"`, .lede, ONE figure,
`.steps > .step`, each with a `.prompt-box.good` carrying a `span.label`) · section#exercise Homework
(ol, 4 items) · section#quiz (3 x `.quiz-q data-answer="0-based"`, `p.qtext`, `button.qopt` "A · ..."
FOUR options, `p.qwhy`; one `p.quiz-score` after the last; vary the correct letter) · section#official, h2 EXACTLY
"What this session teaches, and where it came from", `.covered > .covered-row` (pill solid ✓ / light ◐ +
name + note), then the `.mono` line EXACTLY "Every fact on this page, and its verification tier, is
recorded in the course's source map." · section.cheat#cheatsheet (h3 "Session N cheat sheet <span>· pin
this</span>", .grid-2 of six .cheat-item) · `.callout.next` with `.nx-pill` "Next session" ·
footer.pagefoot · scripts.

Scripts at the end of body: if your page mounts a widget, EXACTLY
`<script src="../assets/cvf-sample.js?v=1"></script>` `<script src="../assets/cvf-live.js?v=2"></script>`
`<script src="../assets/app.js?v=2"></script>`; otherwise only `<script src="../assets/app.js?v=1"></script>`.

Head: the session 1 social meta block with this page's own title/description/url;
`<title>Session N · <Title> - learn computer vision with phoebe</title>`;
`<link rel="stylesheet" href="../assets/style.css?v=2">`. Nothing else external.

First `details.card` in the FIRST Part is `open`; no other. Sentence case headings. Warm
practitioner voice, concrete, never dry. Inside prompt-boxes escape `&` `<` `>`. 450 to 750 lines
is guidance about depth, never a target.

## Build-along code must be REAL

Every printed output in a build-along prompt-box must be what the code actually printed. Run your
code before writing the page:
- Data: the four Fashion-MNIST `.gz` files are in `/private/tmp/claude-504/cvf/data-fmnist/` (read-only,
  untrusted download: read it only with `python3 -I`, never run anything from inside that folder).
- Work in your own empty folder `/private/tmp/claude-504/cvf/run-sN/` (N = your session); run with
  `python3 -I script.py`. numpy 1.26 and scikit-learn 1.9 are installed. Keep each run under ~2 minutes
  (subsample with a fixed seed and say so in the code).
- On the page the code loads from `data/...gz` (learner's folder), as session 1 step 1 does; reuse the
  session 1 loader verbatim rather than inventing another.
- Numbers from your own runs are build-along numbers: they belong in prompt-box comments and the step
  text that introduces them. Never mix them up with the bench canon (which is computed on the 14 x 14
  bench sample, not the full set); say which is which.

## Hard rules (a violation is rework)

- NEVER an em dash or en dash, anywhere (prose, code, aria-labels, comments). Hyphen only.
- No meta text: never "this course", "in this course", "the course teaches", "banned here". State
  the professional norm directly with its reason. The two exact estate phrases above are the only
  self-references; "session 5" cross-references are fine.
- Attribution "by Phoebe Fu". Never "built with" a tool.
- Every number comes from the map, from your own real run, or is labelled constructed. For constructed
  data print "your numbers will differ", never invented outputs as if run.
- Contested or missing evidence: teach the disagreement; never resolve what the literature has not.
- Citations in the exact form of the map's appendix; anything marked "reported" is said "reported".
  Szeliski: further reading only, no claim attributed to it. Never claim a Fashion-MNIST label error rate.
  Lowell et al.: never "worse than random"; say "not consistently better than i.i.d. sampling".
- NEVER "lottery" or "lotteries"; say the mechanism ("decided by row order", "a random draw").
- Default to the English word. Any Chinese term carries its English in brackets.
- No cross marks anywhere (✗ ✘ ✕ ❌ ×): write "x" as the letter for dimensions ("28 x 28"); label a
  wrong state in words.
- Titles, widget ids and class names must not collide with siblings: do not reuse any h1/h2 from
  learn-deep-learning 04 ("Seeing and embedding", "From loops to architectures", "LeNet's three moves",
  "AlexNet and ResNet: the two numbers everyone misquotes", "Embeddings: the layer you actually ship",
  "Count what convolution saves"), learn-autonomous-driving p2/p5, learn-intro-ml b7. Widget ids are
  fixed: `cvf-conv` (s2), `cvf-shift` and `cvf-iou` (s5); no other ids starting `cvf-`.
- Running case: Corvane, a CONSTRUCTED recycling plant; write "constructed Corvane example" on every
  Corvane passage or prompt-box label. The images are Fashion-MNIST garments standing in for textiles
  on the line: say so plainly, never present them as waste photos. Never mention Phoebe's employers,
  clients or projects; no other company case names except published sources.
- Seams: never reteach a sibling's topic; link it once with the absolute URL from the map's seams table.
- Light surfaces only. No dark backgrounds anywhere.

## Figure grammar (hand-drawn, every figure)

Palette, ONLY these hexes (no invented greys): `#AD008B` accent · `#6A0055` deep · `#C7339F` mid ·
`#EDB3DD` soft · `#FCEEF8` accent-50 · `#26121F` ink · `#5E4A57` muted · `#D6C4CF` faint · `#EFE4EB`
hairline · `#5FD3DC` cyan (the ONE contrast fill, never text) · `#005961` cyan-ink (text and strokes
on the cyan side) · `#E6F8F9` cyan-50 · `#FFFBFE` paper · `#FFFFFF` · universal reds `#991B1B`
`#FEF2F2` `#FCA5A5` only for a wrong-way panel.

- `<figure class="zoomable">` > `<svg viewBox="0 0 880 H" xmlns="http://www.w3.org/2000/svg" role="img"
  aria-label="the data, not the shape">` > `<defs>` + `<style>` + content, then
  `<figcaption>🔍 Click to zoom - takeaway</figcaption>`. Grow H, never W.
- Prefix unique per figure, used for every class and id: `s<N><letter>` (s2a, s2b, s2c, s2d for
  session 2's four figures; s3a...; s5a...; s6a...). Session 1 used s1a-s1d, session 4 s4a-s4d.
- `<defs>` holds with the prefix P: a wobble filter `id="PSk"` (copy session 1's exactly, change the
  seed), a hachure pattern `id="PHc"` (copy session 1's), an open arrowhead marker `id="PAr"` (copy
  session 1's). ALL shapes sit inside ONE `<g filter="url(#PSk)" fill="none" stroke="#26121F"
  stroke-width="2" stroke-linecap="round" stroke-linejoin="round">`; rects carry a tiny rotation (-4 to
  4 degrees for hand-placed items, under 1 for panels). Fills: white, accent-50, the hachure for "the
  pile" or "the data", and cyan ONLY for the one thing the figure is about. Text classes: `.PH` 800 12px
  ink heading · `.PL` 600 12px ink label · `.PS` 400 11px muted · `.PB` 800 11px cyan-ink · `.PV` 800
  16-20px deep value · `.PW` 800 12px white on a fill · `.PA` 700 11px accent caption · `.PN` 400 12px
  muted note.
- ALL `<text>` outside any filtered group, Inter sans stack, never below 10.5px.
- Fit: max chars about (box width - 20) / 7 at 12px, 6.4px per char at 11px; full-width note under 110
  chars; 40px between neighbouring point labels; bottom note 22px below the last row, H clears it by 8px.
  When in doubt, shorten. No line, arrow or curve may cross a label. Hand-placed rects must not overlap
  each other (the gate flags a pile): place stacks side by side.
- Floor: one figure per Part plus one in the build-along (four figures). Draw the MECHANISM (a kernel's
  nine weights over a pixel patch and the sum it produces; a confusion table with its biggest off-diagonal
  cells; a box overlap shaded with intersection and union), never a metaphor literally, never decoration.

## Voice and honesty

Every Part gets a real-world story: a published source from the map, or the constructed Corvane case
labelled as such. Bench and widget numbers on the page must be numbers the widget actually prints (they
are in the map's canon section). Teach the active learning disagreement (Settles' own caveats, Lowell et
al., Sener and Savarese, Beck et al., Munjal et al.) as a disagreement.

## Cross-links (absolute URLs)

- Deep Learning 04 (architectures, LeNet, what convolution saves): https://phoebefu6.github.io/learn-deep-learning-with-phoebe/courses/04-seeing-and-embedding.html
- Deep Learning 02 backprop: https://phoebefu6.github.io/learn-deep-learning-with-phoebe/courses/02-backprop-by-hand.html
- Deep Learning 03 training loop: https://phoebefu6.github.io/learn-deep-learning-with-phoebe/courses/03-the-training-loop.html
- Intro ML b7 classification metrics: https://phoebefu6.github.io/learn-intro-ml-with-phoebe/courses/b7-classification-metrics.html
- Model Evaluation b3 curves and calibration: https://phoebefu6.github.io/learn-model-evaluation-with-phoebe/courses/b3-curves-calibration.html
- Autonomous Driving p2 label cost: https://phoebefu6.github.io/learn-autonomous-driving-with-phoebe/courses/p2-labels-cost-by-the-frame.html
- Autonomous Driving p5 perception metrics: https://phoebefu6.github.io/learn-autonomous-driving-with-phoebe/courses/p5-perception-metrics-that-hide-failures.html
- Anomaly Detection: https://phoebefu6.github.io/learn-anomaly-detection-with-phoebe/
- AI Media b7 generated images: https://phoebefu6.github.io/learn-ai-media-domain-with-phoebe/courses/b7-images-and-synthetic-media.html
- Hub: https://phoebefu6.github.io/learn-with-phoebe/

## Footer chain and session titles

Footer left: "Session N of 6 · learn-computer-vision-with-phoebe · by Phoebe Fu &nbsp;·&nbsp; 📚 <a href="https://phoebefu6.github.io/learn-with-phoebe/">Learn with Phoebe ↗</a>"
Footer right: "← Prev: <short title>" and "Next: <short title> →" (session 6: "← Prev" and "Course home →" to `../index.html`).

Session files, titles (exact h1 text, sentence case, one accent span) and short titles for prev/next:
1. `01-an-image-is-a-grid-of-numbers.html` · An image is a grid of numbers · short "An image is a grid of numbers"
2. `02-slide-a-small-window.html` · Slide a small window: convolution · short "Slide a small window"
3. `03-a-first-classifier.html` · A first classifier and what it confuses · short "A first classifier"
4. `04-the-label-budget-bench.html` · The label budget bench · short "The label budget bench"
5. `05-where-it-fails.html` · Where it fails: slices and shift · short "Where it fails"
6. `06-scope-the-vision-feature.html` · Scope the vision feature · short "Scope the vision feature"

Corvane's vision plan grows one page per session: 1 image card · 2 feature choice · 3 baseline classifier
and error table · 4 labelling budget plan · 5 evaluation and failure slices · 6 signed scope note.
