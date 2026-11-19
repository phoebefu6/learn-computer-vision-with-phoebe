# Source map - learn-computer-vision-with-phoebe ("Computer Vision Foundations")

Internal build document. Never linked from an audience page. Built 2026-11-19 by Phoebe Fu's course lead.
Single track, 6 sessions x 45 min, session 4 = the bench. Running case: Corvane (constructed recycling
plant), label "Constructed Corvane example" every time. Teaching images are Fashion-MNIST garments
standing in for "textiles on the line", said plainly; never presented as waste photos.

Evidence tiers: **read-at-source** = the fact was read in the primary document during this build.
**reported** = seen in a secondary listing or search result only; pages must say "reported" or cut it.
**computed** = produced by `assets/cvf-live.js` in the browser AND by `materials/cvf_reference.py`
in numpy, compared by value (`materials/compare-ladder.py`, 1,405 values, 0 mismatches; a planted
0.1% fault fails the comparer).

---

## 1. Canon numbers (computed; the only numbers pages may quote for the bench)

Sample (`materials/cvf_build_sample.py`, seed 20170825): pool 2,000 of the 60,000 Fashion-MNIST
training images; holdout 1,000 of the 10,000 official test images; both pooled 2x2 to 14x14 and
quantised to 16 grey levels (two pixels per byte). Gallery: 20 test images at the original 28x28,
256 levels (two per class). `assets/cvf-sample.js` is 436 KB.

Pool class counts: 188, 165, 212, 193, 200, 223, 245, 177, 185, 212 (T-shirt ... Ankle boot).
Holdout class counts: 117, 104, 101, 90, 95, 84, 98, 105, 101, 105.

Model: softmax regression, features standardised on the unlabelled pool, zero start, 150 full-batch
gradient steps, learning rate 0.5, L2 0.001. Seeded generator mulberry32, seed 7 (noise seed 8).

### The ladder (session 4), every rung at 300 labels, holdout 1,000

| Rung | Correct | Accuracy | Notes |
|---|---|---|---|
| 1 majority class | 98 | 9.8% | the 300 random labels hold 43 Shirts, so it always says Shirt |
| 2 raw pixels (196) | 748 | 74.8% | |
| 3 hand-made features (33) | 694 | 69.4% | worse than raw: the measurements threw information away |
| 4 fixed convolution features (8 kernels, ReLU, 2x2 max pool, 288) | 758 | 75.8% | the reference rung |
| 5 + uncertainty sampling (margin, batches of 50 from the same first 50) | 776 | 77.6% | +1.8 points vs rung 4 |
| 6 + diversity sampling (k-centre greedy) | 727 | 72.7% | -3.1 points vs rung 4: worse than random here |
| 7 ANTI cheap labels (30% rule, look-alike class) | 487 | 48.7% | 27.1 points worse; modelled noise, real training |
| ceiling: conv on all 2,000 labels | 825 | 82.5% | |

Random-label luck (rung 4 with seeds 7, 8, 9, 10, 11): 758, 756, 761, 776, 776, i.e. 75.6% to 77.6%,
a 2.0-point spread. Uncertainty's +1.8 sits INSIDE that spread: the bench's own contested-evidence lesson.

Labels bought per class at 300 (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Boot):
- random 29, 26, 27, 20, 31, 27, 43, 40, 27, 30
- uncertainty 27, 11, 44, 35, 48, 22, 47, 22, 24, 20 (it buys Pullover, Coat, Shirt; skips Trouser)
- diversity 37, 7, 25, 15, 8, 80, 29, 11, 60, 28 (it buys 80 Sandals and 60 Bags: the odd shapes far from everything)

Learning curve, correct of 1,000 at 50 / 100 / 150 / 200 / 300 / 400 / 500 labels (conv features):
- random 629, 724, 733, 753, 758, 757, 778
- uncertainty 629, 701, 719, 767, 776, 806, 812
- diversity 629, 699, 713, 715, 727, 734, 752
Cold start: uncertainty is BEHIND random at 100 and 150 labels, ahead from 200; at 400 it leads by 4.9 points.
Accuracy per hundred labels from 50 to 300: random (75.8 - 62.9) / 2.5 = 5.2 points; uncertainty 5.9; diversity 3.9.

Per-class error at rung 4 (conv, 300 random): T-shirt 30.8%, Trouser 10.6%, Pullover 41.6%, Dress 30.0%,
Coat 40.0%, Sandal 13.1%, Shirt 35.7%, Sneaker 16.2%, Bag 15.8%, Boot 8.6%.
Biggest confusions at rung 4: T-shirt read as Shirt 21; Pullover as Coat 19; Coat as Shirt 16; Coat as Pullover 16.
Per-class error at rung 2 (raw): Shirt 49.0%, Pullover 50.5%, Coat 34.7%, Boot 4.8%.
Ceiling (all 2,000): Shirt 36.7%, Pullover 32.7%, Coat 31.6%, Dress 23.3%, T-shirt 17.9%, Sneaker 12.4%,
Trouser 5.8%, Boot 5.7%, Bag 5.0%, Sandal 4.8%. Biggest confusions: Pullover as Coat 16; Shirt as T-shirt 15; Coat as Shirt 15.

### Session 5: the all-2,000 model under shift (modelled shifts, real scoring)

| Shift (array operation) | Correct | Mean confidence | ECE (10 bins) | Notes |
|---|---|---|---|---|
| clean | 825 | 82.6% | 0.022 | well calibrated |
| dim (x0.6) | 813 | 71.7% | 0.096 | accuracy holds, confidence drops: underconfident |
| flip left-right | 565 | 81.9% | 0.254 | Sneaker 100% wrong, Ankle boot 100%, Sandal 81.0%: shoes face one way |
| shift 2 px right (of 14) | 361 | 64.5% | 0.284 | Pullover and Shirt 100% wrong |
| rotate a quarter turn | 51 | 83.7% | 0.786 | 5.1%, below the 10% of guessing, at 83.7% confidence |

Faint-item slice (lowest quarter of ink, mean <= 0.189): 215 of 250 correct (86.0%) vs 610 of 750 (81.3%).

IoU presets (truth box (40,30)-(160,150)): tight 0.876; slid right by a third 0.500 (fails VOC's
"must exceed 0.5"); too loose 0.498 (fails); too small 0.250.

### Sessions 1 and 2: the gallery (28x28, 256 levels)

Gallery index 0 (T-shirt): mean 36.6, 303 background pixels (exactly 0), 1 at 255.
Brighter +60: mean 96.4, 0 background, 4 clipped. Index 8 (Coat): mean 146.6; brighter clips 475 of 784.
Index 2 (Trouser): vertical-edge kernel strongest 255.0, 228 positions above 40; horizontal kernel 120.5 and 44.
Index 14 (Sneaker): vertical 72 positions above 40, horizontal 182: the sole is a horizontal edge.
Sharpen on index 0 peaks at 824.0: outside the 0 to 255 range the input lives in.

Full gallery canon (the ten thumbnails the s1 and s2 widgets offer; the widget prints these exact values,
one decimal; "> 40" counts the 676 positions of the 26 x 26 output, before pooling; each kernel's output is
divided as the widget states: edges by 4, blur by 9, sharpen by 1, absolute value):

| widget thumb | t10k index | class | mean | background (=0) | at 255 | bright+60 mean/bg/clipped | vertical max, >40 | horizontal max, >40 | blur max, >40 | sharpen max, >40 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 164 | T-shirt/top | 36.6 | 303 | 1 | 96.4 / 0 / 4 | 178.0, 146 | 190.8, 91 | 162.1, 315 | 824.0, 329 |
| 2 | 2 | Trouser | 65.7 | 524 | 40 | 115.8 / 0 / 184 | 255.0, 228 | 120.5, 44 | 247.2, 294 | 680.0, 308 |
| 4 | 191 | Pullover | 36.5 | 263 | 1 | 96.2 / 0 / 6 | 208.2, 198 | 171.8, 63 | 137.7, 324 | 683.0, 380 |
| 6 | 13 | Dress | 55.4 | 542 | 3 | 111.6 / 0 / 166 | 255.0, 135 | 218.0, 35 | 225.3, 258 | 652.0, 268 |
| 8 | 98 | Coat | 146.6 | 219 | 15 | 188.6 / 0 / 475 | 255.0, 152 | 236.8, 99 | 233.1, 561 | 686.0, 573 |
| 10 | 63 | Sandal | 24.6 | 550 | 1 | 84.3 / 0 / 8 | 214.0, 88 | 195.2, 193 | 133.7, 198 | 854.0, 241 |
| 12 | 135 | Shirt | 51.2 | 290 | 2 | 110.3 / 0 / 28 | 206.2, 282 | 168.8, 121 | 216.8, 438 | 770.0, 445 |
| 14 | 230 | Sneaker | 34.9 | 535 | 1 | 94.3 / 0 / 28 | 200.0, 72 | 203.5, 182 | 199.8, 230 | 777.0, 250 |
| 16 | 58 | Bag | 89.2 | 415 | 4 | 142.0 / 0 / 279 | 232.5, 69 | 255.0, 167 | 229.2, 404 | 790.0, 405 |
| 18 | 163 | Ankle boot | 39.1 | 522 | 1 | 97.8 / 0 / 36 | 189.0, 92 | 247.8, 169 | 217.2, 258 | 721.0, 259 |

Cheap-labels rung: 99 of the 300 labels wrong (the 30% rule). Every per-class error rises; Sneaker 16.2% -> 61.9%.

---

## 2. Sources (citation appendix; pages cite in this form)

1. **Xiao, Rasul, Vollgraf (2017).** Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning
   Algorithms. arXiv:1708.07747. Repo github.com/zalandoresearch/fashion-mnist. MIT licence. **read-at-source**
   - 70,000 greyscale 28x28 images of fashion products, 10 classes, 7,000 per class; 60,000 train, 10,000 test.
   - Labels 0-9: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot.
   - Intended as a "direct drop-in replacement for the original MNIST dataset" (same size, format, splits).
   - README: "MNIST is too easy", "Convolutional nets can achieve 99.7% on MNIST", "MNIST is overused".
   - Pipeline (paper section 2): front-look thumbnails of Zalando products; trim, resize longest edge to 28,
     sharpen, extend and centre, NEGATE intensities, convert to 8-bit greyscale. So 0 is background, high is ink.
   - Labels from the product "silhouette code", labelled by in-house fashion experts, reviewed by a separate team.
   - Paper Table 3 (test accuracy, scikit-learn, full 60,000 train): LogisticRegression 0.842; KNeighbors 0.854;
     MLPClassifier 0.871; RandomForest 0.873; GradientBoosting 0.880; SVC rbf 0.897 (best of the table).
   - README human performance 0.835 (crowd, 1,000 test images, 3 labels each, majority vote).
2. **LeCun, Bottou, Bengio, Haffner (1998).** Gradient-Based Learning Applied to Document Recognition.
   Proc. IEEE 86(11):2278-2324. **reported** (citation details only; link only, deep-learning sibling owns it).
3. **Goodfellow, Bengio, Courville (2016).** Deep Learning, MIT Press, ch. 9 Convolutional Networks,
   deeplearningbook.org/contents/convnets.html. **read-at-source**
   - 9.2: convolution "leverages three important ideas": sparse interactions, parameter sharing, equivariant representations.
   - 9.1: "Many machine learning libraries implement cross-correlation but call it convolution".
   - 9.3: pooling makes the representation "approximately invariant to small translations of the input".
4. **Szeliski (2022).** Computer Vision: Algorithms and Applications, 2nd ed., Springer; szeliski.org/Book.
   **reported** (book text not opened: the free PDF sits behind a name-and-email form). Pages may cite it
   only as further reading on linear filtering; no sentence may attribute a specific claim to it.
5. **Settles (2009, rev. 2010).** Active Learning Literature Survey. CS Tech Report 1648, UW-Madison.
   **read-at-source**
   - 3.1 uncertainty sampling: least confident, margin (top two classes), entropy.
   - 3.2 query-by-committee. 3.6 density-weighted: uncertainty and QBC are prone to "querying outliers".
   - 4.1: a set built by an active learner "is inherently tied to the model that was used to generate it";
     Schein and Ungar 2007 saw active learning need MORE labels than passive; positive results may partly
     reflect "publication bias".
6. **Lowell, Lipton, Wallace (2019).** Practical Obstacles to Deploying Active Learning. EMNLP-IJCNLP 2019,
   arXiv:1807.04801. **read-at-source** (abstract)
   - "AL couples the training dataset with the model used to guide its acquisition."
   - Successor models trained on actively acquired data do not consistently outperform i.i.d. sampled data.
   - Do NOT say "worse than random" (not in the abstract).
7. **Sener, Savarese (2018).** Active Learning for Convolutional Neural Networks: A Core-Set Approach.
   ICLR 2018, arXiv:1708.00489. **read-at-source** (abstract): recasts active learning as core-set selection,
   claims it "significantly outperforms existing approaches in image classification experiments by a large margin".
8. **Munjal et al. (2022).** Towards Robust and Reproducible Active Learning Using Neural Networks. CVPR 2022,
   arXiv:2002.09564. **read-at-source** (abstract): under identical settings, methods "produce an inconsistent
   gain over random sampling baseline"; with strong regularisation little or no advantage over random.
9. **Beck et al. (2021).** Effective Evaluation of Deep Active Learning on Image Classification Tasks.
   arXiv:2106.15324. **read-at-source** (abstract): roughly 2x to 4x "more label-efficient compared to RS with
   the use of data augmentation".
10. **Northcutt, Athalye, Mueller (2021).** Pervasive Label Errors in Test Sets Destabilize Machine Learning
    Benchmarks. NeurIPS 2021 Datasets and Benchmarks, arXiv:2103.14749. **read-at-source**
    - "an average of at least 3.3% errors across the 10 datasets"; ImageNet validation at least 6% (2,916 of 50,000).
    - MNIST test set: 15 validated errors (0.15%), the lowest; QuickDraw highest at 10.12%.
    - "lower capacity models may be practically more useful" on corrected labels.
    - Fashion-MNIST is NOT one of the 10 datasets: never claim a Fashion-MNIST error rate.
11. **Everingham et al. (2010).** The PASCAL Visual Object Classes (VOC) Challenge. IJCV 88:303-338. **read-at-source**
    - 4.2, Eq. 3: overlap = area(Bp intersect Bgt) / area(Bp union Bgt); a detection is correct if it "must exceed 0.5 (50%)".
    - The threshold "was set deliberately low to account for inaccuracies in bounding boxes in the ground truth data".
    - Duplicate detections of one object count as false positives.
12. **COCO detection evaluation**, cocodataset.org/#detection-eval. **read-at-source**
    - Primary metric AP averaged over 10 IoU thresholds .50:.05:.95; AP50 = "PASCAL VOC metric", AP75 = "strict metric".
    - Small objects under 32x32 pixels, medium 32x32 to 96x96, large above; about 41% of objects are small.
13. **Guo, Pleiss, Sun, Weinberger (2017).** On Calibration of Modern Neural Networks. ICML 2017, arXiv:1706.04599.
    **read-at-source** (abstract): modern networks are poorly calibrated; temperature scaling, one parameter,
    is surprisingly effective. The ECE formula on the pages is the standard binned definition and is
    described as such, not quoted from the paper body (body not read).
14. **Hendrycks, Dietterich (2019).** Benchmarking Neural Network Robustness to Common Corruptions and Perturbations.
    ICLR 2019, arXiv:1903.12261. **read-at-source** (abstract): ImageNet-C and ImageNet-P; "negligible changes in
    relative corruption robustness from AlexNet classifiers to ResNet classifiers".
15. **Yosinski, Clune, Bengio, Lipson (2014).** How transferable are features in deep neural networks? NIPS 27,
    arXiv:1411.1792. **read-at-source** (abstract): first-layer features resemble Gabor filters and colour blobs,
    general across datasets; features move from general to specific by the last layer; transferred
    initialisation improves generalisation even after fine-tuning.

---

## 3. Per-session coverage

1. **An image is a grid of numbers** (green). Fashion-MNIST facts (1); pixels, channels, resolution; what the
   model receives (784 numbers, negated intensities); brightness/contrast/flip/invert as array operations and
   clipping at 255 (gallery canon). Widget #cvf-pixels. Artifact: Corvane's image card.
2. **Slide a small window: convolution** (yellow). Kernels as weighted sums; cross-correlation vs convolution (3);
   edges, blur, sharpen on real pixels (gallery canon); output size 28 - 3 + 1; stride, padding, pooling (3);
   parameter sharing; why a learned kernel beats a hand-made one (15, first-layer Gabor-like filters).
   Links learn-deep-learning 04 for architectures. Widget #cvf-conv.
3. **A first classifier and what it confuses** (yellow). Raw vs hand vs conv features at 300 labels (canon);
   softmax regression in plain words; confusion matrix and per-class error (canon); paper Table 3 numbers on
   the full 60,000 for scale (1); human 0.835 (1). Links intro-ml b7, deep-learning 02 and 03. Artifact: baseline + error table.
4. **The label budget bench** (orange). Ladder (canon), uncertainty (5) and diversity (7) sampling, the cold start,
   seed luck, label noise anti-lever, label error base rates (10). Widget #cvf-bench. Artifact: labelling budget plan.
5. **Where it fails: slices and shift** (orange). Failure slices (per class, faint items), shift table (canon),
   calibration and ECE (13), corruption benchmarks (14), detection and segmentation as concepts, IoU (11, 12)
   on drawn boxes (canon presets). Links model-evaluation b3 for calibration depth. Widgets #cvf-shift, #cvf-iou.
6. **Scope the vision feature** (red). Pretrained models and transfer (15); label cost vs model cost; what to ask
   a vendor; the active learning disagreement (5, 6, 7, 8, 9) taught as a disagreement; capstone signed scope note.

## 4. Seams (pick the side out loud, link with absolute https, never reteach)

| Sibling | They own | Link |
|---|---|---|
| Deep Learning 04 | LeNet, AlexNet, ResNet, counting what convolution saves | https://phoebefu6.github.io/learn-deep-learning-with-phoebe/courses/04-seeing-and-embedding.html |
| Deep Learning 02, 03 | backprop, the training loop | .../02-backprop-by-hand.html, .../03-the-training-loop.html |
| Autonomous Driving p2, p5 | label cost per frame, perception metrics on fleets | https://phoebefu6.github.io/learn-autonomous-driving-with-phoebe/courses/p2-labels-cost-by-the-frame.html, .../p5-perception-metrics-that-hide-failures.html |
| Anomaly Detection | defect detection, anomaly methods | https://phoebefu6.github.io/learn-anomaly-detection-with-phoebe/ |
| Intro ML b2, b7 | splits, classification metrics | https://phoebefu6.github.io/learn-intro-ml-with-phoebe/courses/b7-classification-metrics.html |
| Model Evaluation b3 | curves and calibration in depth | https://phoebefu6.github.io/learn-model-evaluation-with-phoebe/courses/b3-curves-calibration.html |
| AI Media Domain b7, AI Design | generated images, rights, provenance | https://phoebefu6.github.io/learn-ai-media-domain-with-phoebe/courses/b7-images-and-synthetic-media.html |

## 5. Not covered (honest list)

- Training a deep network: no CNN is trained; features are FIXED kernels, the classifier is softmax regression.
  Learned kernels are explained and linked, not computed.
- Object detectors and segmentation models: concepts and IoU only; no detector is trained.
- Colour images: Fashion-MNIST is greyscale; channels are taught on the page, not computed on RGB data.
- Real waste or recycling imagery: none used; Corvane is constructed and its images are garments.
- Data augmentation, self-supervised pretraining, foundation vision models: named in session 6 only.
- A Fashion-MNIST label error rate: not measured by Northcutt et al.; not claimed.
- Szeliski's text: not opened; further reading only.
- Exact ECE formula from Guo et al.: described as the standard binned definition.
- Vendor pricing for labelling: no figure is quoted; Corvane's costs are constructed and labelled.

## 6. Constructed (label on the page every time)

Corvane, its sorting line, its camera, its budgets and any price: "Constructed Corvane example". Budgets are
stated in labels, not money; where session 6 puts a price on a label it is labelled constructed.

## 7. Build-along run numbers (computed on full-size 28 x 28 files; NOT bench canon)

Re-run independently by the lead on 2026-11-19 from /private/tmp/claude-504/cvf/verify (all matched the pages):
- s1: x_test[164] T-shirt mean 36.58, 303 zeros, 1 at 255; img + 60 in uint8: 4 pixels darker; clipped +60:
  mean 96.36, 0 zeros, 4 at 255; flip changes 518 positions; train ink by class Sandal 34.9 ... Coat 98.3.
- s2: x_test[2] Trouser patch rows 16-18 cols 10-12 sums to -347, /4 abs 86.75, next column 242.5; vertical
  228 / horizontal 44 positions above 40; x_test[230] Sneaker 72 / 182; pooled 76 of 169 above 40; feature-sum
  check Trouser 5,170 vs 1,005, Sneaker 1,098 vs 2,555.
- s3: LogisticRegression(max_iter=1000), 20,000 train (rng seed 0): majority Shirt 2,043, baseline 0.1;
  511 iterations, accuracy 0.8312; T-shirt as Shirt 115, Pullover as Coat 120; Shirt as T-shirt 143 (largest).
- s5: LogisticRegression, 10,000 train (rng seed 5): clean 0.824; faint quarter 2,157 of 2,500 vs 6,086 of
  7,500; dim 0.772 at 0.766 confidence; flip 0.557 at 0.812; 2 px roll 0.458 at 0.745; quarter turn 0.031 at
  0.862; ECE clean 0.045, quarter turn 0.831 (agent run; ECE not re-run by the lead).
