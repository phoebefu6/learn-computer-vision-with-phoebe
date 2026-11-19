<!-- learn-with-phoebe hub banner -->
> ### 📚 Part of [**Learn with Phoebe**](https://phoebefu6.github.io/learn-with-phoebe/)
> The shelf of 204 free, hands-on courses on AI, data, and the craft around them. **[Browse every course ↗](https://phoebefu6.github.io/learn-with-phoebe/)**
<!-- /learn-with-phoebe hub banner -->

# Learn Computer Vision Foundations with Phoebe

Six 45-minute sessions on computer vision foundations, by Phoebe Fu. Images as arrays you can compute on:
run real convolutions on real pixels in your browser, train a small classifier, and spend a fixed
labelling budget where it buys the most accuracy.

Open `index.html` to start. Each session is one static page under `courses/`; no build step, no server.

| # | Session | What you build |
|---|---|---|
| 1 | An image is a grid of numbers | Corvane's image card |
| 2 | Slide a small window: convolution | a numpy feature layer, 288 numbers |
| 3 | A first classifier and what it confuses | a baseline and error table |
| 4 | The label budget bench | a labelling budget plan |
| 5 | Where it fails: slices and shift | evaluation slices and shift tests |
| 6 | Scope the vision feature | a signed scope note |

Corvane is a constructed recycling plant. The images are real: Fashion-MNIST (Xiao, Rasul, Vollgraf 2017,
MIT licence), garments standing in for textiles on a sorting line.

The bench (`assets/cvf-live.js`, data in `assets/cvf-sample.js`) trains every model in the browser on
2,000 pool images and scores it on 1,000 official test images. `materials/cvf_reference.py` recomputes
every number in numpy; `materials/compare-ladder.py` checks the two agree value by value.
