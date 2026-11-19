"""Build assets/cvf-sample.js from the public Fashion-MNIST files.

Fashion-MNIST: Xiao, Rasul, Vollgraf (2017), arXiv:1708.07747, MIT licence,
https://github.com/zalandoresearch/fashion-mnist

Usage (data directory holds the four .gz files from the repo's data/fashion folder):
    python3 -I materials/cvf_build_sample.py <data_dir> assets/cvf-sample.js

What ships, and why it is smaller than the original:
  pool     2,000 images drawn (seed 20170825) from the 60,000 training images
  holdout  1,000 images drawn (same seed) from the 10,000 official test images
  Both are pooled 2x2 to 14x14 (mean of four pixels) and quantised to 16 grey levels
  (value = round(mean / 17), shown as value * 17), packed two pixels per byte, base64.
  gallery  20 test images (the first two of each class in the holdout) at the original
  28x28 and 256 levels, for the pixel-level pages.
The data is serialised with json.dumps, never string interpolation.
"""
import base64
import gzip
import json
import sys

import numpy as np

SEED = 20170825
N_POOL, N_HOLD = 2000, 1000
CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def read_images(path):
    with gzip.open(path, "rb") as f:
        raw = f.read()
    n = int.from_bytes(raw[4:8], "big")
    return np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(n, 28, 28)


def read_labels(path):
    with gzip.open(path, "rb") as f:
        raw = f.read()
    return np.frombuffer(raw, dtype=np.uint8, offset=8)


def pool14(imgs):
    m = imgs.reshape(-1, 14, 2, 14, 2).astype(np.float64).mean(axis=(2, 4))
    q = np.floor(m / 17.0 + 0.5).astype(np.uint8)          # 0..15, round half up
    return q.reshape(-1, 196)


def pack(q):
    flat = q.reshape(-1)
    packed = (flat[0::2] << 4) | flat[1::2]
    return base64.b64encode(packed.astype(np.uint8).tobytes()).decode("ascii")


def main(data_dir, out_path):
    xtr = read_images(f"{data_dir}/train-images-idx3-ubyte.gz")
    ytr = read_labels(f"{data_dir}/train-labels-idx1-ubyte.gz")
    xte = read_images(f"{data_dir}/t10k-images-idx3-ubyte.gz")
    yte = read_labels(f"{data_dir}/t10k-labels-idx1-ubyte.gz")
    assert xtr.shape == (60000, 28, 28) and xte.shape == (10000, 28, 28)

    rng = np.random.default_rng(SEED)
    pi = np.sort(rng.choice(60000, N_POOL, replace=False))
    hi = np.sort(rng.choice(10000, N_HOLD, replace=False))

    gallery_idx = []
    for c in range(10):
        gallery_idx += [int(i) for i in hi[yte[hi] == c][:2]]

    sample = {
        "source": "Fashion-MNIST (Xiao, Rasul, Vollgraf 2017), MIT licence",
        "seed": SEED,
        "classes": CLASSES,
        "side": 14,
        "levels": 16,
        "pool": {"n": N_POOL, "from": "train", "index": pi.tolist(),
                 "y": ytr[pi].tolist(), "px": pack(pool14(xtr[pi]))},
        "holdout": {"n": N_HOLD, "from": "t10k", "index": hi.tolist(),
                    "y": yte[hi].tolist(), "px": pack(pool14(xte[hi]))},
        "gallery": {"side": 28, "index": gallery_idx,
                    "y": yte[gallery_idx].tolist(),
                    "px": base64.b64encode(xte[gallery_idx].tobytes()).decode("ascii")},
    }
    head = ("/* cvf-sample.js - real Fashion-MNIST images for learn-computer-vision-with-phoebe.\n"
            " * Source: Fashion-MNIST, Xiao, Rasul, Vollgraf (2017), arXiv:1708.07747, MIT licence,\n"
            " * github.com/zalandoresearch/fashion-mnist. Built by materials/cvf_build_sample.py.\n"
            " * pool: 2,000 training images, holdout: 1,000 official test images, both pooled to\n"
            " * 14x14 and 16 grey levels (two pixels per byte, base64). gallery: 20 test images at\n"
            " * the original 28x28, 256 levels. */\n")
    with open(out_path, "w") as f:
        f.write(head + "window.CVF_SAMPLE = " + json.dumps(sample, separators=(",", ":")) + ";\n")
    print("pool class counts", np.bincount(ytr[pi], minlength=10).tolist())
    print("holdout class counts", np.bincount(yte[hi], minlength=10).tolist())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
