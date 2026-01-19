#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

try:
    from tensorflow.keras.models import model_from_json
except Exception:
    from keras.models import model_from_json


def load_grid_map(mat_path: str) -> np.ndarray:
    d = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    if "map" not in d:
        raise KeyError(f"{mat_path} missing key 'map'. Found keys: {[k for k in d.keys() if not k.startswith('__')]}")
    grid = d["map"]
    if not (isinstance(grid, np.ndarray) and grid.shape == (5, 5)):
        raise ValueError(f"Expected map shape (5,5), got {type(grid)} {getattr(grid, 'shape', None)}")
    out = np.empty_like(grid, dtype=object)
    for i in range(5):
        for j in range(5):
            v = grid[i, j]
            if isinstance(v, bytes):
                v = v.decode("utf-8", errors="ignore")
            out[i, j] = str(v)
    return out


def make_synthetic_signals(ch_names, n_samples=64, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n_samples, endpoint=False)
    sigs = {}
    for k, name in enumerate(ch_names):
        f = 3 + (k % 10)
        sigs[name] = np.sin(2 * np.pi * f * t) + 0.05 * rng.standard_normal(n_samples)
    return sigs


def build_8x8x8_examples(map5, signals, window_len=8, stride=8):
    r0, c0 = 1, 1
    H, W = 8, 8
    any_sig = next(iter(signals.values()))
    T = len(any_sig)

    examples = []
    for start in range(0, T - window_len + 1, stride):
        x = np.zeros((H, W, window_len), dtype=np.float32)
        for i in range(5):
            for j in range(5):
                ch = map5[i, j]
                if ch is None or ch.strip() == "" or ch.lower() == "none":
                    continue
                x[r0 + i, c0 + j, :] = signals[ch][start:start + window_len]
        examples.append(x)
    return np.array(examples)


def parse_occ_list(s: str):
    """
    Parse occlusions like: "3,3;4,4;2,5"
    Returns list of (r,c) ints.
    """
    out = []
    if s is None or s.strip() == "":
        return out
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        r_str, c_str = part.split(",")
        out.append((int(r_str), int(c_str)))
    return out


def occlude_cells(X, occ_rcs):
    Xo = X.copy()
    for (r, c) in occ_rcs:
        Xo[:, r, c, :] = 0.0
    return Xo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", default="10-20_system.mat")
    parser.add_argument("--model_json", default="./model/topology/model.json")
    parser.add_argument("--weights", default="./model/weights/nn_weights-800.hdf5")
    parser.add_argument("--occ", type=str, default="3,3", help='Occlusions like "3,3;4,4;2,5"')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_png", default="example_plot.png")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(repo_root)

    # Load model
    with open(args.model_json, "r") as f:
        architecture = json.load(f)
    nn = model_from_json(architecture)
    nn.load_weights(args.weights, by_name=True)

    # Load mapping + synthetic data
    map5 = load_grid_map(args.mat)
    ch_names = sorted(set(map5.ravel().tolist()))
    print(f"Loaded 5x5 map with {len(ch_names)} unique channel names")

    signals = make_synthetic_signals(ch_names, n_samples=64, seed=args.seed)
    first_key = next(iter(signals))
    print("signals dict: n_channels =", len(signals), "one signal shape =", signals[first_key].shape, "example key =", first_key)

    # Build examples
    X = build_8x8x8_examples(map5, signals, window_len=8, stride=8)
    print("X shape:", X.shape)

    # Occlude multiple cells
    occ_rcs = parse_occ_list(args.occ)
    if len(occ_rcs) == 0:
        raise ValueError("No occlusions provided. Use --occ like '3,3;4,4'")
    for (r, c) in occ_rcs:
        if not (0 <= r < 8 and 0 <= c < 8):
            raise ValueError(f"Occlusion {(r,c)} out of bounds for 8x8 grid")
    print("Occluding cells:", occ_rcs)

    X_occ = occlude_cells(X, occ_rcs)

    # Predict
    Y_pred = nn.predict(X_occ, verbose=0)

    # Plot per-occluded-cell
    nplots = len(occ_rcs)
    ncols = min(2, nplots)
    nrows = int(np.ceil(nplots / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7 * ncols, 3.5 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax_i, (r, c) in enumerate(occ_rcs):
        true_trace = X[:, r, c, :].reshape(-1)
        pred_trace = Y_pred[:, r, c, :].reshape(-1)

        axes[ax_i].plot(true_trace, label="True")
        axes[ax_i].plot(pred_trace, label="NN recon")
        axes[ax_i].set_title(f"Occluded cell ({r},{c})")
        axes[ax_i].legend()

    # Hide unused axes
    for j in range(nplots, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Multi-channel interpolation (NN) on occluded grid cells")
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=150)
    print(f"Saved plot to {args.out_png}")


if __name__ == "__main__":
    main()
