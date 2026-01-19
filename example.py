#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

from datasets.dreamer import load_dreamer_dataset

try:
    from tensorflow.keras.models import model_from_json
except Exception:
    from keras.models import model_from_json


def parse_occ_list(s: str):
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


def dreamer_to_model_input(x_torch):
    """
    x_torch: torch.Tensor with shape [F, H, W] (here F=4, H=W=9)
    returns: np.ndarray float32 with shape [8, 8, 8] for the CNN
    """
    x = x_torch.detach().cpu().numpy().astype(np.float32)

    # Ensure (F,H,W)
    if x.ndim != 3:
        raise ValueError(f"Expected DREAMER sample with 3 dims [F,H,W], got {x.shape}")

    F, H, W = x.shape
    if H < 8 or W < 8:
        raise ValueError(f"Grid too small to crop to 8x8: got {H}x{W}")

    # 1) Crop center to 8x8 (9x9 -> 8x8 by dropping last row/col or center crop)
    # We'll do a simple top-left crop for determinism; you can switch to center crop if desired.
    x_8 = x[:, :8, :8]  # (F,8,8)

    # 2) Map F feature maps -> 8 slices
    # Placeholder: tile feature dimension to length 8.
    # If F=4 -> repeat twice => 8
    if F == 8:
        x_feat8 = x_8
    elif F < 8:
        reps = int(np.ceil(8 / F))
        x_feat8 = np.tile(x_8, (reps, 1, 1))[:8, :, :]  # (8,8,8)
    else:
        # If F > 8, just take first 8
        x_feat8 = x_8[:8, :, :]

    # 3) Keras model expects (H,W,8) with channels_last
    x_hw8 = np.transpose(x_feat8, (1, 2, 0))  # (8,8,8)
    return x_hw8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_json", default="./model/topology/model.json")
    parser.add_argument("--weights", default="./model/weights/nn_weights-800.hdf5")
    parser.add_argument("--occ", type=str, default="3,3", help='Occlusions like "3,3;4,4;2,5"')
    parser.add_argument("--out_png", default="dreamer_example.png")
    parser.add_argument("--n_samples", type=int, default=16, help="Number of DREAMER samples to use")
    parser.add_argument("--io_path", type=str, default="", help="TorchEEG cache path to skip processing, e.g. .torcheeg/datasets_...")
    parser.add_argument("--mat_path", type=str, default="/workspace/DREAMER.mat", help="Path to DREAMER.mat")
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(repo_root)

    # Load Keras model
    with open(args.model_json, "r") as f:
        architecture = json.load(f)
    nn = model_from_json(architecture)
    nn.load_weights(args.weights, by_name=True)

    # Load DREAMER dataset (use cache if provided)
    if args.io_path.strip():
        ds = load_dreamer_dataset(mat_path=args.mat_path, io_path=args.io_path.strip())
    else:
        ds = load_dreamer_dataset(mat_path=args.mat_path)

    # Build X batch for model
    n = min(args.n_samples, len(ds))
    X_list = []
    y_list = []

    for i in range(n):
        x_t, y = ds[i]  # x_t: torch.Tensor [F,H,W]
        X_list.append(dreamer_to_model_input(x_t))
        y_list.append(int(y))

    X = np.stack(X_list, axis=0)  # (n,8,8,8)
    print("Built model input X:", X.shape, "labels example:", y_list[:5])

    # Occlusions
    occ_rcs = parse_occ_list(args.occ)
    if len(occ_rcs) == 0:
        raise ValueError("No occlusions provided. Use --occ like '3,3;4,4'")
    for (r, c) in occ_rcs:
        if not (0 <= r < 8 and 0 <= c < 8):
            raise ValueError(f"Occlusion {(r,c)} out of bounds for 8x8 grid")
    print("Occluding cells:", occ_rcs)

    X_occ = occlude_cells(X, occ_rcs)

    # Predict
    Y_pred = nn.predict(X_occ, verbose=0)  # (n,8,8,8)

    # Plot per occluded cell: compare original vs recon across dataset samples
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
        axes[ax_i].set_title(f"DREAMER occluded cell ({r},{c})")
        axes[ax_i].legend()

    for j in range(nplots, len(axes)):
        axes[j].axis("off")

    fig.suptitle("NN interpolation on DREAMER (ToGrid + DE features)")
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=150)
    print(f"Saved plot to {args.out_png}")


if __name__ == "__main__":
    main()

