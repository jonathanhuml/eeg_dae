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

def plot_dreamer_raw_sample(x_t, y: int, out_png="dreamer_raw.png", title_prefix="DREAMER raw"):
    """
    Plot a DREAMER raw time-series sample.

    x_t: torch.Tensor or np.ndarray with shape (1, 14, 128)
         (batch, channels, time)
    y: int label
    """
    # Convert to numpy
    if hasattr(x_t, "detach"):
        x = x_t.detach().cpu().numpy()
    else:
        x = np.asarray(x_t)

    if x.ndim != 3:
        raise ValueError(f"Expected x_t with 3 dims (1, C, T), got {x.shape}")
    if x.shape[0] != 1:
        raise ValueError(f"Expected batch size 1, got {x.shape[0]}")
    x = x[0]  # (C, T)

    C, T = x.shape
    t = np.arange(T)

    # Basic stats
    rms = np.sqrt(np.mean(x**2, axis=1))  # (C,)
    std = np.std(x, axis=1)

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.0, 0.8])

    # (1) Heatmap: channels x time
    ax0 = fig.add_subplot(gs[0, :])
    im = ax0.imshow(x, aspect="auto")
    ax0.set_title(f"{title_prefix} | y={y} | shape=(1,{C},{T})")
    ax0.set_xlabel("Time index")
    ax0.set_ylabel("Channel")
    fig.colorbar(im, ax=ax0, fraction=0.02, pad=0.02)

    # (2) Overlay line plot (stacked/offset)
    ax1 = fig.add_subplot(gs[1, :])
    # Offset each channel by a robust scale so traces don't overlap
    scale = np.median(std) if np.median(std) > 0 else 1.0
    offsets = np.arange(C) * (3.0 * scale)

    for ch in range(C):
        ax1.plot(t, x[ch] + offsets[ch], linewidth=1)

    ax1.set_title("All channels (offset for visibility)")
    ax1.set_xlabel("Time index")
    ax1.set_yticks(offsets)
    ax1.set_yticklabels([f"Ch{ch}" for ch in range(C)])
    ax1.grid(True, axis="x", linestyle="--", linewidth=0.5)

    # (3) RMS per channel barplot
    ax2 = fig.add_subplot(gs[2, 0])
    ax2.bar(np.arange(C), rms)
    ax2.set_title("RMS energy per channel")
    ax2.set_xlabel("Channel")
    ax2.set_ylabel("RMS")

    # (4) Histogram of values (quick distribution check)
    ax3 = fig.add_subplot(gs[2, 1])
    ax3.hist(x.flatten(), bins=50)
    ax3.set_title("Value histogram (all channels/time)")
    ax3.set_xlabel("Value")
    ax3.set_ylabel("Count")

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"Saved raw DREAMER time-series visualization to {out_png}")


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_json", default="./model/topology/model.json")
    parser.add_argument("--weights", default="./model/weights/nn_weights-800.hdf5")
    parser.add_argument("--occ", type=str, default="3,3", help='Occlusions like "3,3;4,4;2,5"')
    parser.add_argument("--out_png", default="dreamer_example.png")
    parser.add_argument("--n_samples", type=int, default=16, help="Number of DREAMER samples to use")
    parser.add_argument("--io_path", type=str, default="", help="TorchEEG cache path to skip processing, e.g. .torcheeg/datasets_...")
    parser.add_argument("--mat_path", type=str, default="/workspace/data/DREAMER/DREAMER.mat")
    parser.add_argument("--plot_raw", action="store_true", help="Only plot raw DREAMER sample(s) and exit")
    parser.add_argument("--raw_index", type=int, default=0, help="Which DREAMER sample index to visualize")

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

    # test loading functions
    eeg, label = ds[0]
    print("eeg:", eeg.shape)
    print("label:", label)
    # print("y:", y)

    if args.plot_raw:
        x_t, y = ds[args.raw_index]
        print("Raw sample shape:", tuple(x_t.shape), "label:", int(y))
        plot_dreamer_raw_sample(
            x_t, int(y),
            out_png=args.out_png,
            title_prefix=f"DREAMER raw idx={args.raw_index}"
        )


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

