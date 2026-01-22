# load in map = ./egi256_to_1020.csv

# load in dataset from LocalizeMI (just do one series) using mne

# get channel names from mne

# use map to go from 256 names to 1020 system names

# drop channels not in 1020 system ---> should give us ~20 channels assuming none bad

# grid 5x5 to 8x8 ---> nn.predict


from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio
import json
import re
import mne

import matplotlib.pyplot as plt

try:
    from tensorflow.keras.models import model_from_json
except Exception:
    from keras.models import model_from_json

def canon(s: str) -> str:
    """Lowercase, strip spaces, drop non-alphanum."""
    s = str(s)
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]", "", s)
    return s

def mat_cell_to_str(x) -> str:
    """Convert MATLAB cell entry (often ndarray(['EEGFp1'])) to python string."""
    if isinstance(x, np.ndarray):
        # e.g. array(['EEGFp1'], dtype='<U6')
        if x.size == 1:
            return str(x.item())
        return str(x.flat[0])
    return str(x)

def five_to_eight_indices(i5: int, j5: int):
    """
    Map a 5x5 cell (i5,j5) to one or more 8x8 cells after grid_padding().

    Because grid_padding duplicates row=2 and col=2, a single 5x5 cell can map to:
      - 1 cell (most locations)
      - 2 cells (if on duplicated row OR duplicated col)
      - 4 cells (if at (2,2) center: duplicated row AND duplicated col)
    """
    def map_axis(k5: int):
        if k5 < 2:
            return [k5]          # 0->0, 1->1 in the 6x6 center
        if k5 == 2:
            return [2, 3]        # duplicated axis
        return [k5 + 1]          # 3->4, 4->5

    rows6 = map_axis(i5)
    cols6 = map_axis(j5)

    # center 6x6 goes into 8x8 at indices [1:7, 1:7]
    return [(r + 1, c + 1) for r in rows6 for c in cols6]

def occlude_cell_batch(X: np.ndarray, i: int, j: int, value: float = 0.0) -> np.ndarray:
    """
    Occlude a single spatial cell (i,j) across all samples by setting its entire time vector to `value`.

    X: (N, H, W, T)
    returns: (N, H, W, T)
    """
    if X.ndim != 4:
        raise ValueError(f"Expected (N,H,W,T), got {X.shape}")
    Xo = X.copy()
    Xo[:, i, j, :] = value
    return Xo

def evaluate_and_plot_occluded_cell(
    X8_true: np.ndarray,
    Y_pred: np.ndarray,
    occ_5x5: tuple[int, int],
    out_png: str = "localize_occlusion_eval.png",
    max_epochs_plot: int = 200,
    seed: int = 0,
):
    """
    Evaluate reconstruction quality for a 5x5-occluded cell by inspecting the
    corresponding 8x8 location(s) after broadcast/padding.

    X8_true: (N, 8, 8, 8)  ground truth (NOT occluded)
    Y_pred : (N, 8, 8, 8)  model output when given occluded input
    occ_5x5: (i5, j5) occluded location in the 5x5 grid
    """
    if X8_true.shape != Y_pred.shape:
        raise ValueError(f"Shape mismatch: true={X8_true.shape}, pred={Y_pred.shape}")
    if X8_true.ndim != 4 or X8_true.shape[1:3] != (8, 8):
        raise ValueError(f"Expected (N,8,8,8), got {X8_true.shape}")

    N, H, W, T = X8_true.shape
    i5, j5 = occ_5x5
    affected = five_to_eight_indices(i5, j5)

    # choose epochs for plotting (don’t plot 11k traces)
    rng = np.random.default_rng(seed)
    n_plot = min(max_epochs_plot, N)
    idx = rng.choice(N, size=n_plot, replace=False) if n_plot < N else np.arange(N)

    # Collect per-affected-cell metrics
    metrics = []
    for (r, c) in affected:
        true_flat = X8_true[:, r, c, :].reshape(-1)
        pred_flat = Y_pred[:, r, c, :].reshape(-1)

        mse = float(np.mean((true_flat - pred_flat) ** 2))
        mae = float(np.mean(np.abs(true_flat - pred_flat)))

        # correlation (guard against degenerate std)
        tstd = np.std(true_flat)
        pstd = np.std(pred_flat)
        if tstd < 1e-12 or pstd < 1e-12:
            corr = float("nan")
        else:
            corr = float(np.corrcoef(true_flat, pred_flat)[0, 1])

        metrics.append({"cell": (r, c), "mse": mse, "mae": mae, "corr": corr})

    # Also compute an aggregate metric over all affected cells combined
    true_all = np.concatenate([X8_true[:, r, c, :].reshape(-1) for (r, c) in affected], axis=0)
    pred_all = np.concatenate([Y_pred[:, r, c, :].reshape(-1) for (r, c) in affected], axis=0)

    mse_all = float(np.mean((true_all - pred_all) ** 2))
    mae_all = float(np.mean(np.abs(true_all - pred_all)))
    if np.std(true_all) < 1e-12 or np.std(pred_all) < 1e-12:
        corr_all = float("nan")
    else:
        corr_all = float(np.corrcoef(true_all, pred_all)[0, 1])

    # ---- Plotting ----
    nrows = len(affected)
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(12, 3.0 * nrows), squeeze=False)

    for k, (r, c) in enumerate(affected):
        # stack the selected epochs for a readable line plot
        true_trace = X8_true[idx, r, c, :].reshape(-1)
        pred_trace = Y_pred[idx, r, c, :].reshape(-1)

        ax0 = axes[k, 0]
        ax0.plot(true_trace, label="True", linewidth=1)
        ax0.plot(pred_trace, label="Pred", linewidth=1)
        ax0.set_title(f"8x8 cell ({r},{c})  |  plot_epochs={n_plot}")
        ax0.legend()

        ax1 = axes[k, 1]
        # scatter: all epochs (not just plotted subset) so metric reflects full data
        true_flat = X8_true[:, r, c, :].reshape(-1)
        pred_flat = Y_pred[:, r, c, :].reshape(-1)
        ax1.scatter(true_flat, pred_flat, s=3, alpha=0.25)
        ax1.set_xlabel("True")
        ax1.set_ylabel("Pred")
        ax1.set_title(f"Scatter | MSE={metrics[k]['mse']:.4g}, r={metrics[k]['corr']:.3f}")

    fig.suptitle(
        f"Occlusion eval for 5x5 cell ({i5},{j5}) -> affected 8x8 cells {affected}\n"
        f"AGG: MSE={mse_all:.4g}, MAE={mae_all:.4g}, r={corr_all:.3f}",
        y=0.995
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"Saved plot to {out_png}")

    # Print metrics table to console
    print("\nPer-cell metrics (affected 8x8 positions):")
    for m in metrics:
        print(f"  cell={m['cell']}: MSE={m['mse']:.6g} | MAE={m['mae']:.6g} | r={m['corr']:.4f}")
    print(f"\nAggregate over affected cells: MSE={mse_all:.6g} | MAE={mae_all:.6g} | r={corr_all:.4f}")

    return {
        "affected_cells": affected,
        "per_cell": metrics,
        "aggregate": {"mse": mse_all, "mae": mae_all, "corr": corr_all},
        "plot_path": out_png,
    }




def grid_padding(X5: np.ndarray) -> np.ndarray:
    """
    Pad 5x5 grids to 8x8 with duplicated center row/column and checkerboard border.

    Input:
      X5: (N, 5, 5, T) 

    Output:
      X8: (N, 8, 8, T) 
    """
    if X5.ndim != 4 or X5.shape[1:3] != (5, 5):
        raise ValueError(f"Expected (N, 5, 5, T), got {X5.shape}")

    N, _, _, T = X5.shape

    # 1) Expand columns: 5 -> 6 by duplicating the middle column (col=2)
    # B: (N, 5, 6, T)
    B = np.concatenate(
        [
            X5[:, :, :2, :],                 # cols 0,1
            np.repeat(X5[:, :, 2:3, :], 2, axis=2),  # col 2 repeated twice
            X5[:, :, 3:, :],                 # cols 3,4
        ],
        axis=2
    )

    # 2) Expand rows: 5 -> 6 by duplicating the middle row (row=2)
    # C: (N, 6, 6, T)
    C = np.concatenate(
        [
            B[:, :2, :, :],                  # rows 0,1
            np.repeat(B[:, 2:3, :, :], 2, axis=1),  # row 2 repeated twice
            B[:, 3:, :, :],                  # rows 3,4
        ],
        axis=1
    )

    # 3) Border padding to 8x8 with checkerboard based on (0,0) and (4,4)
    D = X5[:, 0:1, 0:1, :]   # (N,1,1,T)
    E = X5[:, 4:5, 4:5, :]   # (N,1,1,T)

    # F is 2x2:
    # [[D, E],
    #  [E, D]]
    top = np.concatenate([D, E], axis=2)     # (N,1,2,T)
    bot = np.concatenate([E, D], axis=2)     # (N,1,2,T)
    F = np.concatenate([top, bot], axis=1)   # (N,2,2,T)

    # Tile F to 8x8: repeat 4x in both spatial dims (2*4=8)
    G = np.tile(F, (1, 4, 4, 1))            # (N,8,8,T)

    # Put C in the middle
    G[:, 1:7, 1:7, :] = C                   # center is 6x6

    return G

ROOT = Path(__file__).resolve().parent.parent
DERIV = ROOT / "data" / "localize-mi" / "derivatives" / "epochs" / "sub-01"

epochs_path = DERIV / "eeg" / "sub-01_task-seegstim_run-01_epochs.npy"
electrodes_path = DERIV / "eeg" / "sub-01_task-seegstim_electrodes.tsv"
channels_path = DERIV / "eeg" / "sub-01_task-seegstim_run-01_channels.tsv"

X = np.load(epochs_path)  # (N, C, T)
electrodes = pd.read_csv(electrodes_path, sep="\t")
channels = pd.read_csv(channels_path, sep="\t")

print("X shape:", X.shape)

# --- Load mapping: 10-20 -> EGI channel (E37 etc) ---
map_path = ROOT / "egi256_to_1020.csv"
map_df = pd.read_csv(map_path)
map_df.columns = ["ten20", "egi"]  # based on your preview

# Canonical columns
map_df["ten20_c"] = map_df["ten20"].map(canon)     # e.g. "fp1"
map_df["egi_c"]   = map_df["egi"].map(canon)       # e.g. "e37" OR "E37" -> "e37"

# --- Load 10-20 grid from .mat and extract 10-20 labels present ---
mat = sio.loadmat(ROOT / "10-20_system.mat")
grid = mat["map"]  # 5x5 cell array

grid_str = np.vectorize(mat_cell_to_str)(grid)
# grid entries look like 'EEGFp1', 'EEGA2A1', etc.
grid_clean = np.vectorize(canon)(grid_str)  # e.g. 'eegfp1', 'eega2a1'

# Keep only true EEG 10-20 labels, drop the padding token 'eega2a1'
# DAE paper uses labels like EEGFp1 etc; we strip leading "eeg"
ten20_from_mat = set()
for token in grid_clean.flatten():
    if token.startswith("eeg"):
        core = token[3:]  # remove 'eeg'
        if core not in ("a2a1", ""):
            ten20_from_mat.add(core)

ten20_from_mat = sorted(ten20_from_mat)
print("10-20 labels in mat grid:", ten20_from_mat)
print("Count:", len(ten20_from_mat))

# --- Map those 10-20 labels -> EGI channel names (e37, e46, ...) ---
wanted_map = map_df[map_df["ten20_c"].isin(ten20_from_mat)].copy()

# sanity check: are we missing any mat labels?
missing = set(ten20_from_mat) - set(wanted_map["ten20_c"])
if missing:
    print("WARNING: mat labels missing from CSV map:", sorted(missing))

# Convert to dataset naming convention:
# - channels.tsv uses 'e1','e2',...
# - map gives 'E37' so canon -> 'e37' is correct
wanted_egi = sorted(set(wanted_map["egi_c"]))
print("Mapped EGI channels:", wanted_egi)
print("Mapped count:", len(wanted_egi))

# --- Filter electrodes.tsv to those channels and show coordinates ---
electrodes["name_c"] = electrodes["name"].map(canon)  # 'e1' -> 'e1'
electrodes_1020 = electrodes[electrodes["name_c"].isin(wanted_egi)].copy()

print("electrodes_1020 rows:", len(electrodes_1020))
print(electrodes_1020[["name","x","y","z"]].head())

channels["name_c"] = channels["name"].map(canon)
name_to_idx = {nm: i for i, nm in enumerate(channels["name_c"].tolist())}
valid = [nm for nm in wanted_egi if nm in name_to_idx]
missing_in_channels = [nm for nm in wanted_egi if nm not in name_to_idx]
if missing_in_channels:
    print("WARNING: mapped EGI channels not found in channels.tsv:", missing_in_channels)

idxs = np.array([name_to_idx[nm] for nm in valid], dtype=int)

X_1020 = X[:, idxs, :]
print("X_1020 shape:", X_1020.shape)

mask_5x5 = np.array([
    [0, 1, 0, 1, 0],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [0, 1, 0, 1, 0],
], dtype=bool)

N, C, T = X_1020.shape
assert C == mask_5x5.sum() == 19

X_grid = np.zeros((N, 5, 5, T), dtype=X_1020.dtype)

# fill grid in row-major order
k = 0
for i in range(5):
    for j in range(5):
        if mask_5x5[i, j]:
            X_grid[:, i, j, :] = X_1020[:, k, :]
            k += 1

# padding cells are already zero
print("X_grid shape:", X_grid.shape)

# X_1020: (N, 19, T)
N, C, T = X_1020.shape
assert C == 19

win = 128  # 16 ms at 8000 Hz
T2 = (T // win) * win      # drop remainder
n_win = T2 // win          # windows per epoch

# 1) window: (N, 19, n_win, 128)
Xw = X_1020[:, :, :T2].reshape(N, C, n_win, win)

# 2) downsample 128 -> 8 by averaging every 16 samples
#    (N, 19, n_win, 8)
X8 = Xw.reshape(N, C, n_win, 8, 16).mean(axis=-1)

# 3) map 19 channels into 5x5 grid with padding zeros
mask_5x5 = np.array([
    [0, 1, 0, 1, 0],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [0, 1, 0, 1, 0],
], dtype=bool)

# grid: (N, n_win, 5, 5, 8)
G5 = np.zeros((N, n_win, 5, 5, 8), dtype=X8.dtype)
k = 0
for i in range(5):
    for j in range(5):
        if mask_5x5[i, j]:
            G5[:, :, i, j, :] = X8[:, k, :, :]  # (N, n_win, 8)
            k += 1

# G5: (N, n_win, 5, 5, 8)
G5_flat = G5.reshape(N * n_win, 5, 5, 8)

# ---- MINIMAL FIX: match the DAE preprocessing (zscore=True) ----
mu = G5_flat.mean(axis=-1, keepdims=True)
sd = G5_flat.std(axis=-1, keepdims=True) + 1e-8
G5z = (G5_flat - mu) / sd
# ---------------------------------------------------------------

# Apply yesterday's exact padding (5x5 -> 8x8 with duplication + checkerboard)
G8_flat = grid_padding(G5z)  # (N*n_win, 8, 8, 8)

# Step 5: this is already the desired sample format
X_dae = G8_flat  # (N*n_win, 8, 8, 8)

print("X_dae shape:", X_dae.shape)  # (N*n_win, 8, 8, 8)

# Occlude in 5x5 space *after* z-scoring, then pad
grid_occ = occlude_cell_batch(G5z, i=2, j=2, value=0.0)
processed_grids = X_dae
processed_grids_occ = grid_padding(grid_occ)


model_json_path = ROOT / "model" / "topology" / "model.json"
with open(model_json_path, "r") as f:
    architecture = json.load(f)
nn = model_from_json(architecture)
nn.load_weights(ROOT / "model" / "weights" / "nn_weights-800.hdf5", by_name=True)

Y_pred = nn.predict(processed_grids_occ, verbose=0)

print("Predicted shape:", Y_pred.shape)  # (N, 8, 8, 8)

results = evaluate_and_plot_occluded_cell(
    X8_true=processed_grids,
    Y_pred=Y_pred,
    occ_5x5=(2, 2),               
    out_png="localize_occlusion_eval.png",
    max_epochs_plot=30,
    seed=0
)