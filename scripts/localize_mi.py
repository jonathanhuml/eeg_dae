import json
import re
import mne
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from utils.preprocess_dae import (
    canon,
    mat_cell_to_str,
    five_to_eight_indices,
    occlude_cell_batch,
    grid_padding,
    evaluate_and_plot_occluded_cell,
)
try:
    from tensorflow.keras.models import model_from_json
except Exception:
    from keras.models import model_from_json

ROOT = Path(__file__).resolve().parent.parent
DERIV = ROOT / "data" / "localize-mi" / "derivatives" / "epochs" / "sub-01"

epochs_path = DERIV / "eeg" / "sub-01_task-seegstim_run-01_epochs.npy"
electrodes_path = DERIV / "eeg" / "sub-01_task-seegstim_electrodes.tsv"
channels_path = DERIV / "eeg" / "sub-01_task-seegstim_run-01_channels.tsv"

X = np.load(epochs_path)  # (N, C, T)
electrodes = pd.read_csv(electrodes_path, sep="\t")
channels = pd.read_csv(channels_path, sep="\t")

# --- Load mapping: 10-20 -> EGI channel (E37 etc) ---
map_path = ROOT / "egi256_to_1020.csv"
map_df = pd.read_csv(map_path)
map_df.columns = ["ten20", "egi"]  
map_df["ten20_c"] = map_df["ten20"].map(canon)     # e.g. "fp1"
map_df["egi_c"]   = map_df["egi"].map(canon)       # e.g. "e37" OR "E37" -> "e37"

# --- Load 10-20 Intl. Standard grid ---
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
# --- Map those 10-20 labels -> EGI channel names (e37, e46, ...) ---
wanted_map = map_df[map_df["ten20_c"].isin(ten20_from_mat)].copy()
missing = set(ten20_from_mat) - set(wanted_map["ten20_c"])
if missing:
    print("WARNING: mat labels missing from CSV map:", sorted(missing))

# Convert to dataset naming convention:
# - channels.tsv uses 'e1','e2',...
# - map gives 'E37' so canon -> 'e37' is correct
wanted_egi = sorted(set(wanted_map["egi_c"]))

# --- Filter electrodes.tsv to those channels and show coordinates ---
electrodes["name_c"] = electrodes["name"].map(canon)  # 'e1' -> 'e1'
electrodes_1020 = electrodes[electrodes["name_c"].isin(wanted_egi)].copy()
channels["name_c"] = channels["name"].map(canon)
name_to_idx = {nm: i for i, nm in enumerate(channels["name_c"].tolist())}
valid = [nm for nm in wanted_egi if nm in name_to_idx]
missing_in_channels = [nm for nm in wanted_egi if nm not in name_to_idx]
if missing_in_channels:
    print("WARNING: mapped EGI channels not found in channels.tsv:", missing_in_channels)

idxs = np.array([name_to_idx[nm] for nm in valid], dtype=int)

X_1020 = X[:, idxs, :]
print("X_1020 shape:", X_1020.shape)

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

# ---- Match the DAE preprocessing (zscore=True) ----
mu = G5_flat.mean(axis=-1, keepdims=True)
sd = G5_flat.std(axis=-1, keepdims=True) + 1e-8
G5z = (G5_flat - mu) / sd
# ---------------------------------------------------------------

# Occlude in 5x5 space *after* z-scoring, then pad
grid_occ = occlude_cell_batch(G5z, i=2, j=2, value=0.0)

# (5x5 10-20 -> 8x8 with duplication + checkerboard)
processed_grids = grid_padding(G5z) # (N*n_win, 8, 8, 8)
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
    seed=3
)