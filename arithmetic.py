import numpy as np
import scipy.io as sio
import mne

def canon(name: str) -> str:
    return str(name).replace(" ", "").replace("-", "")

def map_cell_str(cell) -> str:
    if isinstance(cell, np.ndarray):
        if cell.size == 1:
            cell = cell.item()
        else:
            cell = cell.flat[0]
    return canon(cell)

def epochs_to_grid(epochs, map5: np.ndarray, zscore=False):
    """
    Vectorized conversion: Epochs -> (N, 5, 5, 8).
    """
    ep = epochs.copy()
    X = ep.get_data()  # (N, C, T)
    N, C, T = X.shape

    name_to_idx = {canon(ch): i for i, ch in enumerate(ep.ch_names)}

    # Build (5,5) grid of channel indices
    idx_grid = np.empty((5, 5), dtype=np.int64)
    for i in range(5):
        for j in range(5):
            key = map_cell_str(map5[i, j])
            if key not in name_to_idx:
                raise KeyError(
                    f"Map cell ({i},{j})='{key}' not found in epochs channels.\n"
                    f"Available keys include: {sorted(list(name_to_idx.keys()))[:10]}..."
                )
            idx_grid[i, j] = name_to_idx[key]

    # Gather: X[:, idx_grid, :] -> (N, 5, 5, T)
    grids = X[:, idx_grid, :]  # numpy advanced indexing
    # shape is (N, 5, 5, T) already

    if zscore:
        mu = grids.mean(axis=-1, keepdims=True)
        sd = grids.std(axis=-1, keepdims=True) + 1e-8
        grids = (grids - mu) / sd

    return grids



mat = sio.loadmat("10-20_system.mat")
# print(mat["map"])

edf_path = "data/arithmetic/Subject00_1.edf"
raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)

print(raw.ch_names)
print("\nSampling frequency:", raw.info["sfreq"])
print("Number of channels:", raw.info["nchan"])
print("Duration (sec):", raw.times[-1])

raw.drop_channels(["ECG ECG"])
epochs = mne.make_fixed_length_epochs(
    raw,
    duration=0.016,
    preload=True
)  # 16 ms epochs

print(raw.ch_names)

print(epochs.get_data().shape)

grids = epochs_to_grid(epochs, mat["map"], zscore=True)
print(grids.shape)  # (N, 5, 5, 8)
