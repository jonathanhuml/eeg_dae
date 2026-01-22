# load in map = ./egi256_to_1020.csv

# load in dataset from LocalizeMI (just do one series) using mne

# get channel names from mne

# use map to go from 256 names to 1020 system names

# drop channels not in 1020 system ---> should give us ~20 channels assuming none bad

# grid 5x5 to 8x8 ---> nn.predict


from pathlib import Path
import numpy as np
import pandas as pd

# scripts/ and data/ are siblings
ROOT = Path(__file__).resolve().parent.parent  # repo root
DERIV = ROOT / "data" / "localize-mi" / "derivatives" / "epochs" / "sub-01"

epochs_path = DERIV / "eeg" / "sub-01_task-seegstim_run-01_epochs.npy"
electrodes_path = DERIV / "eeg" / "sub-01_task-seegstim_electrodes.tsv"
channels_path = DERIV / "eeg" / "sub-01_task-seegstim_run-01_channels.tsv"

X = np.load(epochs_path)                 # epoched data
electrodes = pd.read_csv(electrodes_path, sep="\t")
channels = pd.read_csv(channels_path, sep="\t")

print("X shape:", X.shape)
print("electrodes columns:", electrodes.columns.tolist())
print("channels columns:", channels.columns.tolist())
print("channels head:\n", channels.head())
print("electrodes head:\n", electrodes.head())
