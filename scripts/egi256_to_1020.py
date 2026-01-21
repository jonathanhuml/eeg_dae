import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist


def normalize(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)

df = pd.read_csv("montage_maps.csv")

df_1020 = df[df["layout"].str.contains("standard_1020", case=False)]
df_egi  = df[df["layout"].str.contains("EGI_256", case=False)]

print("10–20:", len(df_1020))
print("EGI  :", len(df_egi))

X_1020 = df_1020[["x", "y", "z"]].to_numpy()
X_egi  = df_egi[["x", "y", "z"]].to_numpy()

channels_1020 = df_1020["channel"].to_list()
channels_egi  = df_egi["channel"].to_list()

X_1020 = normalize(X_1020)
X_egi  = normalize(X_egi)

D = cdist(X_1020, X_egi, metric="euclidean")

used = set()
one_to_one = {}   # 10–20 label -> EGI label

for i, ch in enumerate(channels_1020):
    for j in np.argsort(D[i]):
        egi = channels_egi[j]
        if egi not in used:
            one_to_one[ch] = egi
            used.add(egi)
            break


pd.DataFrame(
    [{"10_20": ch, "egi": egi} for ch, egi in one_to_one.items()]
).to_csv("egi256_to_1020.csv", index=False)
