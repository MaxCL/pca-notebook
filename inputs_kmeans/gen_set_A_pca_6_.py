#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

df = pd.read_csv("input_pca_set_A_.csv")
X = df.select_dtypes(include=[np.number]).copy()

X = X.dropna(axis=0, how="any")

pca = PCA()  # n_components=None -> all components
scores = pca.fit_transform(X.values)  # shape: (n_samples, n_components_total)

n_out = min(6, scores.shape[1])
scores_6 = scores[:, :n_out]

cols = [f"PC{i}" for i in range(1, n_out + 1)]
pd.DataFrame(scores_6, index=X.index, columns=cols).to_csv(
    "set_A_pca_6_.csv", index_label="row_index"
)

