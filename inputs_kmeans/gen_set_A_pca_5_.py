import pandas as pd
from sklearn.decomposition import PCA

# Load data and keep only numeric columns
df = pd.read_csv("input_pca_set_A_.csv")
X = df.select_dtypes(include="number").dropna(axis=0, how="any")

# Fit "full" PCA (all components)
pca = PCA()  # n_components=None -> all components
scores = pca.fit_transform(X.values)  # (n_samples, n_components_total)

# Extract first 5 PCs (or all if fewer exist)
n_out = min(5, scores.shape[1])
scores_5 = scores[:, :n_out]

# Save to CSV
cols = [f"PC{i}" for i in range(1, n_out + 1)]
pd.DataFrame(scores_5, index=X.index, columns=cols).to_csv(
    "set_A_pca_5_.csv", index_label="row_index"
)

