import pandas as pd
from sklearn.decomposition import PCA

df = pd.read_csv("input_pca_set_A_.csv")
X = df.select_dtypes(include="number").dropna(axis=0, how="any")

pca = PCA()  # n_components=None -> all components
scores = pca.fit_transform(X.values)

n_out = min(4, scores.shape[1])
scores_4 = scores[:, :n_out]

cols = [f"PC{i}" for i in range(1, n_out + 1)]
pd.DataFrame(scores_4, index=X.index, columns=cols).to_csv(
    "set_A_pca_4_.csv", index_label="row_index"
)

