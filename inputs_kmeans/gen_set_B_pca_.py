import sys
import pandas as pd
from sklearn.decomposition import PCA

req_n = int(sys.argv[1])

df = pd.read_csv("input_pca_set_B_.csv")
X = df.select_dtypes(include="number").dropna(axis=0, how="any")

if X.empty:
    print("Error: no numeric data found after dropping NaNs.")
    sys.exit(1)

pca = PCA()  # n_components=None -> all possible components
scores = pca.fit_transform(X.values)  # shape = (n_samples, n_components_total)

n_out = min(req_n, scores.shape[1])
scores_sel = scores[:, :n_out]

out_name = f"set_B_pca_{req_n}_.csv"  # keep requested N in filename
cols = [f"PC{i}" for i in range(1, n_out + 1)]
pd.DataFrame(scores_sel, index=X.index, columns=cols).to_csv(out_name, index_label="row_index")

print(f"Wrote {out_name} with {n_out} PC columns (requested {req_n}).")
