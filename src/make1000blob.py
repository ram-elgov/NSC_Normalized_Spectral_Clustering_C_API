from sklearn import datasets
import numpy as np
def save_dataset(dataset, filename):
    lines = map(stringify_row, dataset)
    with open(filename + ".csv", "w") as f:
        f.writelines(lines)
     
def stringify_row(row):
    row = map(str, row)
    return ",".join(row) + "\n"

# datapoints      
save_dataset(datasets.make_blobs(n_samples=1000, centers=3, cluster_std=5, n_features=10, shuffle=True, random_state=31)[0], "1000_blobs_10_feat")

# symmetric matrix
a = np.random.rand(1000, 1000)
m = np.tril(a) + np.tril(a, -1).T
save_dataset(m, "jacobi_input_10_6")