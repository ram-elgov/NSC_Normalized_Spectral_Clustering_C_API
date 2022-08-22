from sklearn.cluster import SpectralClustering
from sklearn.datasets._samples_generator import make_blobs
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
import sklearn.cluster as skl_cluster
import numpy as np
np.set_printoptions(precision=3)
n = 10
k = 3
d = 2
np.random.seed(1)
# X, _ = make_blobs(n_samples=400, centers=4, cluster_std=1)
# print(X[0][0])
# plt.scatter(X[:, 0], X[:, 1])
# plt.show()
#
# sc = SpectralClustering(n_clusters=4, n_init=1).fit(X)
# print(len(sc.labels_))
# labels = sc.labels_
# plt.scatter(X[:, 0], X[:, 1], c=labels)
# plt.show()
# f = plt.figure()
# f.add_subplot(2, 2, 1)
# for i in range(2, 6):
#     sc = SpectralClustering(n_clusters=i).fit(X)
#     f.add_subplot(2, 2, i - 1)
#     plt.scatter(X[:, 0], X[:, 1], s=5, c=sc.labels_, label="n_cluster-" + str(i))
#     plt.legend()
# plt.show()
# X, y = make_blobs(n_samples=10, centers=3, n_features=2, random_state=0)
# print(X)
# print(y)
# X, y = make_blobs(n_samples=[3, 3, 4], centers=None, n_features=2, random_state=0)
# print(X)
# print(y)
# fist
# X, _ = make_blobs(n_samples=n, centers=k, n_features=d, shuffle=True, random_state=31)
# with open("input_1.txt", mode="w", encoding = 'utf-8') as f:
#     for dp in X:
#         for i in range(d):
#             f.write(f"{dp[i]:.3f}")
#             if i != d - 1:
#                 f.write(",")
#         f.write("\n")
circles, circles_clusters = make_circles(n_samples=400, noise=.01, random_state=0)

# cluster with kmeans
Kmean = skl_cluster.KMeans(n_clusters=2)
Kmean.fit(circles)
clusters = Kmean.predict(circles)

# plot the data, colouring it by cluster
plt.scatter(circles[:, 0], circles[:, 1], s=15, linewidth=0.1, c=clusters,cmap='flag')
plt.show()

# cluster with spectral clustering
model = skl_cluster.SpectralClustering(n_clusters=2, affinity='nearest_neighbors', assign_labels='kmeans')
labels = model.fit_predict(circles)
plt.scatter(circles[:, 0], circles[:, 1], s=15, linewidth=0, c=labels, cmap='flag')
plt.show()



