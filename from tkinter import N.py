from tkinter import N
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_moons, make_circles

# c) make_blobs
X, y = make_blobs(n_samples=400, n_features=2, centers=5, cluster_std=1.0, random_state=0)

# h) make_moons
# X, y = make_moons(n_samples=400, noise=0.1)

# h) make_circles
# X, y = make_circles(n_samples=400, noise=0.01)

# d) plot dataset
fig = plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y)
# plt.show()

# e) clustering K=5
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)
centers = kmeans.cluster_centers_

# bonus: elbow plot
inertias = []
for K in range(1,10):
    kmeans = KMeans(n_clusters=K)
    kmeans.fit(X)
    inertias += [kmeans.inertia_]

plt.figure()
plt.plot(range(1,10), inertias, 'bx-')

# f) plot clustering
fig2 = plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans)
plt.plot(centers[:, 0], centers[:, 1], 'ro', ms=20)
plt.show()