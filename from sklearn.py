from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np

# b)
X, y = make_moons(n_samples=400, noise=0.1, random_state=0)

# c) 
fig = plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

# d)
dbscan = DBSCAN(eps=0.22, min_samples=9)
# e)
dbscan.fit(X)
y_dbscan = dbscan.labels_

fig = plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y_dbscan)
plt.show()

# f)
best_score = -1
best_eps = 0
best_min_samples = 0
best_num_clusters = 0

best_values = {}

for eps in np.arange(0.05, 0.5, 0.01):
    for min_samples in range(1, 15, 1):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan.fit(X)
        y_dbscan = dbscan.labels_

        num_clusters = np.max(y_dbscan)+1

        if num_clusters > 1:
            score = silhouette_score(X, y_dbscan)
            if score > 0:

                if not (num_clusters in best_values.keys()) or \
                    score > best_values[num_clusters]["score"]:
                    best = {"score": score, 
                            "eps": eps, 
                            "min_samples": min_samples}
                    best_values[num_clusters] = best

for key, value in best_values.items():
    best_score = value["score"]
    best_min_samples = value["min_samples"]
    best_eps = value["eps"]
    best_num_clusters = key
    print("%.2f - %d : %.2f, %d" % (best_score, best_num_clusters, best_eps, best_min_samples))

# then choose the values for eps and min_samples which give you
# a large silhouette score but also a small number of clusters