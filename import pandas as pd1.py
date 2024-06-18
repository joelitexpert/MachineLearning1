import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Annahme: Die CSV-Datei befindet sich im gleichen Verzeichnis wie dein Skript.
data = pd.read_csv('customer-segmentation-tutorial-in-python/Mall_Customers.csv')

# Auswahl der relevanten Spalten (z.B., Annual Income und Spending Score)
X = data.iloc[:, [3, 4]].values

# Wähle die Anzahl der Cluster (z.B., 5 Cluster)
kmeans = KMeans(n_clusters=5)
kmeans.fit(X)

# Füge die Clusterlabels dem ursprünglichen Datensatz hinzu
data['Cluster'] = kmeans.labels_

# Plot der Cluster
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='rainbow')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('K-Means Clustering')
plt.show()