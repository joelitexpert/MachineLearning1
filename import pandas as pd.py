import pandas as pd

# Annahme: Die CSV-Datei befindet sich im gleichen Verzeichnis wie dein Skript.
data = pd.read_csv('customer-segmentation-tutorial-in-python/Mall_Customers.csv')

# Daten anzeigen
print(data.head())
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X = data.iloc[:, 2:5]  

# Annahme: Auswahl der relevanten Spalten
pca.fit(X)
X_pca = pca.transform(X)
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Customer Data')
plt.show()

# explained_variance = pca.explained_variance_ratio_
plt.bar(range(2), explained_variance)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.xticks(range(2), ['PC1', 'PC2'])
plt.title('Explained Variance Ratio')
plt.show()