import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the data
data = pd.DataFrame({
    'gender': ['Male', 'Female', ...],  # Replace with your gender data
    'age': [19, 15, ...],  # Replace with your age data
    'height': [39, 2, ...],  # Replace with your height data
    'weight': [15, 81, ...]  # Replace with your weight data
})

# Preprocess the data (if needed)
# You may need to convert categorical variables (like gender) to numerical format

# Perform feature scaling (if needed)
# You can use techniques like StandardScaler or MinMaxScaler from sklearn.preprocessing

# Perform dimensionality reduction using PCA
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data[['age', 'height', 'weight']])

# Visualize the data
plt.scatter(data_pca[:, 0], data_pca[:, 1], c='b', alpha=0.5)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Data Visualization')
plt.show()

# Perform clustering using K-Means
kmeans = KMeans(n_clusters=2)
clusters = kmeans.fit_predict(data[['age', 'height', 'weight']])

# Add the cluster labels to the original data
data['cluster'] = clusters

# Print the cluster assignments
print(data[['gender', 'age', 'height', 'weight', 'cluster']])

# Evaluate the clustering results (optional)
# You can use metrics like silhouette score or visual inspection of clusters

# Bonus: PCA for data preprocessing
data_preprocessed = pca.fit_transform(data[['age', 'height', 'weight', 'cluster']])

# Use the preprocessed data for further analysis or modeling

# Additional steps:
# You can further refine the code, handle missing values, optimize hyperparameters, etc. based on your specific requirements.