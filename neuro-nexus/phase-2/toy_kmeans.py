import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Sample feedforward neural network
class ToyNet(nn.Module):
    def __init__(self):
        super(ToyNet, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(100, 64),
            nn.ReLU(),
            nn.Linear(64, 16)  # We'll extract this layer's output
        )
        self.classifier = nn.Linear(16, 2)

    def forward(self, x, return_features=False):
        f = self.features(x)
        if return_features:
            return f
        return self.classifier(f)

# Generate dummy data
X_test = torch.randn(1000, 100)
y_test = torch.randint(0, 2, (1000,))

# Load your model
model = ToyNet()

# Get penultimate layer activations
with torch.no_grad():
    activations = model(X_test, return_features=True).numpy()

# Step 1: Apply k-means clustering
n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(activations)

# Step 2: Visualize with PCA
pca = PCA(n_components=2)
activations_2d = pca.fit_transform(activations)

# Step 3: Plot the clustering results
plt.figure(figsize=(8, 6))
scatter = plt.scatter(activations_2d[:, 0], activations_2d[:, 1],
                      c=cluster_labels, cmap='viridis', s=15)
plt.title("k-Means Clustering on Neural Activations (PCA projection)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(scatter, label='Cluster')
plt.grid(True)
plt.show()

# Optional: Evaluate clustering quality
sil_score = silhouette_score(activations, cluster_labels)
print(f"Silhouette Score: {sil_score:.3f}")
