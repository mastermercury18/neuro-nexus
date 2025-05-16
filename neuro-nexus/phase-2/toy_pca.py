import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FastICA
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Example toy model: define your classical and quantum models
class ClassicalNN(nn.Module):
    def __init__(self):
        super(ClassicalNN, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(100, 64),
            nn.ReLU(),
            nn.Linear(64, 16)  # Penultimate layer
        )
        self.classifier = nn.Linear(16, 2)

    def forward(self, x, return_features=False):
        features = self.feature_extractor(x)
        out = self.classifier(features)
        if return_features:
            return features
        return out

# Simulate test data
X_test = torch.randn(1000, 100)  # 1000 test samples, 100 features
y_test = torch.randint(0, 2, (1000,))  # binary labels

# Instantiate and run both networks
model_classical = ClassicalNN()
model_qnn = ClassicalNN()  # Replace with actual QNN model if available

with torch.no_grad():
    feats_classical = model_classical(X_test, return_features=True).numpy()
    feats_qnn = model_qnn(X_test, return_features=True).numpy()

# Dimensionality reduction: PCA
pca = PCA(n_components=2)
reduced_classical = pca.fit_transform(feats_classical)
reduced_qnn = pca.fit_transform(feats_qnn)

# Or ICA (alternative)
# ica = FastICA(n_components=2)
# reduced_classical = ica.fit_transform(feats_classical)
# reduced_qnn = ica.fit_transform(feats_qnn)

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(reduced_classical[:, 0], reduced_classical[:, 1], c=y_test, cmap='coolwarm', s=10)
axes[0].set_title("Classical NN Latent Space (PCA)")

axes[1].scatter(reduced_qnn[:, 0], reduced_qnn[:, 1], c=y_test, cmap='coolwarm', s=10)
axes[1].set_title("QNN Latent Space (PCA)")

plt.show()
