from sklearn.decomposition import PCA
from sklearn import metrics
import numpy as np

def apply_pca(X_train, X_test, n_components=8, seed=66):
    pca = PCA(n_components=n_components, random_state=seed)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    print("Explained Variance Ratios:")
    for i, v in enumerate(pca.explained_variance_ratio_):
        print(f"PC{i+1}: {v*100:.2f}%")
    return X_train_pca, X_test_pca, pca
