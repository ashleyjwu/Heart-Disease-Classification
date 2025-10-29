from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def elbow_method(X, seed=66):
    inertias = []
    clusters = range(2, 31)
    for k in clusters:
        km = KMeans(n_clusters=k, random_state=seed)
        km.fit(X)
        inertias.append(km.inertia_)
    plt.figure(figsize=(8,5))
    plt.plot(clusters, inertias, marker='x', linestyle='--')
    plt.title("Elbow Method")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.grid(True)
    plt.show()
