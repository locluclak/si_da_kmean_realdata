import numpy as np
from collections import defaultdict

def kmeans(X, k, max_iters=300, tol=1e-4, seed=42):
    n = X.shape[0]
    rng = np.random.RandomState(seed)  # Create a separate random generator
    initial_indices = rng.choice(n, k, replace=False) 
    initial_centroids = X[initial_indices]
    centroids = initial_centroids.copy()
    cluster_labels_all = []
    cluster_members_all = []

    for _ in range(max_iters):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        cluster_labels_all.append(labels.copy())
        cluster_members = defaultdict(set)
        for idx, label in enumerate(labels):
            cluster_members[label].add(idx)
        cluster_members_all.append(cluster_members)
        new_centroids = np.zeros_like(centroids)
        for cluster in range(k):
            instances_in_cluster = X[labels == cluster]
            if len(instances_in_cluster) > 0:
                new_centroids[cluster] = instances_in_cluster.mean(axis=0)
            else:
                new_centroids[cluster] = centroids[cluster]
        if np.linalg.norm(new_centroids - centroids) < tol:
            break
        centroids = new_centroids

    return initial_indices, cluster_labels_all, cluster_members_all