from scipy.sparse import csr_matrix
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np


###############################################
# ========== CLUSTERING ALGORITHMS ==========
###############################################

# initial_centroid_method = ['given']
# => 'given': use initial_centroids (DEFAULT)
# => 'random': generate randomly
# => 'k++': use kmeans++ to generate
class COPKMeans:
    def __init__(self, n_clusters, must_link=[], cannot_link=[], max_iter=300, initial_centroid_method='given', initial_centroids=[]):
        self.n_clusters = n_clusters
        self.must_link = must_link
        self.cannot_link = cannot_link
        self.max_iter = max_iter
        self.initial_centroid_method = initial_centroid_method
        self.initial_centroids = initial_centroids

    def fit(self, X):
        # Step 1: Initialize centroids randomly
        centroids = []
        if self.initial_centroid_method == 'given':
            centroids = self.initial_centroids
            assert(len(centroids) == self.n_clusters)
        elif self.initial_centroid_method == 'random':
            centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        elif self.initial_centroid_method == 'k++':
            centroids = self._kmeans_plus_plus_initialization(self, X)
        else:
            raise ValueError("Provide valid centroid generation method")
        labels = np.full(X.shape[0], -1)
        
        for _ in range(self.max_iter):
            new_labels = np.full(X.shape[0], -1)
            # new_labels = np.zeros(X.shape[0])
            
            # Step 2: Assign points to the nearest cluster considering constraints
            for i, point in enumerate(X):
                distances = euclidean_distances([point], centroids).flatten()
                
                # Sort clusters by distance
                sorted_clusters = np.argsort(distances)
                for cluster_id in sorted_clusters:
                    if self._satisfies_constraints(self, i, cluster_id, new_labels):
                        new_labels[i] = cluster_id
                        break
            
            # Step 3: Check for convergence
            if np.array_equal(labels, new_labels):
                break
            labels = new_labels
            
            # Step 4: Update centroids
            for cluster_id in range(self.n_clusters):
                points_in_cluster = X[labels == cluster_id]
                if len(points_in_cluster) > 0:
                    centroids[cluster_id] = np.mean(points_in_cluster, axis=0)
        
        self.labels_ = labels
        self.cluster_centers_ = centroids
    
    def _satisfies_constraints(self, point_index, cluster_id, labels):
        for (i, j) in self.must_link:
            if point_index == i and labels[j] != cluster_id:
                return False
            if point_index == j and labels[i] != cluster_id:
                return False
        
        for (i, j) in self.cannot_link:
            if point_index == i and labels[j] == cluster_id:
                return False
            if point_index == j and labels[i] == cluster_id:
                return False
        
        return True
    
    def _kmeans_plus_plus_initialization(self, X):
        # X is the dataset, k is the number of clusters
        centroids = []

        # Step 1: Initialize first centroid randomly
        centroids.append(X[np.random.choice(X.shape[0], replace=False)])

        # Step 2: Use k-means++ for the remaining centroids
        remaining_centroids_needed = self.n_clusters - 1
        if remaining_centroids_needed > 0:
            # Select remaining centroids using k-means++
            distances = np.min(euclidean_distances(X, centroids), axis=1) ** 2

            for _ in range(remaining_centroids_needed):
                probabilities = distances / distances.sum()
                next_centroid_idx = np.random.choice(X.shape[0], p=probabilities)
                centroids.append(X[next_centroid_idx])

                # Update distances with the new centroid
                new_distances = euclidean_distances(X, [X[next_centroid_idx]]) ** 2
                distances = np.minimum(distances, new_distances.flatten())
        return np.array(centroids)

##########################################
# ========== CLUSTERING PIPES ==========
##########################################

# Clustering():
# => labeled_embeddings: Array
# => unlabeled_embeddings: Array
# => labels: integer speaker labels similarly indexed to labeled_embeddings
class Clustering():
    def __init__(self, n_clusters, labeled_embeddings, unlabeled_embeddings, labels):
        self.labels = []
        self.n_clusters = n_clusters
        self.labeled_embeddings = labeled_embeddings
        self.unlabeled_embeddings = unlabeled_embeddings
        self.labels = labels
        self.combined_embeddings = np.vstack((labeled_embeddings, unlabeled_embeddings))

    # 'virtual' placeholder, gets overriden by other clustering methods
    def fit(self):
        pass

# Recommended: Using kneighbors and labels to create connectivity graph into constrained agglomerative clustering.
class ConstrainedAgglomerative(Clustering):
    def __init__(self, n_clusters, labeled_embeddings, unlabeled_embeddings, labels, n_neighbors=30):
        self.n_neighbors = n_neighbors
        super().__init__(n_clusters, labeled_embeddings, unlabeled_embeddings, labels)


    def fit(self):
        connectivity_knn = kneighbors_graph(self.combined_embeddings, n_neighbors=self.n_neighbors, mode='connectivity', include_self=False)
        n_labeled = len(self.labeled_embeddings)
        n_unlabeled = len(self.unlabeled_embeddings)

        # Connectivity graph for labeled embeddings
        labeled_connectivity = np.zeros((n_labeled, n_labeled))
        for i in range(n_labeled):
            for j in range(i + 1, n_labeled):
                if self.labels[i] == self.labels[j]:
                    labeled_connectivity[i, j] = 1
                    labeled_connectivity[j, i] = 1
        labeled_connectivity_sparse = csr_matrix(labeled_connectivity)
        connectivity = csr_matrix((n_labeled + n_unlabeled, n_labeled + n_unlabeled))
        connectivity[:n_labeled, :n_labeled] = labeled_connectivity_sparse
        connectivity = connectivity + connectivity_knn

        #  This line, if connectivity_knn run on only the unlabeled_embeddings?
        # connectivity[n_labeled:, n_labeled:] = connectivity_knn

        model = AgglomerativeClustering(
            linkage='ward', connectivity=connectivity, n_clusters=9
        ).fit(self.combined_embeddings)
        self.labels = model.labels_[len(self.labeled_embeddings):]

# (not recommended) two-pass approach to use agglomerative to influence centroids for semi-supervised COPKmeans
class AgglomerativeCOPKmeans(Clustering):
    def __init__(self, n_clusters, labeled_embeddings, unlabeled_embeddings, labels, max_iter=300):
        self.max_iter = max_iter
        super().__init__(n_clusters, labeled_embeddings, unlabeled_embeddings, labels)

    def _compute_aggromerative_centroids(self, embeddings, labels):
        unique_labels = np.unique(labels)
        centroids = []
        for label in unique_labels:
            # Get all points belonging to the current cluster
            cluster_points = embeddings[labels == label]
            # Compute the mean of the points in the cluster to approximate the centroid
            centroid = np.mean(cluster_points, axis=0)
            centroids.append(centroid)
        return np.array(centroids)

    def _generate_constraints(self, labeled_labels):
        must_link = []
        cannot_link = []
        
        # Generate must-link constraints
        for i in range(len(labeled_labels)):
            for j in range(i + 1, len(labeled_labels)):
                if labeled_labels[i] == labeled_labels[j]:
                    must_link.append((i, j))
                else:
                    cannot_link.append((i, j))
        
        return must_link, cannot_link

    def fit(self):
        # Pass 1: Use Agglomerative to find centroids
        agglo = AgglomerativeClustering(n_clusters=9, distance_threshold=None).fit(self.unlabeled_embeddings)
        agglo_labels = agglo.labels_
        agglo_centroids = self._compute_aggromerative_centroids(self, self.unlabeled_embeddings, agglo_labels)
        mlink, clink = self._generate_constraints(self, self.labels)

        cop_kmeans = COPKMeans(n_clusters=len(agglo_centroids), must_link=mlink, cannot_link=clink, initial_centroids=agglo_centroids)
        cop_kmeans.fit(self.combined_embeddings)
        self.labels = cop_kmeans.labels_[len(self.labeled_embeddings):]