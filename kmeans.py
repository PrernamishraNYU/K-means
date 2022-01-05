import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances


class Kmeans:
   
    def __init__(self, k, seed=None, value="euclid", max_iter=400):
        self.k = k
        self.seed = seed
        self.value = value
        if self.seed is not None:
            np.random.seed(self.seed)
        self.max_iter = max_iter

    def initialise_centroids(self, data):
        
        initial_centroids = [0,8,16]
        #print(initial_centroids)
        self.centroids = data[initial_centroids]

        return self.centroids

    def assign_clusters(self, data):
        

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        dist_to_centroid = pairwise_distances(data, self.centroids, metric='euclidean')
        if self.value == "cosine":
            new_distance = 1 - np.square(dist_to_centroid)
            self.cluster_labels = np.argmax(new_distance, axis=1)
        else:
            self.cluster_labels = np.argmin(dist_to_centroid, axis=1)

        return self.cluster_labels

    def update_centroids(self, data):
        
        self.centroids = np.array([data[self.cluster_labels == i].mean(axis=0) for i in range(self.k)])

        return self.centroids

    def predict(self, data):
        
        return self.assign_clusters(data)

    def fit_kmeans(self, data):
        
        self.centroids = self.initialise_centroids(data)

        # Main kmeans loop
        for iter in range(self.max_iter):

            self.cluster_labels = self.assign_clusters(data)
            self.centroids = self.update_centroids(data)
            if iter % 100 == 0:
                print("Running Model Iteration %d " % iter)

        print("Model finished running")
        return self

