from mpi4py import MPI
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import random

comm = MPI.COMM_WORLD
rank = comm.Get_rank

if rank == 0:

    # Generate random sample with 3 clusters
    X, y_true = make_blobs(n_samples=500, centers=3, cluster_std=0.80, random_state=0)
    gen = np.random.RandomState(4)
    datapoints = np.dot(X, gen.randn(2,2))

    max_iter = 50     # Maximum number of iterations allowed
    order = 3
    mu_centroid = np.array(random.sample(list(datapoints), order))    # randomly selected centroids to begin
    ind_cur = []    # These are used to check if the centroids
    ind_prev = []    # are still changing every iteration
    clusters = []
    Sinv = np.linalg.inv(np.cov(datapoints.T))      # the inverse of the covariance matrix generated from the data
    colors = ['Red', 'Yellow', 'Green']

    # Broadcast initial data for other processes
    comm.bcast(Sinv, root=0)

    # Run until reached maximum allowed iterations
    for t in range(max_iter):

        ind_prev = ind_cur
        clusters = []
    
        comm.bcast(mu_centroid, root=0)

        for i in range(1,comm.Get_size):
            print(datapoints[i-1])
            comm.send(datapoints[i-1], i, 1)
        
        i = comm.Get_size
        while(i < len(datapoints)):
            cluster = np.empty(shape=2)
            status = MPI.Status()
            comm.recv(cluster, MPI.ANY_SOURCE, 1, status)
            comm.send(datapoints[i], status.Get_source, 1)
            i += 1
            clusters.append(cluster)

        getCentroids = []

        for j in range(len(clusters)):
            getCentroids.append(clusters[j][0])

        ind_cur = getCentroids

        # Update the centroids
        temp = []
        for c in range(order):
            temp.append(np.mean([datapoints[x] for x in range(
                len(datapoints)) if getCentroids[x] == c], axis=0))
        mu_centroid = temp

        # Send reset signal
        for i in range(1, comm.Get_size):
            resetSig = [np.nan, np.nan]
            comm.send(resetSig, i, 1)

        # Stop looping if the centroids are not changing
        if ind_cur == ind_prev:
            break

    # Send reset signal
    resetSig = -1
    comm.bcast(resetSig, root=0)

    # Create a scattered plot
    for i in range(order):
        arr = []
        for j in range(len(clusters)):
            if clusters[j][0] == i:
                arr.append(clusters[j][1])
        for k in range(len(arr)):
            plt.scatter(arr[k][0], arr[k][1], marker='.', c=colors[i])

    # Scatter centroids
    plt.scatter(np.array(mu_centroid)[:, 0], np.array(
        mu_centroid)[:, 1], color='black', marker='s')
    plt.show()

else:
    mu_centroid = []
    Sinv = []
    clusters = []
    comm.bcast(Sinv, root=0)

    while True:
        comm.bcast(mu_centroid, root=0)

        if mu_centroid == -1:
            break

        while True:
            datapoint = np.empty(shape=2)
            comm.recv(datapoint, 0, 1, None)

            if datapoint == [np.nan, np.nan]:
                break

            distance = []

            # Calculate mahalanobis distance between every point and centroid
            for j in mu_centroid:
                np_array_i = np.array(datapoint)
                np_array_j = np.array(j)
                mahalanobis = np.sqrt(np.dot(np.dot((np_array_i-np_array_j).T, Sinv), np_array_i-np_array_j))
                distance.append(mahalanobis) 

            cluster = [np.argmin(distance), datapoint]   # find the minimum distance
            comm.send(cluster, 0, 1)