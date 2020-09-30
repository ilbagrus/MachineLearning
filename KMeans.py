from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np


def choose_centroid(point):
    return np.argmin(np.linalg.norm((centroids-point), axis=1))


def k_means():
    global centroids
    C = np.zeros(X.shape)
    new_centroids = np.zeros((clasters_num, 2))
    choose = np.vectorize(choose_centroid, signature='(a)->()')

    for iteration in range(100):
        C = choose(X)

        for cluster in range(0, clasters_num):
            temp = X[C == cluster]
            if len(temp) != 0:
                new_centroids[cluster] = np.sum(temp, axis=0)/len(temp)
            else:
                new_centroids[cluster] = X[np.random.randint(3000, size=1)]

        if np.array_equal(new_centroids, centroids):
            break
        centroids = new_centroids.copy()
    return C


def rate_clusters():
    clusters_sum = 0
    for cluster in range(0, clasters_num):
        temp = X[clusters == cluster]
        clusters_sum += np.sum(np.linalg.norm((centroids[cluster] - temp), axis=1))
    return clusters_sum


def image_clusters():
    for cluster in range(clasters_num):
        temp = X[clusters == cluster]
        plt.plot(temp[:, 0], temp[:, 1], linestyle='', marker='.', markersize=10)
    plt.plot(centroids[:, 0], centroids[:, 1], linestyle='', marker='*', markersize=30)
    plt.show()


def image_e_d():
    x = np.arange(1, clasters_num + 1)
    plt.plot(x, E)
    plt.show()

    x = np.arange(2, clasters_num)
    plt.plot(x, D)
    plt.show()


centers = [[-1, -1], [0, 1], [1, -1]]
X, _ = make_blobs(n_samples=3000, centers=centers, cluster_std=0.5)

K = 10
E = []

for clasters_num in range(1, K + 1):
    centroids = X[np.random.randint(3000, size=clasters_num)]
    clusters = k_means()
    E.append(rate_clusters())
    # image_clusters()

D = []
for i in range(1, len(E)-1):
    D.append(abs((E[i] - E[i+1]))/abs((E[i-1]-E[i])))

image_e_d()

clasters_num = np.argmin(D) + 2
centroids = X[np.random.randint(3000, size=clasters_num)]
clusters = k_means()
image_clusters()
