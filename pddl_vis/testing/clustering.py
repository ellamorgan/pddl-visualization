import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import metrics
import matplotlib.pyplot as plt




def k_neighbours_test(train_embed, train_labels, test_embed, test_labels):

    neighbours = KNeighborsClassifier(n_neighbors=5).fit(train_embed, train_labels)
    result = neighbours.predict(test_embed)

    correct = 100 * np.sum(result == test_labels) / len(result)
    print("\nPercent correct: %.2f%%" % (correct))
    print()

    return correct, train_labels


def clustering_test(embeddings, labels, n_clusters):

    clusters = KMeans(n_clusters=n_clusters).fit(embeddings)

    pca = PCA(2)

    df = pca.fit_transform(embeddings)
    u_labels = np.unique(clusters.labels_)

    for i in u_labels:
        plt.scatter(df[clusters.labels_ == i , 0] , df[clusters.labels_ == i , 1] , label = i)
    plt.savefig('result.png')

    homogeneity = metrics.homogeneity_score(labels, clusters.labels_)
    completeness = metrics.completeness_score(labels, clusters.labels_)
    v_measure = metrics.v_measure_score(labels, clusters.labels_)

    print("Homogeneity: %0.3f" % homogeneity)
    print("Completeness: %0.3f" % completeness)
    print("V-measure: %0.3f" % v_measure)
    print()

    return homogeneity, completeness, v_measure