
from sklearn.cluster import KMeans
def kMeans_use(embedding,cluster_number):
    kmeans = KMeans(n_clusters=cluster_number,
                init="k-means++",
                random_state=0)
    pred = kmeans.fit_predict(embedding)
    return pred