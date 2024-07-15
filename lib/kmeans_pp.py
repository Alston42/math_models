import numpy as np

'''
Params:
X: ndarray, shape (n_samples, n_features)
k: int, number of clusters
'''
def kmeans_pp(X, k):
    # Initialize the list of centroids
    centroids = []
    
    # randomly choose one point as the first centroid
    centroids.append(X[np.random.choice(X.shape[0])])
    
    # To cal the dis between each data point and the centroid
    distances = np.array([np.linalg.norm(X - centroid, axis=1) for centroid in centroids])
    
    for _ in range(k - 1):
        # 计算每个数据点到聚类中心的距离之和
        sum_distances = np.sum(distances, axis=0)
        
        # 计算每个数据点被选为下一个聚类中心的概率
        probabilities = distances / sum_distances
        
        # 选择下一个聚类中心
        centroids.append(X[np.random.choice(X.shape[0], p=probabilities)])
        
        # 更新距离矩阵
        distances = np.array([np.linalg.norm(X - centroid, axis=1) for centroid in centroids])
    
    return np.array(centroids)

# 示例
X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 30]])
k = 2
centroids = kmeans_pp(X, k)
print(centroids)
