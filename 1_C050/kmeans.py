import pandas as pd
from sklearn.cluster import KMeans

filename = 'D:/A LBYYY/数学建模/~数学建模算法学习/2023 C题/聚类分析.xlsx' 
column_names = ['lable','销量', '最大销量', '平均销量']
data = pd.read_excel(filename, names=column_names)
seed=7
array = data.values
X_samples = array[:,1:4]

kmeans = KMeans(n_clusters=4, init='k-means++', random_state=seed)
kmeans.fit(X_samples)
# 获取样本的聚类标签
sample_clusters = kmeans.labels_
# 将聚类结果保存到数据集中
data['Cluster'] = sample_clusters
# 输出聚类结果
print(data)
# 保存带有聚类结果的数据集到文件中
print("Cluster centers:")
print(kmeans.cluster_centers_)
output_filename = 'D:/A LBYYY/数学建模/~数学建模算法学习/2023 C题/聚类分析输出.xlsx'  # 修改为输出文件路径
data.to_excel(output_filename, index=False)
