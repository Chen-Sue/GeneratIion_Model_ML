
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
from sklearn.manifold import TSNE, MDS
from sklearn.cluster import KMeans, DBSCAN
from sklearn import preprocessing, metrics

import config
from utils import read_data, save_data

seed = config.seed
n_clusters = config.n_clusters
n_components = config.n_components

colors = ['#4EACC5', '#FF9C34']
file_location = os.getcwd()

x = read_data(file_location=file_location + r'\data', name='x')
y = read_data(file_location=file_location+r'\data', name='percolation')

pca = PCA(n_components=n_components)
x = pca.fit_transform(x)

k_means = KMeans(n_clusters=n_clusters, random_state=seed)
k_means = k_means.fit(x)
k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0)
k_means_labels = metrics.pairwise.pairwise_distances_argmin(x, k_means_cluster_centers)


label_pred = k_means.labels_ #获取聚类标签
print(label_pred[:20])

class1 = y[np.where(label_pred==0)]
class2 = y[np.where(label_pred==1)]
print(class1[:20])
print(max(class1), min(class1))
print(max(class2), min(class2))

# centroids = k_means.cluster_centers_ #获取聚类中心
# inertia = k_means.inertia_ # 获取聚类准则的总和
# mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
# #这里'or'代表中的'o'代表画圈，'r'代表颜色为红色，后面的依次类推
# color = 0
# j = 0 
# for i in label_pred:
#     plt.plot([x[j:j+1,0]], [x[j:j+1,1]], mark[i], markersize = 5)
#     j +=1
# plt.show()


# # 需要知道每个类别有哪些参数
# C_i = k_means.predict(x)
# print(C_i)
# # 还需要知道聚类中心的坐标
# Muk = k_means.cluster_centers_
# print(Muk)

# # 画图
# plt.scatter(x[:,0], x[:,1], c=C_i, cmap=plt.cm.Paired)
# # 画聚类中心
# plt.scatter(Muk[:,0], Muk[:,1], marker='*', s=60)
# for i in range(2):
#     plt.annotate('center'+str(i+1), (Muk[i,0], Muk[i,1]))
# plt.show()


# fig = plt.figure(figsize=(6.5, 4))
# plt.plot(x[np.where(x[:,0]==0), 0], x[np.where(x[:,0]==0), 1],
#             c=y[np.where(x[:,0]==0)], marker='.')	# 将同一类的点表示出来
# plt.show()

# # K-means
# ax = fig.add_subplot(1, 1, 1)
# for k, col in zip(range(n_clusters), colors):
#     my_members = k_means_labels == k		# my_members是布尔型的数组（用于筛选同类的点，用不同颜色表示）
#     cluster_center = k_means_cluster_centers[k]
#     ax.plot(x[my_members, 0], x[my_members, 1], 'w',
#             markerfacecolor=y[np.where(x[0] == 0)], 
#             marker='.')	# 将同一类的点表示出来
#     ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
#             markeredgecolor='k', marker='o')	# 将聚类中心单独表示出来

# plt.subplot(1,1,1) 
# plt.plot(l, x_oringin, c=kmeans.labels_, marker='o')
# # plt.errorbar(l, x_oringin, color='dodgerblue', xerr=0.3)
# plt.grid(True, linestyle='--')
# plt.xlabel('principal component')
# plt.ylabel('$\lambda_l$')
# plt.xticks(np.arange(11))
# plt.title('PCA')
# # plt.yscale('log') 
# # plt.legend()
# plt.savefig(r'.\figure\eigenvalues.pdf')
# plt.show()