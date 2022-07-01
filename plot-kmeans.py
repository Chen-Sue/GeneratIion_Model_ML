
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
from sklearn.manifold import TSNE, MDS
from sklearn.cluster import KMeans, DBSCAN
from sklearn import preprocessing, metrics

from utils import read_data, save_data

seed = 12345
n_clusters = 2
n_components = 2
colors = ['#4EACC5', '#FF9C34']

file_location = os.getcwd()

x = read_data(file_location=file_location + r"\data", name="x_1000")
y = read_data(file_location=file_location+r"\data", 
    name="percolation_1000")

# pca = PCA(n_components=n_components)
# x = pca.fit_transform(x)

k_means = KMeans(n_clusters=n_clusters, random_state=seed).fit(x)
k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0)
k_means_labels = metrics.pairwise.pairwise_distances_argmin(x, k_means_cluster_centers)



fig = plt.figure(figsize=(6.5, 4))
plt.plot(x[np.where(x[:,0]==0), 0], x[np.where(x[:,0]==0), 1],
            c=y[np.where(x[:,0]==0)], 
            marker='.')	# 将同一类的点表示出来
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
# plt.plot(l, x_oringin, c=kmeans.labels_, marker="o")
# # plt.errorbar(l, x_oringin, color="dodgerblue", xerr=0.3)
# plt.grid(True, linestyle='--')
# plt.xlabel('principal component')
# plt.ylabel('$\lambda_l$')
# plt.xticks(np.arange(11))
# plt.title("PCA")
# # plt.yscale("log") 
# # plt.legend()

# plt.savefig(r".\figure\eigenvalues.pdf")
# plt.savefig(r".\figure\eigenvalues.eps")
plt.show()