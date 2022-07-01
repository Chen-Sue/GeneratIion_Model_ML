import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
from sklearn.manifold import TSNE, MDS
from sklearn.cluster import KMeans, DBSCAN
from sklearn import preprocessing 

from utils import read_data, save_data

file_location = os.getcwd()

x = read_data(file_location=file_location + r"\data", name="x_1000")
print(x.shape)

n_components = 784
l = np.arange(1, n_components+1, 1)

pca = PCA(n_components=n_components)
# kernels = ['linear','poly','rbf','sigmoid']
# pca = KernelPCA(n_components=n_components, kernel=kernels[0])
# pca = KernelPCA(n_components=n_components, kernel=kernels[1])
# pca = KernelPCA(n_components=n_components, kernel=kernels[2])
# pca = KernelPCA(n_components=n_components, kernel=kernels[3])
# pca = IncrementalPCA(n_components=n_components)
# pca = TSNE(n_components=1)  # one-component TSNE
# pca = TSNE(n_components=n_components)  # two-component TSNE
# pca = MDS(n_components=1)  # one-component MDS
# pca = MDS(n_components=2)  # two-component MDS
# pca = DBSCAN()
# pca = pca.fit_predict(x_oringin) # c = y_pred

x_new_oringin = pca.fit_transform(x)
x_oringin = pca.explained_variance_ratio_
# save_data(file_location=file_location + r"\data", 
#           name="x_oringin_{}".format(name), value=x_oringin)    

fig = plt.figure(figsize=(7.5, 5))

plt.subplot(1,1,1) 
plt.scatter(l, x_oringin, c='dodgerblue', marker="v", s=10, linewidth=2)
# plt.errorbar(l, x_oringin, color="dodgerblue", xerr=0.3)
plt.grid(True, linestyle='--', linewidth=1.5)
plt.xlabel('$n$', fontsize=16)
plt.ylabel('${\widetilde{\lambda}}_n$', fontsize=16)
# plt.xticks(np.arange(n_components))
# plt.yticks(np.arange(n_components),)
plt.ylim(0.8e-4, 1.2e-1)
# plt.title("PCA")
plt.yscale("log") 
plt.xticks(size=14)
plt.yticks(size=14)
# plt.legend()
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)

# plt.plot(p, y_pred_average, c='indianred', marker="o", 
#     label="predicted $\Pi(p,L)$ by cVAE and CNN-1", linewidth=2)
# ax.text(0.41, 0.93, '(a)', fontsize=14)
# plt.xlabel('permeability', fontsize=14)
# plt.ylabel('$\Pi(p,L)$', fontsize=14)
# plt.xticks(size=12)
# plt.yticks(size=12)
# plt.xlim(0.39, 0.81)
# plt.ylim(-0.09, 1.09)
# plt.grid(True, linestyle='--', linewidth=1.5)
# plt.legend(loc="lower right")



plt.savefig(r".\figure\eigenvalues.pdf")
plt.savefig(r".\figure\eigenvalues.eps")
plt.show()