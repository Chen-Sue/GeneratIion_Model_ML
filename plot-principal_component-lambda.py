import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
from sklearn.manifold import TSNE, MDS
from sklearn.cluster import KMeans, DBSCAN
from sklearn import preprocessing 

from utils import read_data, save_data

file_location = os.getcwd() + r"\VAE"

x = read_data(file_location=file_location + r"\data", name="x_1000")
print(x.shape)
percolation = read_data(file_location=file_location + r"\data", 
    name="percolation_1000").reshape(-1, )

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

# fig = plt.figure(figsize=(6, 8))
# plt.subplot(211)
fig = plt.figure(figsize=(8, 4))
plt.subplot(121)  
plt.scatter(l, x_oringin, c='dodgerblue', marker="v", s=10, linewidth=2)
# plt.errorbar(l, x_oringin, color="dodgerblue", xerr=0.3)
plt.xlabel('$j$', fontsize=16)
plt.ylabel('${\widetilde{\lambda}}_j$', fontsize=16)
# plt.xticks(np.arange(n_components))
# plt.yticks(np.arange(n_components),)
plt.ylim(0.8e-4, 4e-1)
# plt.title("PCA")
plt.yscale("log") 
plt.xticks(size=12)
plt.yticks(size=12)
# plt.legend()
plt.text(0.0, 1.5e-1, '(a)', fontsize=14)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
from pylab import *
tick_params(which='major', width=2)
tick_params(which='minor', width=1)
ax.xaxis.set_major_locator(plt.MultipleLocator(100))
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

x = read_data(file_location=file_location + r"\data" , name="x_1000")
x = pca.fit_transform(x)

# plt.subplot(212) 
plt.subplot(122) 
h1 = plt.scatter(x[:, 0], x[:, 1], c=percolation, marker="o", linewidth=2)
# plt.title(r"1st principal component - 2nd principal component")
plt.xlabel("1st principal component", fontsize=13)
plt.ylabel("2nd principal component", fontsize=13)
plt.xlim(-7.1, 7.1)
plt.ylim(-3.1, 3.9)
plt.xticks(size=12)
plt.yticks(size=12)
plt.text(-6.2, 3.3, '(b)', fontsize=14)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
# plt.subplot(1,3,2) 
# # h2 = plt.scatter(x_vae[:, 0], x_vae[:, 1], c=percolation, cmap=plt.cm.Blues, marker="o", label='vae')
# plt.title("vae")
# plt.xlabel("1st principal component")
# plt.ylabel("2nd principal component")
# plt.grid(True, linestyle='--')

# plt.subplot(1,3,3) 
# # h3 = plt.scatter(x_cvae[:, 0], x_cvae[:, 1], c=percolation, cmap=plt.cm.Blues, marker="o", label='cvae')
# plt.title("vcae")
# plt.xlabel("1st principal component")
# plt.ylabel("2nd principal component")
# plt.grid(True, linestyle='--')

# l = 0.92
# b = 0.12
# w = 0.02
# h = 1 - 2*b 
# cbar_ax = fig.add_axes([l,b,w,h]) 
# cbar = plt.colorbar(h1, cax=cbar_ax)
# cbar.ax.yaxis.set_major_locator(MultipleLocator(0.1))
# cbar.ax.yaxis.set_minor_locator(MultipleLocator(0.01))
# cbar.ax.tick_params(labelsize=12)
# # cbar.ax.set_title('Permeability', fontsize=14)
# # cbar.set_label('Permeability', fontdict=14)
plt.colorbar()
# l = 0.91
# b = 0.12
# w = 0.02
# h = 1 - 2*b 
# cbar = fig.add_axes([l,b,w,h]) 
# cbar = plt.colorbar(h1, cax=cbar)
# cbar.ax.yaxis.set_major_locator(MultipleLocator(0.1))
# cbar.ax.yaxis.set_minor_locator(MultipleLocator(0.01))
# cbar.ax.tick_params(labelsize=12)
# cbar.ax.set_title('Percolation', fontsize=14)
# plt.subplots_adjust(wspace=1, hspace=1)


plt.tight_layout()

plt.savefig(file_location + r"\figure\eigenvalues.pdf")
plt.savefig(file_location + r"\figure\eigenvalues.eps")
plt.show()