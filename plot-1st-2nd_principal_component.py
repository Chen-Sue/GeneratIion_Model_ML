
import numpy as np
import matplotlib.pyplot as plt
import h5py
from sklearn.decomposition import PCA, KernelPCA, IncrementalPCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn import preprocessing 
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, \
                                AutoMinorLocator
import os

from utils import read_data


sweeps= 1000

file_location = os.getcwd()
x = read_data(file_location=file_location + r"\data" , name="x_1000")
percolation = read_data(file_location=file_location + r"\data", 
    name="percolation_1000").reshape(-1, )

n_components = 2

pca = PCA(n_components=n_components)
name = "PCA"
# kernels = ['linear','poly','rbf','sigmoid']
# pca = KernelPCA(n_components=n_components, kernel=kernels[0])
# pca = KernelPCA(n_components=n_components, kernel=kernels[1])
# pca = KernelPCA(n_components=n_components, kernel=kernels[2])
# pca = KernelPCA(n_components=n_components, kernel=kernels[3])
# pca = IncrementalPCA(n_components=n_components)
# pca = TSNE(n_components=n_components)  # two-component TSNE
# pca = MDS(n_components=2)  # two-component MDS
# pca = DBSCAN()

x = pca.fit_transform(x)
# x_vae = pca.fit_transform(x_vae)
# x_cvae = pca.fit_transform(x_cvae.reshape(-1, 28*28))

fig = plt.figure(figsize=(7, 5.5))

plt.subplot(1,1,1) 
plt.grid(True, linestyle='--', linewidth=1.5)
h1 = plt.scatter(x[:, 0], x[:, 1], c=percolation, marker="o", linewidth=2)
# plt.title(r"1st principal component - 2nd principal component")
plt.xlabel("1st principal component", fontsize=14)
plt.ylabel("2nd principal component", fontsize=14)
plt.xlim(-7.1, 7.1)
plt.ylim(-3.1, 3.8)
plt.xticks(size=12)
plt.yticks(size=12)
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

l = 0.91
b = 0.12
w = 0.02
h = 1 - 2*b 
cbar = fig.add_axes([l,b,w,h]) 
cbar = plt.colorbar(h1, cax=cbar)
# cbar.ax.yaxis.set_major_locator(MultipleLocator(0.1))
# cbar.ax.yaxis.set_minor_locator(MultipleLocator(0.01))
# cbar.ax.tick_params(labelsize=12)
# cbar.ax.set_title('Percolation', fontsize=14)
# plt.subplots_adjust(wspace=1, hspace=1)

plt.savefig(r".\figure\1st principal component - " +
            r"2nd principalcomponent.pdf")
plt.savefig(r".\figure\1st principal component - " +
            r"2nd principalcomponent.eps")
plt.show()
