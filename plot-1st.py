

import os
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import h5py
import time
from scipy.ndimage import measurements
import numpy as np
import matplotlib.pyplot as plt
import h5py
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn import preprocessing 

from utils import read_data

start_time = time.time()

p = linspace(0.41, 0.8, 40)  # 0.593 
L = 28
sweeps = 10**3

file_location = os.getcwd() + r"\VAE"
x = read_data(file_location=file_location+r"\data", name="x_1000")
print(x.shape) 
p = read_data(file_location=file_location+r"\data", name="per_1000")
print(p.shape) 
Pi = read_data(file_location=file_location+r"\data", name="Pi_1000")
print("Pi.shape = ", Pi.shape)
P = read_data(file_location=file_location+r"\data", name="P_1000")
print("P.shape = ", P.shape)

lp = len(p) 
lp = 40

n_components = 2
pca = PCA(n_components = n_components)
name = "PCA"
x = pca.fit_transform(x)
# x_vae = pca.fit_transform(x_vae)
# x_cvae = pca.fit_transform(x_cvae)


# Pi_origin = np.tile(Pi_origin, sweeps)
# P_origin = np.tile(P_origin, sweeps)

# x_40 = np.zeros(len(p))
# for i in np.arange(0, len(x_40), 1):
#     for j in np.arange(i, len(x), 40):
#         x_40[i] += x[j, 0]
# x_40 = x_40/10000
# print("x_40.shape = ", x_40.shape)

# x = read_data(file_location=file_location+r".\latent=1", name="x_vae_encoded")
x_40 = np.zeros(len(p))
for i in np.arange(0, len(x_40), 1):
    for j in np.arange(i, len(x), 40):
        x_40[i] += x[j, 0]
x_40 = x_40/1000
print("x_40.shape = ", x_40.shape)

# x_40 = (x_40-min(x_40))/(max(x_40)-min(x_40))
# fig = plt.figure(figsize=(6, 10))
# plt.subplot(3,1,1) 
fig = plt.figure(figsize=(10, 3))
plt.subplot(1,3,1) 
# h1 = plt.scatter(p, x_40, c=p, marker="o")
h1 = plt.plot(x_40, p, c='dodgerblue', marker="o", linewidth=2)
plt.xlabel('$<x^{1st}>$', fontsize=14)
plt.ylabel('$p$', fontsize=14)
plt.xlim(-6.09, 6.09)
# plt.ylim(3.9, 8.1)
plt.xticks(size=12)
plt.yticks(size=12)
plt.text(-5.6, 0.74, '(a)', fontsize=14)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.xaxis.set_major_locator(plt.MultipleLocator(2))
ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
# plt.subplot(3,1,2) 
plt.subplot(1,3,2) 
h1 = plt.plot(x_40, Pi, c='dodgerblue', marker="o", linewidth=2)
plt.xlabel('$<x^{1st}>$', fontsize=14)
plt.ylabel('$\Pi(p,L)$', fontsize=14)
plt.xlim(-6.09, 6.09)
plt.ylim(-0.05, 1.05)
plt.xticks(size=12)
plt.yticks(size=12)
plt.text(-5.6, 0.87, '(b)', fontsize=14)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.xaxis.set_major_locator(plt.MultipleLocator(2))
ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
# plt.subplot(3,1,3) 
plt.subplot(1,3,3) 
h1 = plt.plot(x_40, P, c='dodgerblue', marker="o", linewidth=2)
plt.xlabel('$<x^{1st}>$', fontsize=14)
plt.ylabel('$P(p,L)$', fontsize=14)
plt.xlim(-6.09, 6.09)
plt.ylim(-0.05, 1.05)
plt.xticks(size=12)
plt.yticks(size=12)
plt.text(-5.6, 0.87, '(c)', fontsize=14)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.xaxis.set_major_locator(plt.MultipleLocator(2))
ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
# # colorbar 左 下 宽 高 
# l = 0.95
# b = 0.12
# w = 0.015
# h = 1 - 2*b 
# cbar_ax = fig.add_axes([l,b,w,h]) 
# cbar = plt.colorbar(h1, cax=cbar_ax)
# cbar.ax.yaxis.set_major_locator(MultipleLocator(0.1))
# cbar.ax.yaxis.set_minor_locator(MultipleLocator(0.01))
# cbar.ax.tick_params(labelsize=9.5)
# cbar.ax.set_title('Percolation', fontsize=10.5)
# plt.subplots_adjust(wspace=1, hspace=1)
plt.subplots_adjust(left=None,bottom=None,
    right=None,top=None,wspace=0.5,hspace=None)
plt.tight_layout()
plt.savefig(file_location + r"\figure\1st.pdf")
plt.savefig(file_location + r"\figure\1st.eps")
plt.show()

end_time = time.time()
print("times = ", (end_time-start_time)/60)
