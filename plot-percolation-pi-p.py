
import os
os.environ["TF_CPP_spill_LOG_LEVEL"]="3"
os.environ["CUDA_VISIBLE_DEVICES"]="0"  # gpu if "0"; cpu if "-1"
import numpy as np
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_virtual_device_configuration(gpus[0],
#     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4608),
#     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4608)])
tf.keras.backend.clear_session()
import time
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

from utils import read_data

start_time = time.time()
seed = 12345
random.seed(seed)
np.random.seed(seed=seed)

image_height = 28
image_size = 28*28
epochs = 10**3
learning_rate = 1e-4
batch_size = 256
l2_rate = 1e-4

num_classes = 1
l2_rate = 0
filter1 = 32
filter2 = 64
fc1 = 128
dropout_rate = 0.5

per = np.linspace(0.41, 0.80, 40) # 0.593
file_location = os.getcwd()
print("*"*50, file_location)

x = read_data(file_location=file_location + r"\VAE\data", 
    name="x_1000").reshape(-1, image_height, image_height, 1)
x_vae = read_data(file_location=file_location + r"\VAE\data", 
    name="vae-x_decoded").reshape(-1, image_height, image_height, 1)
x_cvae = read_data(file_location=file_location + r"\VAE\data", 
    name="cvae-x_decoded").reshape(-1, image_height, image_height, 1)

percolation = read_data(file_location=file_location + r"\VAE\data", 
    name="percolation_1000")
pi = read_data(file_location=file_location + r"\VAE\data", 
    name="Pi_1000")
p = read_data(file_location=file_location + r"\VAE\data", 
    name="P_1000")

# fig = plt.figure(figsize=(6,8)) 
# ax = fig.add_subplot(211) 
fig = plt.figure(figsize=(8,4)) 
ax = fig.add_subplot(121) 
ax.plot(per, pi, label='MC', c='dodgerblue', marker="o", linewidth=2)
ax.text(0.40, 0.98, '(a)', fontsize=14)
plt.xlabel('$p$', fontsize=14)
plt.ylabel('$\Pi(p,L)$', fontsize=14)
plt.xticks(size=12)
plt.yticks(size=12)
plt.xlim(0.39, 0.81)
plt.ylim(-0.09, 1.09)
plt.legend(loc='lower right', fontsize=12)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
from pylab import *
tick_params(which='major', width=2)
ax.xaxis.set_major_locator(plt.MultipleLocator(0.05))
ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
# ax.xaxis.set_minor_locator(plt.MultipleLocator(0.01))
# ax = fig.add_subplot(212) 
ax = fig.add_subplot(122) 
ax.plot(per, p, label='MC', c='dodgerblue', marker="o", linewidth=2)
ax.text(0.40, 0.98, '(b)', fontsize=14)
plt.xlabel('$p$', fontsize=14)
plt.ylabel('$P(p,L)$', fontsize=14)
plt.xticks(size=12)
plt.yticks(size=12)
plt.xlim(0.39, 0.81)
plt.ylim(-0.09, 1.09)
plt.legend(loc='lower right', fontsize=12)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
from pylab import *
tick_params(which='major', width=2)
ax.xaxis.set_major_locator(plt.MultipleLocator(0.05))
ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
# ax.xaxis.set_minor_locator(plt.MultipleLocator(0.01))
plt.subplots_adjust(left=None,bottom=None,
    right=None,top=None,wspace=None,hspace=None)
plt.tight_layout()
plt.savefig(file_location + r"\VAE\figure\percolation-pi-p.pdf")
plt.savefig(file_location + r"\VAE\figure\percolation-pi-p.eps")
plt.show()


# model = tf.keras.models.load_model(file_location +
#     r"\model_cnn_percolation" +
#     r"\20201226-200824" +
#     r"\00968-0.000001-0.000725-0.000958-0.000189-0.010358-0.013753.h5")

# y_pred = model.predict(x)
# y_pred_vae = model.predict(x_vae)
# y_pred_cvae = model.predict(x_cvae)
# # y_pred_average = [y_pred[i::40] for i in np.arange(40)]
# # y_pred_average = np.array(y_pred_average).reshape(-1, 1000)
# # y_pred_average = np.mean(y_pred_average, axis=1)

# fig = plt.figure(figsize=(14,9)) 
# ax = fig.add_subplot(121) 
# ax.scatter(percolation, y_pred, label="CNN", c='dodgerblue', marker="^")
# ax.scatter(percolation, y_pred_vae, label="VAE", c='indianred', marker="*")
# ax.scatter(percolation, y_pred_cvae, label="cVAE", c='darkviolet', marker="+")
# plt.legend() 
# # plt.xlabel('Monte Carlo')
# plt.xlabel('origin')
# plt.ylabel('predict')
# plt.title("occupied probability")
# plt.xlim(0.39, 0.82)
# # plt.ylim(0.39, 0.85)
# plt.ylim(0.32, 0.85)
# plt.grid(True, linestyle='--')
# plt.tight_layout()
# plt.savefig(r".\figure\cnn-percolation12.pdf")
# plt.savefig(r".\figure\cnn-percolation12.eps")
# plt.show()
