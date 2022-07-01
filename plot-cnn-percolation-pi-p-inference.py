
import os
os.environ["TF_CPP_spill_LOG_LEVEL"]="3"
os.environ["CUDA_VISIBLE_DEVICES"]="0"  # gpu if "0"; cpu if "-1"
import numpy as np
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4608),
    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4608)])
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

sweeps = 10**3
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

per = np.linspace(0.41, 0.80, 40).reshape(-1,) # 0.593
file_location = os.getcwd()
x = read_data(file_location=file_location + r"\data", 
    name="x_1000")
pi1 = pi = read_data(file_location=file_location+r"\data", name="Pi_1000")
p1 = p = read_data(file_location=file_location+r"\data", name="P_1000")

pi_m = np.array(np.tile(pi, sweeps))
a = np.argwhere((pi_m<=0.1) & (pi_m>=0.9))
b = np.argwhere((pi_m>0.1) & (pi_m<0.9))

# pi = pi[a].reshape(-1, 1)
# p = p[a].reshape(-1, 1)
pi = np.append(pi[np.argwhere(pi<=0.1)], pi[np.argwhere(pi>=0.9)],axis=0)
p = np.append(p[np.argwhere(pi<=0.1)], p[np.argwhere(pi>=0.9)],axis=0)
# per = np.concatenate((per[np.argwhere(pi<=0.1)], per[np.argwhere(pi>=0.9)]),axis=0)
per = list(np.arange(0.41, 0.54, 0.01)) + list(np.arange(0.65, 0.80, 0.01))
per_test = list(np.arange(0.55, 0.65, 0.01))

# x1 = x[a].reshape(-1, image_height, image_height, 1)
x_test = x[b].reshape(-1, image_height, image_height, 1)

model_pi = tf.keras.models.load_model(file_location +
    r"\model_cnn_pi" +
    r"\inference" +
    r"\00956-0.000071-0.005427-0.008450-0.000504-0.009347-0.022452.h5")
model_p = tf.keras.models.load_model(file_location +
    r"\model_cnn_p" +
    r"\inference" +
    r"\00814-0.000030-0.003213-0.005510-0.000164-0.006277-0.012796.h5")


fig = plt.figure(figsize=(6,5)) 
ax = fig.add_subplot(111) 
plt.plot(np.linspace(0.41, 0.80, 40), pi1, c='grey', 
    marker="*", label="raw", linewidth=2)
plt.scatter(per, pi, c='dodgerblue', marker="s", 
    label="truncated dataset", linewidth=2)
y_pred = model_pi.predict(x_test)
y_pred_average = [y_pred[i::10] for i in np.arange(10)]
y_pred_average = np.array(y_pred_average).reshape(-1, 1000)
y_pred_average = np.mean(y_pred_average, axis=1)
plt.scatter(per_test, y_pred_average, c='indianred', 
    label="extrapolated dataset", marker="o", linewidth=2)
# ax.text(0.41, 0.93, '(a)', fontsize=14)
plt.xlabel('permeability', fontsize=14)
plt.ylabel('$\Pi(p,L)$', fontsize=14)
plt.xticks(size=12)
plt.yticks(size=12)
plt.xlim(0.39, 0.81)
plt.ylim(-0.09, 1.09)
plt.grid(True, linestyle='--', linewidth=1.5)
plt.legend(loc="lower right")
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)

# ax = fig.add_subplot(122) 
# plt.plot(np.linspace(0.41, 0.80, 40), p1, c='grey', marker="*", label="raw", linewidth=2)
# # plt.scatter(per, p, c='dodgerblue', marker="s", label=" truncated dataset", linewidth=2)
# y_pred = model_p.predict(x_test)
# y_pred_average = [y_pred[i::10] for i in np.arange(10)]
# y_pred_average = np.array(y_pred_average).reshape(-1, 1000)
# y_pred_average = np.mean(y_pred_average, axis=1)
# plt.scatter(per_test, y_pred_average, c='indianred', 
#     label=" removed dataset", marker="o", linewidth=2)
# ax.text(0.41, 0.93, '(a)', fontsize=14)
# plt.xlabel('permeability', fontsize=14)
# plt.ylabel('$P(p,L)$', fontsize=14)
# plt.xticks(size=12)
# plt.yticks(size=12)
# plt.xlim(0.39, 0.81)
# plt.ylim(-0.09, 1.09)
# plt.grid(True, linestyle='--', linewidth=1.5)
# plt.legend(loc="lower right")
# ax = plt.gca()
# ax.spines['bottom'].set_linewidth(1.5)
# ax.spines['left'].set_linewidth(1.5)
# ax.spines['top'].set_linewidth(1.5)
# ax.spines['right'].set_linewidth(1.5)

plt.savefig(r".\figure\cnn-percolation-pi-inference.pdf")
plt.savefig(r".\figure\cnn-percolation-pi-inference.eps")
plt.show()
