import os
os.environ["TF_CPP_spill_LOG_LEVEL"]="3"
os.environ["CUDA_VISIBLE_DEVICES"]="0"  # gpu if "0"; cpu if "-1"
import numpy as np
import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
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

p = np.linspace(0.41, 0.80, 40) # 0.593
file_location = os.getcwd()
x = read_data(file_location=file_location + r"\VAE\data", 
    name="x_1000").reshape(-1, image_height, image_height, 1)
x_vae = read_data(file_location=file_location + r"\VAE\data", 
    name="vae-x_decoded").reshape(-1, image_height, image_height, 1)
x_cvae = read_data(file_location=file_location + r"\VAE\data", 
    name="cvae-x_decoded").reshape(-1, image_height, image_height, 1)

Pi = read_data(file_location=file_location + r"\VAE\data", 
    name="Pi_1000")

P = read_data(file_location=file_location + r"\VAE\data", 
    name="P_1000")

model_pi = tf.keras.models.load_model(file_location +
    r"\VAE\model_cnn_pi" +
    r"\20210111-120537" +
    r"\00662-0.000818-0.018294-0.028606-0.001806-0.021274-0.042496.h5")
model_p = tf.keras.models.load_model(file_location +
    r"\VAE\model_cnn_p" +
    r"\20201226-203238" +
    r"\00833-0.000315-0.012202-0.017740-0.000772-0.016134-0.027789.h5")


# fig = plt.figure(figsize=(14,5)) 
# ax = fig.add_subplot(121) 
fig = plt.figure(figsize=(5,14)) 
ax = fig.add_subplot(211) 
y_pred = model_pi.predict(x)
y_pred_average = [y_pred[i::40] for i in np.arange(40)]
y_pred_average = np.array(y_pred_average).reshape(-1, 1000)
y_pred_average = np.mean(y_pred_average, axis=1)
plt.plot(p, Pi, c='grey', marker="*", markersize=12, label="raw", linewidth=2)
plt.plot(p, y_pred_average, c='dodgerblue', label="CNN-1", marker="o", linewidth=2)
ax.text(0.41, 0.93, '(a)', fontsize=14)
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
ax = fig.add_subplot(212) 
y_pred = model_p.predict(x)
y_pred_average = [y_pred[i::40] for i in np.arange(40)]
y_pred_average = np.array(y_pred_average).reshape(-1, 1000)
y_pred_average = np.mean(y_pred_average, axis=1)
# ax.scatter(p, y_pred_average, c='', edgecolors='dodgerblue', marker="o")
plt.plot(p, P, c='grey', marker="*", markersize=12, label="raw", linewidth=2)
plt.plot(p, y_pred_average, c='dodgerblue', label="CNN-2", marker="o", linewidth=2)
ax.text(0.41, 0.93, '(b)', fontsize=14)
plt.xlabel('permeability', fontsize=14)
plt.ylabel('$P(p,L)$', fontsize=14)
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
plt.savefig(r".\figure\cnn-percolation-pi-p.pdf")
plt.savefig(r".\figure\cnn-percolation-pi-p.eps")
plt.show()



model_pi_005 = tf.keras.models.load_model(file_location +
    r"\model_cnn_pi" +
    r"\0.05noise" +
    r"\00634-0.001531-0.029299-0.039130-0.003730-0.042500-0.061076.h5")
model_pi_010 = tf.keras.models.load_model(file_location +
    r"\model_cnn_pi" +
    r"\0.10noise" +
    r"\00370-0.004754-0.052782-0.068948-0.008440-0.069633-0.091871.h5")
model_pi_020 = tf.keras.models.load_model(file_location +
    r"\model_cnn_pi" +
    r"\0.20noise" +
    r"\00121-0.019483-0.107892-0.139580-0.025276-0.121460-0.158985.h5")
model_p_005 = tf.keras.models.load_model(file_location +
    r"\model_cnn_p" +
    r"\0.05noise" +
    r"\00160-0.002538-0.037629-0.050374-0.003461-0.043311-0.058831.h5")
model_p_010 = tf.keras.models.load_model(file_location +
    r"\model_cnn_p" +
    r"\0.10noise" +
    r"\00137-0.007155-0.064488-0.084585-0.009680-0.074368-0.098387.h5")
model_p_020 = tf.keras.models.load_model(file_location +
    r"\model_cnn_p" +
    r"\0.20noise" +
    r"\00061-0.025883-0.124777-0.160881-0.030106-0.134895-0.173512.h5")


fig = plt.figure(figsize=(14,5)) 
ax = fig.add_subplot(121) 
plt.plot(p, Pi, c='grey', marker="o", label="raw", linewidth=2)
y_pred = model_pi_005.predict(x)
y_pred_average = [y_pred[i::40] for i in np.arange(40)]
y_pred_average = np.array(y_pred_average).reshape(-1, 1000)
y_pred_average = np.mean(y_pred_average, axis=1)
plt.plot(p, y_pred_average, c='indianred', marker="*", label="5%", linewidth=2)
y_pred = model_pi_010.predict(x)
y_pred_average = [y_pred[i::40] for i in np.arange(40)]
y_pred_average = np.array(y_pred_average).reshape(-1, 1000)
y_pred_average = np.mean(y_pred_average, axis=1)
plt.plot(p, y_pred_average, c='dodgerblue', marker="^", label="10%", linewidth=2)
y_pred = model_pi_020.predict(x)
y_pred_average = [y_pred[i::40] for i in np.arange(40)]
y_pred_average = np.array(y_pred_average).reshape(-1, 1000)
y_pred_average = np.mean(y_pred_average, axis=1)
plt.plot(p, y_pred_average, c='chocolate', marker="+", 
    label="20%", linewidth=2)
ax.text(0.41, 0.93, '(a)', fontsize=14)
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

ax = fig.add_subplot(122) 
plt.plot(p, P, c='grey', marker="o", label="raw", linewidth=2)
y_pred = model_p_005.predict(x)
y_pred_average = [y_pred[i::40] for i in np.arange(40)]
y_pred_average = np.array(y_pred_average).reshape(-1, 1000)
y_pred_average = np.mean(y_pred_average, axis=1)
plt.plot(p, y_pred_average, c='indianred', marker="*", label="5%", linewidth=2)
y_pred = model_p_010.predict(x)
y_pred_average = [y_pred[i::40] for i in np.arange(40)]
y_pred_average = np.array(y_pred_average).reshape(-1, 1000)
y_pred_average = np.mean(y_pred_average, axis=1)
plt.plot(p, y_pred_average, c='dodgerblue', marker="^", label="10%", linewidth=2)
y_pred = model_p_020.predict(x)
y_pred_average = [y_pred[i::40] for i in np.arange(40)]
y_pred_average = np.array(y_pred_average).reshape(-1, 1000)
y_pred_average = np.mean(y_pred_average, axis=1)
plt.plot(p, y_pred_average, c='chocolate', marker="+", label="20%", linewidth=2)
ax.text(0.41, 0.93, '(b)', fontsize=14)
plt.xlabel('permeability', fontsize=14)
plt.ylabel('$P(p,L)$', fontsize=14)
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
plt.savefig(r".\figure\cnn-percolation-pi-p-noise.pdf")
plt.savefig(r".\figure\cnn-percolation-pi-p-noise.eps")
plt.show()




fig = plt.figure() 
ax = fig.add_subplot(111) 
y_pred = model.predict(x)
y_pred_average = [y_pred[i::40] for i in np.arange(40)]
y_pred_average = np.array(y_pred_average).reshape(-1, 1000)
y_pred_average = np.mean(y_pred_average, axis=1)
ax.scatter(p, y_pred_average, c='', edgecolors='dodgerblue', marker="o")
plt.plot(p, y_pred_average, c='dodgerblue', label="CNN")
y_pred = model.predict(x_vae)
y_pred_average = [y_pred[i::40] for i in np.arange(40)]
y_pred_average = np.array(y_pred_average).reshape(-1, 1000)
y_pred_average = np.mean(y_pred_average, axis=1)
ax.scatter(p, y_pred_average, c='', edgecolors='indianred', marker="*")
plt.plot(p, y_pred_average, c='indianred', label="VAE")
y_pred = model.predict(x_cvae)
y_pred_average = [y_pred[i::40] for i in np.arange(40)]
y_pred_average = np.array(y_pred_average).reshape(-1, 1000)
y_pred_average = np.mean(y_pred_average, axis=1)
ax.scatter(p, y_pred_average, c='', edgecolors='green', marker="+")
plt.plot(p, y_pred_average, c='green', label="cVAE")
plt.xlabel('percolation')
plt.ylabel('P$^{predict}$(p,L)')
plt.xlim(0.39, 0.82)
plt.ylim(-0.1, 1.1)
plt.grid(True, linestyle='--')
plt.legend()
plt.savefig(r".\figure\cnn-percolation-p.pdf")
plt.savefig(r".\figure\cnn-percolation-p.eps")
plt.show()
