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
x = read_data(file_location=file_location + r"\data", 
    name="x_1000")

model = tf.keras.models.load_model(file_location +
    r"\model_cnn_pi" +
    r"\nonoise" +
    r"\00944-0.000591-0.015427-0.024315-0.001752-0.020634-0.041857.h5")

x_1 = np.empty(shape=(len(x), image_height*image_height))
x_2 = np.empty(shape=(len(x), image_height*image_height))
x_3 = np.empty(shape=(len(x), image_height*image_height))
x_4 = np.empty(shape=(len(x), image_height*image_height))
x_5 = np.empty(shape=(len(x), image_height*image_height))
x_6 = np.empty(shape=(len(x), image_height*image_height))
x_7 = np.empty(shape=(len(x), image_height*image_height))
x_8 = np.empty(shape=(len(x), image_height*image_height))
x_9 = np.empty(shape=(len(x), image_height*image_height))
x_10 = np.empty(shape=(len(x), image_height*image_height))

for i in np.arange(len(x)):
    length = int(image_height*image_height*0.01)
    l = list(np.where(x[i, :]==0)[0])
    if len(l) < length:
        length = len(l)
    s = random.sample(l, length)
    for j in s:
        x_1[i, j] = 1
    print(np.where(x[i, :]==0))

for i in np.arange(len(x)):
    length = int(image_height*image_height*0.02)
    l = list(np.where(x[i, :]==0)[0])
    if len(l) < length:
        length = len(l)
    s = random.sample(l, length)
    for j in s:
        x_2[i, j] = 1

for i in np.arange(len(x)):
    length = int(image_height*image_height*0.03)
    l = list(np.where(x[i, :]==0)[0])
    if length > len(l):
        length = len(l)
    s = random.sample(l, length)
    for j in s:
        x_3[i, j] = 1

for i in np.arange(len(x)):
    length = int(image_height*image_height*0.04)
    l = list(np.where(x[i, :]==0)[0])
    if len(l) < length:
        length = len(l)
    s = random.sample(l, length)
    for j in s:
        x_4[i, j] = 1

for i in np.arange(len(x)):
    length = int(image_height*image_height*0.05)
    l = list(np.where(x[i, :]==0)[0])
    if len(l) < length:
        length = len(l)
    s = random.sample(l, length)
    for j in s:
        x_5[i, j] = 1

for i in np.arange(len(x)):
    length = int(image_height*image_height*0.06)
    l = list(np.where(x[i, :]==0)[0])
    if len(l) < length:
        length = len(l)
    s = random.sample(l, length)
    for j in s:
        x_6[i, j] = 1

for i in np.arange(len(x)):
    length = int(image_height*image_height*0.07)
    l = list(np.where(x[i, :]==0)[0])
    if len(l) < length:
        length = len(l)
    s = random.sample(l, length)
    for j in s:
        x_7[i, j] = 1

for i in np.arange(len(x)):
    length = int(image_height*image_height*0.08)
    l = list(np.where(x[i, :]==0)[0])
    if len(l) < length:
        length = len(l)
    s = random.sample(l, length)
    for j in s:
        x_8[i, j] = 1

for i in np.arange(len(x)):
    length = int(image_height*image_height*0.09)
    l = list(np.where(x[i, :]==0)[0])
    if len(l) < length:
        length = len(l)
    s = random.sample(l, length)
    for j in s:
        x_9[i, j] = 1

for i in np.arange(len(x)):
    length = int(image_height*image_height*0.10)
    l = list(np.where(x[i, :]==0)[0])
    if len(l) < length:
        length = len(l)
    s = random.sample(l, length)
    for j in s:
        x_10[i, j] = 1

x = x.reshape(-1, image_height, image_height, 1)
x_05 = x_1.reshape(-1, image_height, image_height, 1)
x_1 = x_1.reshape(-1, image_height, image_height, 1)
x_2 = x_2.reshape(-1, image_height, image_height, 1)
x_3 = x_3.reshape(-1, image_height, image_height, 1)
x_4 = x_4.reshape(-1, image_height, image_height, 1)
x_5 = x_5.reshape(-1, image_height, image_height, 1)
x_6 = x_6.reshape(-1, image_height, image_height, 1)
x_7 = x_7.reshape(-1, image_height, image_height, 1)
x_8 = x_8.reshape(-1, image_height, image_height, 1)
x_9 = x_9.reshape(-1, image_height, image_height, 1)
x_10 = x_10.reshape(-1, image_height, image_height, 1)


y_pred = model.predict(x)
y_pred_average = [y_pred[i::40] for i in np.arange(40)]
y_pred_average = np.array(y_pred_average).reshape(-1, 1000)
y_pred_average = np.mean(y_pred_average, axis=1)
# plt.scatter(p, y_pred_average, c='', edgecolors='dodgerblue', marker="o")
plt.plot(p, y_pred_average, c='r', label='0')

y_pred = model.predict(x_1)
y_pred_average = [y_pred[i::40] for i in np.arange(40)]
y_pred_average = np.array(y_pred_average).reshape(-1, 1000)
y_pred_average = np.mean(y_pred_average, axis=1)
plt.plot(p, y_pred_average, c='c', label='0.010')

y_pred = model.predict(x_2)
y_pred_average = [y_pred[i::40] for i in np.arange(40)]
y_pred_average = np.array(y_pred_average).reshape(-1, 1000)
y_pred_average = np.mean(y_pred_average, axis=1)
plt.plot(p, y_pred_average, c='g', label='0.020')

y_pred = model.predict(x_3)
y_pred_average = [y_pred[i::40] for i in np.arange(40)]
y_pred_average = np.array(y_pred_average).reshape(-1, 1000)
y_pred_average = np.mean(y_pred_average, axis=1)
plt.plot(p, y_pred_average, c='b', label='0.030')

y_pred = model.predict(x_4)
y_pred_average = [y_pred[i::40] for i in np.arange(40)]
y_pred_average = np.array(y_pred_average).reshape(-1, 1000)
y_pred_average = np.mean(y_pred_average, axis=1)
plt.plot(p, y_pred_average, c='IndianRed', label='0.040')

y_pred = model.predict(x_5)
y_pred_average = [y_pred[i::40] for i in np.arange(40)]
y_pred_average = np.array(y_pred_average).reshape(-1, 1000)
y_pred_average = np.mean(y_pred_average, axis=1)
plt.plot(p, y_pred_average, c='DarkCyan', label='0.050')

y_pred = model.predict(x_6)
y_pred_average = [y_pred[i::40] for i in np.arange(40)]
y_pred_average = np.array(y_pred_average).reshape(-1, 1000)
y_pred_average = np.mean(y_pred_average, axis=1)
plt.plot(p, y_pred_average, c='darkviolet', label='0.060')

y_pred = model.predict(x_7)
y_pred_average = [y_pred[i::40] for i in np.arange(40)]
y_pred_average = np.array(y_pred_average).reshape(-1, 1000)
y_pred_average = np.mean(y_pred_average, axis=1)
plt.plot(p, y_pred_average, c='dodgerblue', label='0.070')

y_pred = model.predict(x_8)
y_pred_average = [y_pred[i::40] for i in np.arange(40)]
y_pred_average = np.array(y_pred_average).reshape(-1, 1000)
y_pred_average = np.mean(y_pred_average, axis=1)
plt.plot(p, y_pred_average, c='midnightblue', label='0.080')

y_pred = model.predict(x_9)
y_pred_average = [y_pred[i::40] for i in np.arange(40)]
y_pred_average = np.array(y_pred_average).reshape(-1, 1000)
y_pred_average = np.mean(y_pred_average, axis=1)
plt.plot(p, y_pred_average, c='black', label='0.090')

y_pred = model.predict(x_10)
y_pred_average = [y_pred[i::40] for i in np.arange(40)]
y_pred_average = np.array(y_pred_average).reshape(-1, 1000)
y_pred_average = np.mean(y_pred_average, axis=1)
# plt.scatter(p, y_pred_average, c='', edgecolors='dodgerblue', marker="o")
plt.plot(p, y_pred_average, c='m', label='0.10')


plt.xlabel('$p$')
plt.ylabel('$\Pi$$^{predict}$(p,L)')
plt.xlim(0.39, 0.82)
plt.ylim(-0.1, 1.1)
plt.axhline(y=0.5, color="black", linestyle=":")
plt.axvline(x=0.593, color="black", linestyle="--")
plt.legend(loc='upper left')
plt.savefig(r".\figure\cnn-percolation-pi.pdf")
plt.savefig(r".\figure\cnn-percolation-pi.eps")
plt.show()
