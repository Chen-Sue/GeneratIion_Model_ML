



import os
from tensorflow import keras
from tensorflow.keras import models
import matplotlib.pyplot as plt
import tensorflow as tf
import os
os.environ["TF_CPP_spill_LOG_LEVEL"]="3"
os.environ["CUDA_VISIBLE_DEVICES"]="0"  # gpu if "0"; cpu if "-1"
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4608),
    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4608)])
tf.keras.backend.clear_session()
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, MaxAbsScaler
from tensorflow.keras import layers, optimizers, metrics, Sequential, initializers, losses
from tensorflow.keras import Model, layers, datasets, Sequential, optimizers
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPool2D, BatchNormalization, \
                                    Dropout, Reshape, Conv2DTranspose, UpSampling2D, \
                                    Flatten, Dense, Input
import h5py
from pylab import *
import time
import datetime
import random

from sklearn.model_selection import train_test_split
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA, IncrementalPCA

from utils import binary, save_data, read_data, reparameter,\
    shuffle_data

start_time = time.time()
seed = 12345
random.seed(seed)
np.random.seed(seed=seed)

p = np.linspace(0.41, 0.80, 40) # 0.593

file_location = os.getcwd()
image_height = 28
image_size = 28*28
epochs = 10**3
learning_rate = 1e-4
batch_size = 256
l2_rate = 1e-4

p = np.linspace(0.41, 0.80, 40) # 0.593
x = read_data(file_location=file_location + r"\data", 
    name="x_1000").reshape(-1, image_height, image_height, 1)

model = keras.models.load_model(
    r"D:\hoffee\generate_model\VAE\model_cnn_critical" +
    r"\0.55" +
    r"\00099-0.194387-0.960125-0.183689-0.961344.h5")
y_pred = model.predict(x)
y_pred_average = [y_pred[i::40] for i in np.arange(40)]
y_pred_average = np.array(y_pred_average).reshape(-1, 1000)
y_pred_average = np.mean(y_pred_average, axis=1)
plt.plot(p, y_pred_average, c='chocolate', marker="d", label='0.55', linewidth=2)

model = keras.models.load_model(
    r"D:\hoffee\generate_model\VAE\model_cnn_critical" +
    r"\0.56" +
    r"\00093-0.201413-0.954625-0.183066-0.961469.h5")
y_pred = model.predict(x)
y_pred_average = [y_pred[i::40] for i in np.arange(40)]
y_pred_average = np.array(y_pred_average).reshape(-1, 1000)
y_pred_average = np.mean(y_pred_average, axis=1)
plt.plot(p, y_pred_average, c='m', marker="h", label='0.56', linewidth=2)


model = keras.models.load_model(
    r"D:\hoffee\generate_model\VAE\model_cnn_critical" +
    r"\0.57" +
    r"\00096-0.176472-0.965375-0.177085-0.963875.h5")
y_pred = model.predict(x)
y_pred_average = [y_pred[i::40] for i in np.arange(40)]
y_pred_average = np.array(y_pred_average).reshape(-1, 1000)
y_pred_average = np.mean(y_pred_average, axis=1)
plt.plot(p, y_pred_average, c='c', marker="^", label='0.57', linewidth=2)

model = keras.models.load_model(
    r"D:\hoffee\generate_model\VAE\model_cnn_critical" +
    r"\0.58" +
    r"\00099-0.195470-0.965500-0.187231-0.963531.h5")
y_pred = model.predict(x)
y_pred_average = [y_pred[i::40] for i in np.arange(40)]
y_pred_average = np.array(y_pred_average).reshape(-1, 1000)
y_pred_average = np.mean(y_pred_average, axis=1)
plt.plot(p, y_pred_average, c='g', marker="v", label='0.58', linewidth=2)

model = keras.models.load_model(
    r"D:\hoffee\generate_model\VAE\model_cnn_critical" +
    r"\0.59" +
    r"\00095-0.191527-0.960375-0.179923-0.963188.h5")
y_pred = model.predict(x)
y_pred_average = [y_pred[i::40] for i in np.arange(40)]
y_pred_average = np.array(y_pred_average).reshape(-1, 1000)
y_pred_average = np.mean(y_pred_average, axis=1)
plt.plot(p, y_pred_average, c='b', marker="+", label='0.59', linewidth=2)

model = keras.models.load_model(
    r"D:\hoffee\generate_model\VAE\model_cnn_critical" +
    r"\0.60" +
    r"\00097-0.163298-0.972625-0.173939-0.963031.h5")
y_pred = model.predict(x)
y_pred_average = [y_pred[i::40] for i in np.arange(40)]
y_pred_average = np.array(y_pred_average).reshape(-1, 1000)
y_pred_average = np.mean(y_pred_average, axis=1)
plt.plot(p, y_pred_average, c='indianred', marker="o", label='0.60', linewidth=2)

model = keras.models.load_model(
    r"D:\hoffee\generate_model\VAE\model_cnn_critical" +
    r"\0.61" +
    r"\00096-0.208728-0.954750-0.183661-0.963344.h5")
y_pred = model.predict(x)
y_pred_average = [y_pred[i::40] for i in np.arange(40)]
y_pred_average = np.array(y_pred_average).reshape(-1, 1000)
y_pred_average = np.mean(y_pred_average, axis=1)
plt.plot(p, y_pred_average, c='y', marker="x", label='0.61', linewidth=2)

model = keras.models.load_model(
    r"D:\hoffee\generate_model\VAE\model_cnn_critical" +
    r"\0.62" +
    r"\00100-0.177160-0.969500-0.176295-0.965156.h5")
y_pred = model.predict(x)
y_pred_average = [y_pred[i::40] for i in np.arange(40)]
y_pred_average = np.array(y_pred_average).reshape(-1, 1000)
y_pred_average = np.mean(y_pred_average, axis=1)
plt.plot(p, y_pred_average, c='darkviolet', marker=">", label='0.62', linewidth=2)

model = keras.models.load_model(
    r"D:\hoffee\generate_model\VAE\model_cnn_critical" +
    r"\0.63" +
    r"\00098-0.177702-0.965625-0.178431-0.963656.h5")
y_pred = model.predict(x)
y_pred_average = [y_pred[i::40] for i in np.arange(40)]
y_pred_average = np.array(y_pred_average).reshape(-1, 1000)
y_pred_average = np.mean(y_pred_average, axis=1)
plt.plot(p, y_pred_average, c='dodgerblue', marker="<", label='0.63', linewidth=2)

model = keras.models.load_model(
    r"D:\hoffee\generate_model\VAE\model_cnn_critical" +
    r"\0.64" +
    r"\00088-0.179632-0.961625-0.171374-0.965062.h5")
y_pred = model.predict(x)
y_pred_average = [y_pred[i::40] for i in np.arange(40)]
y_pred_average = np.array(y_pred_average).reshape(-1, 1000)
y_pred_average = np.mean(y_pred_average, axis=1)
plt.plot(p, y_pred_average, c='grey', marker="s", label='0.64', linewidth=2)

model = keras.models.load_model(
    r"D:\hoffee\generate_model\VAE\model_cnn_critical" +
    r"\0.65" +
    r"\00100-0.171331-0.967000-0.170876-0.965375.h5")
y_pred = model.predict(x)
y_pred_average = [y_pred[i::40] for i in np.arange(40)]
y_pred_average = np.array(y_pred_average).reshape(-1, 1000)
y_pred_average = np.mean(y_pred_average, axis=1)
plt.plot(p, y_pred_average, c='midnightblue', marker="p",label='0.65', linewidth=2)

plt.xlabel('permeability', size=14)
plt.ylabel('output layer', size=14)
plt.xlim(0.39, 0.81)
# plt.ylim(-0.1, 1.1)
plt.xticks(size=12)
plt.yticks(size=12)
plt.grid(True, linestyle='--', linewidth=1.5)
plt.axhline(y=0.5, color="black", linestyle="--")
plt.axvline(x=0.593, color="black", linestyle="--")
plt.legend(loc='upper left')
plt.show()
