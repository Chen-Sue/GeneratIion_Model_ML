




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

from utils import callback, binary, save_data, read_data, reparameter,\
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
num_classes = 40

p = np.linspace(0.41, 0.80, 40) # 0.593
x = read_data(file_location=file_location + r"\data", 
    name="x_1000").reshape(-1, image_height, image_height, 1)

model = keras.models.load_model(
    r"D:\hoffee\generate_model\VAE\model_cnn_critical_reg" +
    r"\20201215-225630" +
    r"\00010-0.020572-0.021250-0.013137-0.025937.h5")

probability_model = tf.keras.Sequential([model, layers.Softmax()])
predictions = probability_model.predict(x)
predictions = np.argmax(predictions)
y_shuffle = tf.squeeze(tf.one_hot(predictions, depth=num_classes))


y_pred = model.predict(x)
y_pred_average = [y_pred[i::40] for i in np.arange(40)]
y_pred_average = np.array(y_pred_average).reshape(-1, 1000)
y_pred_average = np.mean(y_pred_average, axis=1)

plt.scatter(p, y_pred_average, alpha=0.8, c='', edgecolors='blue', 
    marker="o", label='output layer')

plt.plot(p, y_pred_average, c='r')
plt.xlabel('percolation')
plt.ylabel('y_pred_output layer')
plt.xlim(0.39, 0.81)
# plt.ylim(-0.1, 1.1)
plt.grid(True, linestyle='--')
plt.axhline(y=0.5, color="black", linestyle=":")
plt.axvline(x=0.593, color="black", linestyle="--")
plt.legend(loc='upper left')
plt.show()
