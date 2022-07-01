
import time
import os
import numpy as np
import tensorflow as tf
import os
os.environ["TF_CPP_spill_LOG_LEVEL"]="3"
os.environ["CUDA_VISIBLE_DEVICES"]="0"  # gpu if "0"; cpu if "-1"
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4608),
    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4608)])
tf.keras.backend.clear_session()
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, MaxAbsScaler
from tensorflow.keras import layers, optimizers, metrics, Sequential, initializers, losses
from tensorflow.keras import Model, layers, datasets, Sequential, optimizers
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPool2D, BatchNormalization, \
                                    Dropout, Reshape, Conv2DTranspose, UpSampling2D, \
                                    Flatten, Dense, Input
import h5py
from pylab import *
import time
import datetime
import random
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from utils import read_data, shuffle_data, callback
start_time = time.time()
seed = 12345
random.seed(seed)
np.random.seed(seed=seed)
file_location = os.getcwd()

L =28
image_height = 28
image_size = 28*28
epochs = 10**2
learning_rate = 1e-3
batch_size = 512
l2_rate = 1e-3

x = read_data(file_location=file_location+r"\data", 
    name="x_1000").reshape(-1, image_height, image_height, 1)
y = read_data(file_location=file_location+r"\data", 
    name="percolation_1000")
model = models.Sequential()
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), 
    activation='relu', 
    kernel_initializer=initializers.he_normal(), 
    kernel_regularizer=keras.regularizers.l2(l2_rate),
    input_shape=(28, 28, 1)
))
# model.add(layers.BatchNormalization())
# model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), 
    activation='relu', 
    kernel_initializer=initializers.he_normal(), 
    kernel_regularizer=keras.regularizers.l2(l2_rate),
))
# model.add(layers.BatchNormalization())
# model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), 
#     activation='relu',))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, 
    activation='relu', 
    kernel_initializer=initializers.he_normal(), 
    kernel_regularizer=keras.regularizers.l2(l2_rate),
))
model.add(layers.Dropout(0.5))
# model.add(layers.BatchNormalization())
# model.add(layers.Activation('relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary() 
# model.build(input_shape=[None, 28, 28, 1])
optimizer = optimizers.Adam(learning_rate=learning_rate)
# model.compile(optimizer=optimizer, 
#     loss="mean_squared_error", 
#     metrics=["mae", tf.keras.metrics.RootMeanSquaredError(name='rmse')])
# model.compile(optimizer='adam',
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     metrics=['accuracy'])
model.compile(optimizer=optimizers.RMSprop(lr=0.001), 
    loss=losses.binary_crossentropy, 
    metrics=[metrics.binary_accuracy])
# for pc in np.arange(0.55, 0.65+0.01, 0.01):
pc = 0.65
y = np.where(y>=pc, 1, 0)
x_shuffle, y_shuffle = shuffle_data(x, y, seed=seed)
x_train, x_test, y_train, y_test = train_test_split(
    x_shuffle, y_shuffle, test_size=0.8, random_state=0)
history = model.fit(x_train, y_train, epochs=epochs, shuffle=True, 
    batch_size=batch_size, verbose=2, 
    validation_data=(x_test, y_test),
    callbacks=callback(name="model_cnn_critical"),
    )
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.title(r"critical={}".format(pc))
# plt.title(pc)
# plt.title(r"critical={} loss={} val_loss={} accuracy={} val_accuracy={}"\
#     .formmat(pc, loss, val_loss, accuracy, val_accuracy))
plt.legend(loc='upper left')
plt.savefig(r".\figure\critical={}.png".format(pc))
# plt.show()
print("times = ", (time.time()-start_time)/60)
