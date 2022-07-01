

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

from utils import read_data, save_data, shuffle_data, checkpoints
from CNN import CNN0

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
split_rate = 0.8

num_classes = 1
l2_rate = 0
filter1 = 32
filter2 = 64
fc1 = 128
dropout_rate = 0.5

file_location = os.getcwd()
x = read_data(file_location=file_location + r"\data", 
    name="x_1000")
y = read_data(file_location=file_location+r"\data", 
    name="Pi_1000")

noise_factor = 0.2
y_noisy = y + noise_factor * tf.random.normal(shape=y.shape) 
y_noisy = tf.clip_by_value(y_noisy, clip_value_min=0., clip_value_max=1.)

print(max(y_noisy),min(y_noisy))
# noise_rate = 0.50
# for i in np.arange(len(x)):
#     length = int(image_height*image_height*noise_rate)
#     l = list(np.where(x[i, :]==0)[0])
#     if len(l) < length:
#         length = len(l)
#     s = random.sample(l, length)
#     for j in s:
#         x[i, j] = 1

x = x.reshape(-1, image_height, image_height, 1)
y_noisy = np.tile(y_noisy, sweeps)
x_shuffle, y_shuffle = shuffle_data(x, y_noisy, seed=seed)
x_train, x_test, y_train, y_test = train_test_split(
    x_shuffle, y_shuffle, test_size=split_rate, random_state=0)

cnn = CNN0(num_classes=num_classes, l2_rate=l2_rate,
    filter1=filter1, filter2=filter2, 
    fc1=fc1, dropout_rate=dropout_rate)
cnn.build(input_shape=[None, image_height, image_height, 1])
cnn.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
cnn.compile(optimizer=optimizer, 
    loss="mean_squared_error", 
    metrics=["mae", tf.keras.metrics.RootMeanSquaredError(name='rmse')])

history = cnn.fit(x_train, y_train, epochs=epochs, shuffle=True, 
    batch_size=batch_size, verbose=2, 
    validation_data=(x_test, y_test),
    callbacks=checkpoints(name="model_cnn_pi")
)

save_data(file_location=file_location + r"\data", 
    name="train-cnn-pi-loss", value=history.history['loss'])    
save_data(file_location=file_location + r"\data", 
    name="train-cnn-pi-mae", value=history.history['mae'])    
save_data(file_location=file_location + r"\data", 
    name="train-cnn-pi-rmse", value=history.history['rmse'])  
save_data(file_location=file_location + r"\data", 
    name="train-cnn-pi-val_loss", value=history.history['val_loss'])    
save_data(file_location=file_location + r"\data", 
    name="train-cnn-pi-val_mae", value=history.history['val_mae'])    
save_data(file_location=file_location + r"\data", 
    name="train-cnn-pi-val_rmse", value=history.history['val_rmse'])  

plt.plot(history.history['mae'], label='mae')
plt.plot(history.history['val_mae'], label='val_mae')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend(loc='upper left')
plt.savefig(r".\figure\train-cnn-pi.pdf")
plt.savefig(r".\figure\train-cnn-pi.eps")
plt.show()

