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

from utils import read_data, shuffle_data, checkpoints, save_data
from CNN import CNN, CNN_c

start_time = time.time()
seed = 12345
random.seed(seed)
np.random.seed(seed=seed)

sweeps = 10**3
image_height = 28
image_size = 28*28
epochs = 10**3
learning_rate = 1e-4#
batch_size = 512#256
split_rate = 0.8

num_classes = 1
l2_rate = 0#1e-3
filter1 = 32
filter2 = 64
fc1 = 128
dropout_rate = 0.5

file_location = os.getcwd()
x = read_data(file_location=file_location + r"\data", 
    name="x_1000").reshape(-1, image_height, image_height, 1)
y = read_data(file_location=file_location+r"\data", 
    name="percolation_1000")

# for pc in np.arange(0.55, 0.65+0.01, 0.01):
pc = 0.60
y = np.where(y>=pc, 1, 0)
x_shuffle, y_shuffle = shuffle_data(x, y, seed=seed)
x_train, x_test, y_train, y_test = train_test_split(
    x_shuffle, y_shuffle, test_size=split_rate, random_state=seed)

model = CNN(num_classes=num_classes, l2_rate=l2_rate,
    filter1=filter1, filter2=filter2, fc1=fc1, dropout_rate=dropout_rate)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
# optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
model.compile(optimizer=optimizer, 
    loss=tf.keras.losses.binary_crossentropy, 
    metrics=[tf.keras.metrics.binary_accuracy])

history = model.fit(x_train, y_train, epochs=epochs, shuffle=True, 
    batch_size=batch_size, verbose=2, 
    validation_data=(x_test, y_test),
    callbacks=checkpoints(name="model_cnn_critical_percolation_classifier"))

# save_data(file_location=file_location + r"\data", 
#     name="train-cnn-pc-loss-{}".format(int(pc*100)), 
#     value=history.history['loss'])    
# save_data(file_location=file_location + r"\data", 
#     name="train-cnn-pc-binary_accuracy-{}".format(int(pc*100)), 
#     value=history.history['binary_accuracy'])    
# save_data(file_location=file_location + r"\data", 
#     name="train-cnn-pc-val_loss-{}".format(int(pc*100)), 
#     value=history.history['val_loss'])    
# save_data(file_location=file_location + r"\data", 
#     name="train-cnn-pc-val_binary_accuracy-{}".format(int(pc*100)), 
#     value=history.history['val_binary_accuracy'])     

save_data(file_location=file_location + r"\data", 
    name="train-cnn-pc-loss-601", 
    value=history.history['loss'])    
save_data(file_location=file_location + r"\data", 
    name="train-cnn-pc-binary_accuracy-601", 
    value=history.history['binary_accuracy'])    
save_data(file_location=file_location + r"\data", 
    name="train-cnn-pc-val_loss-601", 
    value=history.history['val_loss'])    
save_data(file_location=file_location + r"\data", 
    name="train-cnn-pc-val_binary_accuracy-601", 
    value=history.history['val_binary_accuracy'])     

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.title(r"critical={}".format(pc))
plt.legend(loc='upper left')
plt.savefig(r".\figure\critical={}.png".format(pc))
plt.show()


print("times = ", (time.time()-start_time)/60)
