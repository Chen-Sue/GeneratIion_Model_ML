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

from utils import read_data, shuffle_data, callback
from CNN import CNN

start_time = time.time()
seed = 12345
random.seed(seed)
np.random.seed(seed=seed)

sweeps = 10**3
image_height = 28
image_size = 28*28
epochs = 10**3
learning_rate = 1e-2
batch_size = 256
split_rate = 0.8

num_classes = 40
l2_rate = 0
filter1 = 32
filter2 = 64
fc1 = 128
dropout_rate = 0.5

file_location = os.getcwd()
x = read_data(file_location=file_location + r"\data", 
    name="x_1000").reshape(-1, image_height, image_height, 1)
y = read_data(file_location=file_location+r"\data", 
    name="percolation_1000")
# y = (y/0.01-41).astype(int)
# y = tf.squeeze(y)
# y = tf.one_hot(y, depth=40) 
x_shuffle, y_shuffle = shuffle_data(x, y, seed=seed)
y_shuffle = (y_shuffle-min(y_shuffle))*100

x_train, x_test, y_train, y_test = train_test_split(
    x_shuffle, y_shuffle, test_size=split_rate, random_state=0)

# db = tf.data.Dataset.from_tensor_slices((x,y))
# db = db.map(preprocess).shuffle(batch_size*10).batch(batch_size)

model = CNN(num_classes=num_classes, l2_rate=l2_rate,
    filter1=filter1, filter2=filter2, fc1=fc1, dropout_rate=dropout_rate)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# loss = 'categorical_crossentropy'
metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
model.compile(optimizer=optimizer,
                loss=loss,
                metrics=metrics)   
history = model.fit(x_train, y_train, epochs=epochs, shuffle=True, 
    batch_size=batch_size, verbose=2, 
    validation_data=(x_test, y_test),
    # callbacks=callback(name="model_cnn_critical_reg"),
)

# loss = tf.losses.CategoricalCrossentropy(from_logits=True)
# metrics = [tf.keras.metrics.CategoricalAccuracy()]
# network.compile(optimizer=optimizer,
#                 loss='categorical_crossentropy',
#                 metrics=['accuracy'])
# network.fit(db, epochs=100, # validation_data=test_db, 
#             validation_freq=2, callbacks=callback())

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.title(r"critical")
plt.legend(loc='upper left')
plt.savefig(r".\figure\critical.png")
plt.show()

# network.save_weights('ckpt/weights.ckpt')
# print('saved to ckpt/weights.ckpt')
# network.load_weights('ckpt/weights.ckpt')
# print('loaded weights from file.')
# weight_values = [v.name for v in network.weights]
# print("variables: {}".format(weight_values))
# weight = network.get_layer('dense_1')
# print(weight)
# weight = network.get_weights()[2]
# print(network.get_weights()[2])
# y_pre = network.predict(db)
# print(y_pre.shape, y_pre)
# order = tf.reduce_mean(y_pre*weight, 0).numpy()
# def save_data(name, value):
#     import h5py
#     with h5py.File(r".\data\{}.h5".format(name),'w') as hf:
#         hf.create_dataset("elem", data=value, compression="gzip", compression_opts=9)
#         hf.close()
# save_data(name="mlp", value=order)


print("times = ", (time.time()-start_time)/60)
