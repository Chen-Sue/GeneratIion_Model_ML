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

from CVAE import CVAE
from utils import checkpoints, binary, save_data, read_data, reparameter

start_time = time.time()
seed = 12345
random.seed(seed)
np.random.seed(seed=seed)

image_height = 28
image_size = 28*28
filter1 = 32
filter2 = 64
latent_dim = 400
batch_size = 256
epochs = 10**3
learning_rate = 1e-3

file_location = os.getcwd()
x = read_data(file_location=file_location + r"\data",  name="x_1000"
    ).reshape(-1, image_height, image_height, 1)
percolation = read_data(file_location=file_location+r"\data", 
    name="percolation_1000")

cvae = CVAE(filter1, filter2, latent_dim, image_size)
cvae.build(input_shape=(batch_size, 28, 28, 1))
cvae.summary()
# print(cvae.layers)
# print(cvae.weights)
optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
cvae.compile(optimizer=optimizer)
training_history = cvae.fit(x=x, shuffle=True, verbose=2,
                            epochs=epochs, batch_size=batch_size, 
                            callbacks=checkpoints(name="model_cvae"))

loss = training_history.history["loss"]
reconstruction_loss = training_history.history["reconstruction_loss"]
kl_loss = training_history.history["kl_loss"]

mu, log_var = cvae.encoder(x)
z = cvae.reparameter(mu, log_var)

x_hat = np.empty(shape=(0, 28, 28, 1))
x_hat_binary = np.empty(shape=(0, 28, 28, 1))
for i in np.arange(40):
    x_hat1 = cvae.decoder(z[i*1000:(i+1)*1000])
    x_hat_binary1 = binary(x=x_hat1.numpy(), 
        per=percolation[i*1000:(i+1)*1000], sweeps=1000, skip=40)
    x_hat = np.concatenate((x_hat, x_hat1), axis=0)
    x_hat_binary = np.concatenate((x_hat_binary, x_hat_binary1), axis=0)

x_hat = x_hat.reshape(-1, 28*28)
x_hat_binary = x_hat_binary.reshape(-1, 28*28)

save_data(file_location=file_location + r"\data", 
    name="cvae-loss", value=loss)   
save_data(file_location=file_location + r"\data", 
    name="cvae-reconstruction_loss", value=reconstruction_loss)   
save_data(file_location=file_location + r"\data", 
    name="cvae-kl_loss", value=kl_loss)  
save_data(file_location=file_location + r"\data", 
    name="cvae-x_encoded", value=z)    
save_data(file_location=file_location + r"\data", 
    name="cvae-x_decoded", value=x_hat)  
save_data(file_location=file_location + r"\data", 
    name="cvae-x_decoded_binary", value=x_hat_binary) 

print("times = ", (time.time()-start_time)/60)
