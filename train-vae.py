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

from VAE import VAE
from utils import checkpoints, binary, save_data, read_data

start_time = time.time()
seed = 12345
random.seed(seed)
np.random.seed(seed=seed)

image_size = 28*28
h_dim1 = 512
# h_dim2 = 256
latent_dim = 256
batch_size = 256
epochs = 10**3
learning_rate = 1e-3

file_location = os.getcwd()
x = read_data(file_location=file_location + r"\data", name="x_1000")
percolation = read_data(file_location=file_location + r"\data", 
    name="percolation_1000")

vae = VAE(h_dim1, latent_dim, image_size)
vae.build(input_shape=(batch_size, image_size))
vae.summary()
optimizer = tf.keras.optimizers.Adam(learning_rate)
vae.compile(optimizer=optimizer)
training_history = vae.fit(x=x, y=None, shuffle=True, 
    epochs=epochs, batch_size=batch_size, verbose=2, 
    # validation_data=(x_test, x_test),
    callbacks=checkpoints(name="model_vae")
)

loss = training_history.history["loss"]
reconstruction_loss = training_history.history["reconstruction_loss"]
kl_loss = training_history.history["kl_loss"]
save_data(file_location=file_location + r"\data", 
    name="vae-loss", value=loss)    
save_data(file_location=file_location + r"\data", 
    name="vae-reconstruction_loss", value=reconstruction_loss)    
save_data(file_location=file_location + r"\data", 
    name="vae-kl_loss", value=kl_loss)  

mu, log_var = vae.encoder(x)
z = vae.reparameter(mu, log_var)
x_hat_logits = vae.decode_logits(z)
x_hat = vae.decoder(z)
x_hat_binary = binary(x=x_hat.numpy(), per=percolation, sweeps=1000, skip=40)
save_data(file_location=file_location + r"\data", 
    name="vae-x_encoded", value=z)    
save_data(file_location=file_location + r"\data", 
    name="vae-x_decoded_logits", value=x_hat_logits)  
save_data(file_location=file_location + r"\data", 
    name="vae-x_decoded", value=x_hat)  
save_data(file_location=file_location + r"\data", 
    name="vae-x_decoded_binary", value=x_hat_binary) 

print("times = ", (time.time()-start_time)/60)