
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ["CUDA_VISIBLE_DEVICES"]="0"  # gpu  "-1"  # cpu
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4608),
    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4608)])
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses, optimizers, Sequential
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model

from generate_data import CAE
from utils import callback, binary, save_data

file_location = r"C:\Users\cshu\Desktop\hoffee\generate_model\VAE\data"
latent_dim = 1 
learning_rate = 1e-2
epochs = 10**3
batch_size=256

x = dataset(file_location=file_location, name="x").reshape(-1, 28, 28, 1)

cae = CAE(latent_dim=latent_dim)          
optimizer = optimizers.Adam(lr=learning_rate)
cae.compile(optimizer=optimizer, 
            loss=losses.MeanSquaredError(),
            metrics=['accuracy'])
cae.fit(x, x, epochs=epochs, shuffle=True,
        # validation_data=(x_test, x_test),
        batch_size=batch_size,
        callbacks=callback(name="model_cae"),)

cae_encoded = cae.encoder(x).numpy()
cae_decoded = cae.decoder(ae_encoded).numpy()
percolation = dataset(name="percolation")
cae_decoded = binary(x=cae_decoded, per=percolation)

save_data(file_location=file_location + r".\latent=1",
    name="cae_encoded", value=cae_encoded)
save_data(file_location=file_location + r".\latent=1",
    name="cae_decoded", value=cae_decoded)