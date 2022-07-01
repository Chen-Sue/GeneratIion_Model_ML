
import numpy as np
import pandas as pd
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["CUDA_VISIBLE_DEVICES"]="0"  # gpu  "-1"  # cpu
gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.keras.backend.set_floatx('float32')
tf.config.experimental.set_virtual_device_configuration(gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4608),
    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4608)])
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses, optimizers, initializers

from utils import rec_loss, kl_div, reparameter

image_height = 28
latent_dim = 200
intermediate_dim = 256


class CAE(Model):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Sequential([
            layers.Input(shape=(28, 28, 1)), 
            layers.Conv2D(filters=16, kernel_size=(3,3), strides=(2,2), activation='relu'),
            layers.Conv2D(filters=8, kernel_size=(3,3), strides=(2,2), activation='relu'),
            # layers.Flatten(),            
            # layers.Dense(latent_dim, activation='sigmoid'), # No activation
        ])
        self.decoder = Sequential([
            # layers.InputLayer(input_shape=(latent_dim,)),
            # layers.Dense(units=7*7*32, activation='relu'),
            # layers.Reshape(target_shape=(7, 7, 32)),
            layers.Conv2DTranspose(filters=8, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu'),
            layers.Conv2DTranspose(filters=16, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu'),
            layers.Conv2DTranspose(filters=1, kernel_size=(3,3), strides=(1,1), padding="same", activation='sigmoid'),
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

