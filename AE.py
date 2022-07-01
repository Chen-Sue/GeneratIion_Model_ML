
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, initializers

class AE(tf.keras.Model):

    def __init__(self, h_dim1, h_dim2, latent_dim, image_size):
        super().__init__()
        self.h_dim1 = h_dim1
        self.h_dim2 = h_dim2
        self.latent_dim = latent_dim
        self.image_size = image_size

        # input => h
        self.fc1 = layers.Flatten()
        self.fc2 = layers.Dense(self.h_dim1, 
            kernel_initializer=initializers.he_normal(), 
            activation="relu",
        )
        self.fc3 = layers.Dense(self.h_dim2, 
            kernel_initializer=initializers.he_normal(), 
            activation="relu",
        )
        # h => z
        self.fc4 = layers.Dense(self.latent_dim, 
            kernel_initializer=initializers.he_normal(), 
            activation="relu",
        )
        
        # sampled z => h
        self.fc5 = layers.Dense(self.h_dim2, 
            kernel_initializer=initializers.he_normal(), 
            activation="relu",
        )
        self.fc6 = layers.Dense(self.h_dim1, 
            kernel_initializer=initializers.he_normal(), 
            activation="relu",
        )
        # h => image
        self.fc7 = layers.Dense(self.image_size)

    def encoder(self, x):
        h1 = self.fc1(x)       
        h2 = self.fc2(h1)       
        h3 = self.fc3(h2)
        z = self.fc4(h3)
        return z

    def decode_logits(self, z):
        h1 = self.fc5(z)       
        h2 = self.fc6(h1)
        x_hat_logits = self.fc7(h2)
        return x_hat_logits

    def decoder(self, z):
        return tf.nn.sigmoid(self.decode_logits(z))

    def call(self, inputs, training=None, mask=None):
        z = self.encoder(inputs)
        x_hat_logits = self.decode_logits(z)
        return x_hat_logits, z
