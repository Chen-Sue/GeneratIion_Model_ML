import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, initializers

class VAE(tf.keras.Model):

    def __init__(self, h_dim1, latent_dim, image_size):
        super().__init__()
        self.h_dim1 = h_dim1
        # self.h_dim2 = h_dim2
        self.latent_dim = latent_dim
        self.image_size = image_size

        # input => h
        self.fc1 = layers.Dense(self.h_dim1, 
            # kernel_initializer=initializers.he_normal(), 
            activation="relu",
        )
        # self.fc2 = layers.Dense(self.h_dim2, 
        #     # kernel_initializer=initializers.he_normal(), 
        #     activation="relu",
        # )
        # h => mu and variance
        self.fc3 = layers.Dense(self.latent_dim, name="x_mu")
        self.fc4 = layers.Dense(self.latent_dim, name="x_log_var")

        # sampled z => h
        # self.fc5 = layers.Dense(self.h_dim2, 
        #     # kernel_initializer=initializers.he_normal(), 
        #     activation="relu",
        # )
        self.fc6 = layers.Dense(self.h_dim1, 
            # kernel_initializer=initializers.he_normal(), 
            activation="relu",
        )
        # h => image
        self.fc7 = layers.Dense(self.image_size)

    def encoder(self, x):
        h1 = self.fc1(x)       
        # h2 = self.fc2(h1)       
        x_mu = self.fc3(h1)
        x_log_var = self.fc4(h1)
        return x_mu, x_log_var

    def reparameter(self, x_mu, x_log_var, seed=12345):
        batch = tf.shape(x_mu)[0]
        dim = tf.shape(x_mu)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim), seed=seed)
        std = tf.exp(x_log_var) ** 0.5
        z = x_mu + std * epsilon
        return z

    def decode_logits(self, z):
        # h1 = self.fc5(z)       
        h1 = self.fc6(z)
        x_hat_logits = self.fc7(h1)
        return x_hat_logits

    def decoder(self, z):
        return tf.nn.sigmoid(self.decode_logits(z))

    def call(self, inputs, training=None, mask=None):
        mu, log_var = self.encoder(inputs)
        z = self.reparameter(mu, log_var)
        x_hat_logits = self.decode_logits(z)
        return x_hat_logits, mu, log_var

    def rec_loss(self, x, x_hat):
        reconstruction_loss = tf.reduce_mean(keras.losses.binary_crossentropy(x, x_hat))
        reconstruction_loss *= self.image_size
        return reconstruction_loss

    def kl_div(self, x_mu, x_log_var): 
        kl_loss = 1 + x_log_var - tf.square(x_mu) - tf.exp(x_log_var)
        kl_loss = tf.reduce_mean(kl_loss)
        kl_loss *= -0.5
        return kl_loss

    def compute_loss(self, x):
        x_mu, x_log_var = self.encoder(x)
        z = self.reparameter(x_mu, x_log_var)
        x_hat = self.decoder(z)
        reconstruction_loss = self.rec_loss(x, x_hat)
        kl_loss = self.kl_div(x_mu, x_log_var)
        total_loss = reconstruction_loss + kl_loss
        return total_loss, kl_loss, reconstruction_loss

    def train_step(self, x):
        if isinstance(x, tuple):
            x = x[0]
        with tf.GradientTape() as tape:
            total_loss, kl_loss, reconstruction_loss = self.compute_loss(x)
        grads = tape.gradient(total_loss, self.trainable_weights) 
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights)) 
        return {"loss": total_loss,
                "reconstruction_loss": reconstruction_loss,
                "kl_loss": kl_loss}
