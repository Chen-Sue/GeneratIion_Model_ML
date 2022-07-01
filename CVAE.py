
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, Sequential, initializers


class CVAE(tf.keras.Model):

    def __init__(self, filter1=32, filter2=64, latent_dim=200, image_size=28**2):
        super().__init__()
        self.filter1 = filter1
        self.filter2 = filter2
        self.latent_dim = latent_dim
        self.image_size = image_size

        # input => h
        self.conv1 = layers.Conv2D(filters=self.filter1, 
            kernel_size=3, strides=(2, 2), 
            activation='relu', 
            # kernel_initializer=initializers.he_normal(), 
        )
        self.conv2 = layers.Conv2D(filters=self.filter2, 
            kernel_size=3, strides=(2, 2), 
            activation='relu',
            # kernel_initializer=initializers.he_normal(), 
        )
        self.flatten1 = layers.Flatten()
        # h => mu and variance
        self.x_mu = layers.Dense(self.latent_dim, name='x_mu')
        self.x_log_var = layers.Dense(self.latent_dim, name='x_log_var')

        # sampled z => h
        self.dense1 = layers.Dense(units=7*7*32,
            activation='relu', 
            # kernel_initializer=initializers.he_normal(), 
        )
        self.reshape = layers.Reshape(target_shape=(7, 7, 32))
        self.conv3 = layers.Conv2DTranspose(filters=self.filter2,  
            kernel_size=3, strides=(2, 2), padding='same',
            activation='relu',  
            # kernel_initializer=initializers.he_normal(),  
        )
        self.conv4 = layers.Conv2DTranspose(filters=self.filter1, 
            kernel_size=3, strides=(2, 2), padding='same',
            activation='relu',  
            # kernel_initializer=initializers.he_normal(), 
        )
        # h => image
        self.conv5 = layers.Conv2DTranspose(filters=1, 
            kernel_size=3, strides=(1, 1), padding='same',
            activation=None,  
            # kernel_initializer=initializers.he_normal(), 
        )

    def encoder(self, x):
        h1 = self.conv1(x)       
        h2 = self.conv2(h1)       
        h3 = self.flatten1(h2)            
        x_mu = self.x_mu(h3)
        x_log_var = self.x_log_var(h3)
        return x_mu, x_log_var

    def reparameter(self, x_mu, x_log_var, seed=12345):
        batch = tf.shape(x_mu)[0]
        dim = tf.shape(x_mu)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim), seed=seed)
        std = tf.exp(x_log_var) ** 0.5
        z = x_mu + std * epsilon
        return z

    def decode_logits(self, z):
        h1 = self.dense1(z)       
        h2 = self.reshape(h1)       
        h3 = self.conv3(h2)
        h4 = self.conv4(h3)
        x_hat_logits = self.conv5(h4)
        return x_hat_logits

    def decoder(self, z):
        return tf.nn.sigmoid(self.decode_logits(z))

    def call(self, inputs, training=None, mask=None):
        inputs = tf.reshape(inputs, [-1, 28, 28, 1])
        mu, log_var = self.encoder(inputs)
        z = self.reparameter(mu, log_var)
        x_hat_logits = self.decode_logits(z)
        return x_hat_logits, mu, log_var

    def rec_loss(self, x, x_hat):
        reconstruction_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(x, x_hat))
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
        return {'loss': total_loss,
                'reconstruction_loss': reconstruction_loss,
                'kl_loss': kl_loss}

# # encoder
# encoder_inputs = keras.Input(shape=(28, 28, 1))
# encoder_h1 = layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), 
#     activation='relu', kernel_initializer=initializers.he_normal(), 
#     kernel_regularizer=keras.regularizers.l2(l2_rate))(encoder_inputs)
# encoder_h2 = layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), 
#     activation='relu', kernel_initializer=initializers.he_normal(), 
#     kernel_regularizer=keras.regularizers.l2(l2_rate))(encoder_h1)
# encoder_h3 = layers.Flatten()(encoder_h2)
# x_mu = layers.Dense(latent_dim, name='x_mu')(encoder_h3)
# x_log_var = layers.Dense(latent_dim, name='x_log_var')(encoder_h3)
# z = reparameter(x_mu, x_log_var)
# encoder = keras.Model(encoder_inputs, [x_mu, x_log_var, z], name='encoder')
# encoder.summary() 
# # decoder
# latent_inputs = keras.Input(shape=(latent_dim,))
# decoder_h1 = layers.Dense(units=7*7*32,
#     activation='relu', kernel_initializer=initializers.he_normal(), 
#     kernel_regularizer=keras.regularizers.l2(l2_rate),)(latent_inputs)
# decoder_h2 = layers.Reshape((7, 7, 32))(decoder_h1)
# decoder_h3 = layers.Conv2DTranspose(filters=64,  kernel_size=3, strides=(2, 2), 
#     activation='relu',  kernel_initializer=initializers.he_normal(), 
#     kernel_regularizer=keras.regularizers.l2(l2_rate),
#     padding='same',)(decoder_h2)
# decoder_h4 = layers.Conv2DTranspose(filters=32, kernel_size=3, strides=(2, 2), 
#     activation='relu',  kernel_initializer=initializers.he_normal(), 
#     kernel_regularizer=keras.regularizers.l2(l2_rate),
#     padding='same',)(decoder_h3)
# decoder_outputs = layers.Conv2DTranspose(filters=1, kernel_size=3, strides=(1, 1), 
#             activation='relu',  kernel_initializer=initializers.he_normal(), 
#             kernel_regularizer=keras.regularizers.l2(l2_rate),
#             padding='same',)(decoder_h4)
# decoder = keras.Model(latent_inputs, decoder_outputs, name='decoder')
# decoder.summary()


# class CVAE(tf.keras.Model):

#     def __init__(self, encoder, decoder, image_size=28*28):
#         super().__init__()
#         self.encoder = encoder
#         self.decoder = decoder
#         self.image_size = image_size

#     def call(self, inputs):
#         self.x_mu, self.x_log_var, self.z = self.encoder(inputs)
#         self.x_hat = self.decoder(self.z)
#         return self.x_hat

#     def rec_loss(self, x, x_hat):
#         reconstruction_loss = tf.reduce_mean(keras.losses.binary_crossentropy(x, x_hat))
#         reconstruction_loss *= self.image_size
#         return reconstruction_loss

#     def kl_div(self, x_mu, x_log_var): 
#         kl_loss = 1 + x_log_var - tf.square(x_mu) - tf.exp(x_log_var)
#         kl_loss = tf.reduce_mean(kl_loss)
#         kl_loss *= -0.5
#         return kl_loss

#     def compute_loss(self, x):
#         x_mu, x_log_var, z = self.encoder(x)
#         x_hat = self.decoder(z)
#         reconstruction_loss = self.rec_loss(x, x_hat)
#         kl_loss = self.kl_div(x_mu, x_log_var)
#         total_loss = reconstruction_loss + kl_loss
#         return total_loss, kl_loss, reconstruction_loss

#     def train_step(self, x):
#         if isinstance(x, tuple):
#             x = x[0]
#         with tf.GradientTape() as tape:
#             total_loss, kl_loss, reconstruction_loss = self.compute_loss(x)
#         grads = tape.gradient(total_loss, self.trainable_weights) 
#         self.optimizer.apply_gradients(zip(grads, self.trainable_weights)) 
#         return {'loss': total_loss,
#                 'reconstruction_loss': reconstruction_loss,
#                 'kl_loss': kl_loss}


