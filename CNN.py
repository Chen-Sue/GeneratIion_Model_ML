import os
import tensorflow as tf
from tensorflow.keras import layers, initializers, regularizers

def CNN0(num_classes, l2_rate, 
            filter1, filter2, fc1, dropout_rate):
    cnn = tf.keras.Sequential()
    cnn.add(layers.Conv2D(
            filters=filter1, 
            kernel_size=(3, 3), 
            activation='relu', 
        ))
    cnn.add(layers.MaxPooling2D((2, 2)))
    cnn.add(layers.Conv2D(
            filters=filter2, 
            kernel_size=(3, 3), 
            activation='relu', 
        ))
    cnn.add(layers.MaxPooling2D((2, 2)))    
    cnn.add(layers.Flatten())    
    cnn.add(layers.Dropout(dropout_rate))
    cnn.add(layers.Dense(fc1, 
            activation='relu', 
        ))
    cnn.add(layers.Dense(num_classes))
    return cnn

def CNN(num_classes, l2_rate, 
            filter1, filter2, fc1, dropout_rate):
    cnn = tf.keras.Sequential()
    cnn.add(layers.Conv2D(
            filters=filter1, 
            kernel_size=(3, 3), 
            activation='relu', 
            kernel_initializer=initializers.he_normal(), # c
            kernel_regularizer=regularizers.l2(l2_rate), # c
        ))
    cnn.add(layers.MaxPooling2D((2, 2)))
    cnn.add(layers.Conv2D(
            filters=filter2, 
            kernel_size=(3, 3), 
            activation='relu', 
            kernel_initializer=initializers.he_normal(), # c
            kernel_regularizer=regularizers.l2(l2_rate), # c
        ))
    cnn.add(layers.MaxPooling2D((2, 2)))    
    cnn.add(layers.Flatten())    
    cnn.add(layers.Dropout(dropout_rate))
    cnn.add(layers.Dense(fc1, 
            activation='relu', 
            kernel_initializer=initializers.he_normal(), # c
            kernel_regularizer=regularizers.l2(l2_rate), # c
        ))
    cnn.add(layers.Dense(num_classes))
    if num_classes != 1:
        cnn.add(layers.Softmax())
    if dropout_rate != 0: # c
        cnn.add(layers.Activation("sigmoid"))
    return cnn

def CNN_c(num_classes, l2_rate, 
            filter1, filter2, fc1, dropout_rate):
    cnn = tf.keras.Sequential()
    cnn.add(layers.Conv2D(
            filters=filter1, 
            kernel_size=(3, 3), 
            activation='relu', 
            kernel_initializer=initializers.he_normal(), # c
            kernel_regularizer=regularizers.l2(l2_rate), # c
        ))
    # model.add(layers.BatchNormalization())
    # model.add(layers.Activation('relu'))
    cnn.add(layers.MaxPooling2D((2, 2)))
    cnn.add(layers.Conv2D(
            filters=filter2, 
            kernel_size=(3, 3), 
            activation='relu', 
            kernel_initializer=initializers.he_normal(), # c
            kernel_regularizer=regularizers.l2(l2_rate), # c
        ))
    # model.add(layers.BatchNormalization())
    # model.add(layers.Activation('relu'))
    cnn.add(layers.MaxPooling2D((2, 2)))    
    cnn.add(layers.Flatten())    
    cnn.add(layers.Dropout(dropout_rate))
    cnn.add(layers.Dense(fc1, 
            activation='relu', 
            kernel_initializer=initializers.he_normal(), # c
            kernel_regularizer=regularizers.l2(l2_rate), # c
        ))
    cnn.add(layers.Dropout(dropout_rate))
    cnn.add(layers.Dense(num_classes, 
        activation='sigmoid'))
    return cnn


# class CNN(tf.keras.Model):

#     def __init__(self, num_classes, l2_rate, 
#             filter1, filter2, fc1, dropout_rate):
#         super().__init__()
#         self.num_classes = num_classes
#         self.l2_rate = l2_rate
#         self.filter1 = filter1
#         self.filter2 = filter2
#         self.fc1 = fc1
#         self.dropout_rate = dropout_rate

#         self.conv1 = layers.Conv2D(
#             filters=self.filter1, 
#             kernel_size=(3, 3), 
#             activation='relu', 
#             # kernel_initializer=initializers.he_normal(), 
#             # kernel_regularizer=regularizers.l2(l2_rate),
#             )
#         self.pool1 = layers.MaxPooling2D((2, 2))
#         self.conv2 = layers.Conv2D(
#             filters=self.filter2, 
#             kernel_size=(3, 3), 
#             activation='relu', 
#             # kernel_initializer=initializers.he_normal(), 
#             # kernel_regularizer=regularizers.l2(l2_rate),
#             )
#         self.pool2 = layers.MaxPooling2D((2, 2))
#         self.flatten = layers.Flatten()
#         self.dropout = layers.Dropout(self.dropout_rate)
#         self.dense1 = layers.Dense(self.fc1, 
#             activation='relu', 
#             # kernel_initializer=initializers.he_normal(), 
#             # kernel_regularizer=regularizers.l2(l2_rate),
#             )
#         self.dense2 = layers.Dense(self.num_classes)
#         self.dense3 = layers.Activation('softmax')

#     def call(self, x, training=False):
#         x = tf.reshape(x, [-1, 28, 28, 1])
#         conv1 = self.conv1(x)
#         pool1 = self.pool1(conv1)
#         conv2 = self.conv2(pool1)
#         pool2 = self.pool2(conv2)
#         flatten = self.flatten(pool2)
#         dropout = self.dropout(flatten)
#         dense1 = self.dense1(dropout)
#         output = self.dense2(dense1)
#         if self.num_classes != 1:
#             output = self.dense3(output)
#         return output
