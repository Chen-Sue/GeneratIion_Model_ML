
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'  # gpu if '0'; cpu if '-1'
import numpy as np
os.environ['TF_CPP_spill_LOG_LEVEL']='3'
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
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
import config

start_time = time.time()

seed = config.seed
image_height = config.image_height
image_size = config.image_size
epochs = config.epochs
learning_rate = config.learning_rate
batch_size = config.batch_size
split_rate = config.split_rate

num_classes = config.num_classes
l2_rate = config.l2_rate
filter1 = config.filter1
filter2 = config.filter2
fc1 = config.fc1
dropout_rate = config.dropout_rate

random.seed(seed)
np.random.seed(seed=seed)
file_location = os.getcwd()

for pc in config.hyp:
    x = read_data(file_location=file_location + r'\data', name='x').reshape(-1, image_height, image_height, 1)
    y = read_data(file_location=file_location+r'\data', name='percolation')

    y = np.where(y>=pc, 1, 0)
    x_shuffle, y_shuffle = shuffle_data(x, y, seed=seed)
    x_train, x_test, y_train, y_test = train_test_split(
        x_shuffle, y_shuffle, test_size=split_rate, random_state=seed)

    model = CNN(num_classes=num_classes, l2_rate=l2_rate,
        filter1=filter1, filter2=filter2, fc1=fc1, dropout_rate=dropout_rate)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.binary_crossentropy, 
        metrics=[tf.keras.metrics.binary_accuracy])

    history = model.fit(x_train, y_train, epochs=epochs, shuffle=True, 
        batch_size=batch_size, verbose=2, 
        validation_data=(x_test, y_test),
        callbacks=checkpoints(name="model_cnn_critical_percolation_classifier", pc=pc))

    save_data(file_location=file_location + r'\data', 
        name='train-cnn-pc-loss-{}'.format(int(pc*100)), 
        value=history.history['loss'])    
    save_data(file_location=file_location + r'\data', 
        name='train-cnn-pc-binary_accuracy-{}'.format(int(pc*100)), 
        value=history.history['binary_accuracy'])    
    save_data(file_location=file_location + r'\data', 
        name='train-cnn-pc-val_loss-{}'.format(int(pc*100)), 
        value=history.history['val_loss'])    
    save_data(file_location=file_location + r'\data', 
        name='train-cnn-pc-val_binary_accuracy-{}'.format(int(pc*100)), 
        value=history.history['val_binary_accuracy'])     

    # plt.plot(history.history['loss'], label='loss')
    # plt.plot(history.history['val_loss'], label = 'val_loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('loss')
    # plt.ylim(-0.1, 0.31)
    # plt.title(r'critical={}'.format(pc))
    # plt.legend(loc='upper left')
    # plt.savefig(r'.\figure\critical={}.png'.format(pc))
    # plt.show()


    print('times = ', (time.time()-start_time)/60)
