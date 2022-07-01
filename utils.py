#-------------------------------------------------------------#
#                         Model utilities                     #
#                                                             #
#              Sampling of the 2D percolation Problem.        #
#                                                             #
#                             2020                            #
#                           Shu Cheng                         #
#-------------------------------------------------------------#

import numpy as np
import tensorflow as tf
import datetime
import os
import h5py

def read_data(file_location, name):
    data = h5py.File(file_location + r'\{}.h5'.format(name),'r')
    data = data['elem'][:].astype('float32')
    # train_db = tf.data.Dataset.from_tensor_slices(data
    #             ).shuffle(self.batch_size*4).batch(self.batch_size)
    return data

def save_data(file_location, name, value):
    with h5py.File(file_location + r'\{}.h5'.format(name),'w') as hf:
        hf.create_dataset('elem', data=value, 
            compression='gzip', compression_opts=9)
        hf.close()

def binary(x, per, sweeps=1000, skip=40):
    for i in np.arange(sweeps):
        x[i*skip:(i+1)*skip] = np.where(
            x[i*skip:(i+1)*skip]>=per[i], 1, 0)
    return x

def checkpoints(name):
    current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    model_dir = os.path.join(name)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model_dir = os.path.join(model_dir, current_time)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if name == 'model_ae' or name == 'model_cae':
        filepath = os.path.join(model_dir, 
            '{epoch:05d}-{loss:.6f}-{mae:.6f}-{rmse:.6f}.h5')
    elif name == 'model_vae' or name == 'model_cvae':
        filepath = os.path.join(model_dir, 
            '{epoch:05d}-{loss:.6f}-{reconstruction_loss:.6f}-{kl_loss:.6f}.h5')
        checkpoint = [tf.keras.callbacks.ModelCheckpoint(filepath, 
                    monitor='loss', verbose=2, 
                    save_best_only=True, save_weights_only=False),
                tf.keras.callbacks.EarlyStopping(monitor='loss', 
                    patience=10000, verbose=True)]                 
    elif name == 'model_cnn_p' or \
        name == 'model_cnn_pi' or \
        name == 'model_cnn_percolation' or \
        name == 'model_cnn_critical_percolation_reg40':
        filepath = os.path.join(model_dir, 
            '{epoch:05d}-{loss:.6f}-{mae:.6f}-{rmse:.6f}-' +
            '{val_loss:.6f}-{val_mae:.6f}-{val_rmse:.6f}.h5')
        checkpoint = [tf.keras.callbacks.ModelCheckpoint(filepath, 
                    monitor='val_loss', verbose=2, 
                    save_best_only=True, save_weights_only=False),
                tf.keras.callbacks.EarlyStopping(monitor='loss', 
                    patience=10000, verbose=True)]                 
    elif name=='model_cnn_critical_percolation_classifier':
        filepath = os.path.join(model_dir, 
            '{epoch:05d}-{loss:.6f}-{binary_accuracy:.6f}-' +
            '{val_loss:.6f}-{val_binary_accuracy:.6f}.h5')
        checkpoint = [tf.keras.callbacks.ModelCheckpoint(filepath, 
                monitor='val_loss', verbose=2, 
                save_best_only=True,save_weights_only=False)]   
    return checkpoint
    
def reparameter(x_mu, x_log_var, seed=12345):
    batch = tf.shape(x_mu)[0]
    dim = tf.shape(x_mu)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim), seed=seed)
    std = tf.exp(x_log_var) ** 0.5
    z = x_mu + std * epsilon
    return z

def shuffle_data(x, y, seed=12345):
    np.random.seed(seed=seed)
    index = [i for i in np.arange(len(x))]
    np.random.shuffle(index)
    x = x[index].astype('float32')
    y = y[index]
    return x, y

def findIntersection(x1,y1,x2,y2,x3,y3,x4,y4):
    px= ( (x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) ) 
    py= ( (x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) )
    return [px, py]