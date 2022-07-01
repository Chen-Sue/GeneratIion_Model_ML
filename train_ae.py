
from tensorflow.keras import losses, optimizers
import os 
import time
import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
os.environ["CUDA_VISIBLE_DEVICES"]="0"  # gpu if "0"; cpu if "-1"
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(
    gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4608),
    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4608)])

from AE import AE
from utils import callback, binary, save_data, read_data

start_time = time.time()
L = 28
image_size = 28*28
h_dim1 = 256
h_dim2 = 64
latent_dim = 1
batch_size = 512
epochs = 10**3
learning_rate = 1e-4

optimizer = optimizers.Adam(lr=learning_rate)

file_location = os.getcwd()
x = read_data(file_location=file_location+r"\data", 
    name="x_1000").reshape(-1, L*L)
percolation = read_data(file_location=file_location+r"\data", 
    name="percolation_1000")

ae = AE(h_dim1, h_dim2, latent_dim, image_size)      
ae.compile(optimizer=optimizer, 
    loss=losses.MeanSquaredError(),
    metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae'), 
        tf.keras.metrics.RootMeanSquaredError(name='rmse')])
training_history = ae.fit(x, x, epochs=epochs, shuffle=True, 
        batch_size=batch_size, callbacks=callback(name="model_ae"))

mse = training_history.history["loss"]
mae = training_history.history["mae"]
rmse = training_history.history["rmse"]
save_data(file_location=file_location + r"\data", 
    name="ae-mse", value=mse)    
save_data(file_location=file_location + r"\data", 
    name="ae-mae", value=mae)    
save_data(file_location=file_location + r"\data", 
    name="ae-rmse", value=rmse)  

z = ae.encoder(x).numpy()
x_hat_logits = ae.decode_logits(z).numpy()
x_hat = ae.decoder(z).numpy()
x_hat_binary = binary(x=x_hat, per=percolation, sweeps=1000, skip=40)

save_data(file_location=file_location + r"\data", 
    name="x_ae_encoded", value=z)    
save_data(file_location=file_location + r"\data", 
    name="x_ae_decoded_logits", value=x_hat_logits)  
save_data(file_location=file_location + r"\data", 
    name="x_ae_decoded", value=x_hat)  
save_data(file_location=file_location + r"\data", 
    name="x_ae_decoded_binary", value=x_hat_binary) 

print("times = ", (time.time()-start_time)/60)