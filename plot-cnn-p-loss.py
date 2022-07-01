import os
import matplotlib.pyplot as plt
import numpy as np

from utils import read_data

file_location = os.getcwd()
loss = read_data(file_location=file_location + r"\data", 
    name="train-cnn-p-loss")
mae = read_data(file_location=file_location + r"\data", 
    name="train-cnn-p-mae")
rmse = read_data(file_location=file_location + r"\data", 
    name="train-cnn-p-rmse")
val_loss = read_data(file_location=file_location + r"\data", 
    name="train-cnn-p-val_loss")  
val_mae = read_data(file_location=file_location + r"\data", 
    name="train-cnn-p-val_mae")
val_rmse = read_data(file_location=file_location + r"\data", 
    name="train-cnn-p-val_rmse")

x = np.arange(len(loss))

fig = plt.figure(figsize=(14, 4))

plt.subplot(1,3,1) 
plt.plot(x, loss, color="dodgerblue", label='loss')
plt.plot(x, val_loss, color="indianred", label='val_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.ylim(-0.001, 0.011)
plt.legend()
plt.grid(True, linestyle='--')

plt.subplot(1,3,2) 
plt.plot(x, mae, color="dodgerblue", label='mae')
plt.plot(x, val_mae, color="indianred", label='val_mae')
plt.xlabel('epoch')
plt.ylabel('MAE')
plt.ylim(-0.01, 0.11)
plt.legend()
plt.grid(True, linestyle='--')

plt.subplot(1,3,3) 
plt.plot(x, rmse, color="dodgerblue", label='rmse')
plt.plot(x, val_rmse, color="indianred", label='val_rmse')
plt.xlabel('epoch')
plt.ylabel('RMSE')
plt.ylim(-0.01, 0.11)
plt.legend()
plt.grid(True, linestyle='--')

plt.subplots_adjust(wspace=0.35)
plt.savefig(r".\figure\cnn-p-loss.pdf")
plt.savefig(r".\figure\cnn-p-loss.eps")
plt.show()