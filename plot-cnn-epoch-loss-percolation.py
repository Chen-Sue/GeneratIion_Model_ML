import os
import matplotlib.pyplot as plt
import numpy as np

from utils import read_data

file_location = os.getcwd()
loss = read_data(file_location=file_location + r"\data", 
    name="train-cnn-percolation-loss")
mae = read_data(file_location=file_location + r"\data", 
    name="train-cnn-percolation-mae")
rmse = read_data(file_location=file_location + r"\data", 
    name="train-cnn-percolation-rmse")
val_loss = read_data(file_location=file_location + r"\data", 
    name="train-cnn-percolation-val_loss")  
val_mae = read_data(file_location=file_location + r"\data", 
    name="train-cnn-percolation-val_mae")
val_rmse = read_data(file_location=file_location + r"\data", 
    name="train-cnn-percolation-val_rmse")

x = np.arange(len(loss))

fig = plt.figure(figsize=(14, 4.5))

plt.subplot(1,3,1) 
plt.plot(x, loss, color="dodgerblue", label='Training set', linewidth=2)
plt.plot(x, val_loss, color="indianred", label='Testing set', linewidth=2)
plt.xlabel('epoch', fontsize=14)
plt.ylabel('MSE', fontsize=14)
plt.ylim(-0.0001, 0.0051)
plt.xticks(size=12)
plt.yticks(size=12)
plt.legend()
plt.grid(True, linestyle='--', linewidth=1.5)
plt.text(50, 0.0045, '(a)', fontsize=14)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)

plt.subplot(1,3,2) 
plt.plot(x, mae, color="dodgerblue", label='Training set', linewidth=2)
plt.plot(x, val_mae, color="indianred", label='Testing set', linewidth=2)
plt.xlabel('epoch', fontsize=14)
plt.ylabel('MAE', fontsize=14)
plt.ylim(-0.001, 0.051)
plt.xticks(size=12)
plt.yticks(size=12)
plt.legend()
plt.grid(True, linestyle='--', linewidth=1.5)
plt.text(50, 0.045, '(b)', fontsize=14)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)

plt.subplot(1,3,3) 
plt.plot(x, rmse, color="dodgerblue", label='Training set', linewidth=2)
plt.plot(x, val_rmse, color="indianred", label='Testing set', linewidth=2)
plt.xlabel('epoch', fontsize=14)
plt.ylabel('RMSE', fontsize=14)
plt.ylim(-0.001, 0.051)
plt.xticks(size=12)
plt.yticks(size=12)
plt.legend()
plt.grid(True, linestyle='--', linewidth=1.5)
plt.text(50, 0.045, '(c)', fontsize=14)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)

plt.subplots_adjust(wspace=0.35)
plt.savefig(r".\figure\cnn-epoch-loss-percolation.pdf")
plt.savefig(r".\figure\cnn-epoch-loss-percolation.eps")
plt.show()