
import os
import matplotlib.pyplot as plt
import numpy as np

from utils import read_data

file_location = os.getcwd()
loss_pi = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pi-loss-inference")
mae_pi = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pi-mae-inference")
rmse_pi = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pi-rmse-inference")
val_loss_pi = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pi-val_loss-inference")  
val_mae_pi = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pi-val_mae-inference")
val_rmse_pi = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pi-val_rmse-inference")

x = np.arange(len(loss_pi))

fig = plt.figure(figsize=(14, 4.5))

plt.subplot(1,3,1) 
plt.plot(x, loss_pi, color="dodgerblue", label='Training set', linewidth=2)
plt.plot(x, val_loss_pi, color="indianred", label='Testing set', linewidth=2)
plt.xlabel('epoch', fontsize=14)
plt.ylabel('MSE', fontsize=14)
plt.ylim(-0.001, 0.013)
plt.xticks(size=12)
plt.yticks(size=12)
plt.legend()
plt.grid(True, linestyle='--', linewidth=1.5)
plt.text(50, 0.011, '(a)', fontsize=14)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)

plt.subplot(1,3,2) 
plt.plot(x, mae_pi, color="dodgerblue", label='Training set', linewidth=2)
plt.plot(x, val_mae_pi, color="indianred", label='Testing set', linewidth=2)
plt.xlabel('epoch', fontsize=14)
plt.ylabel('MAE', fontsize=14)
plt.ylim(-0.01, 0.13)
plt.xticks(size=12)
plt.yticks(size=12)
plt.legend()
plt.grid(True, linestyle='--', linewidth=1.5)
plt.text(50, 0.11, '(b)', fontsize=14)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)

plt.subplot(1,3,3) 
plt.plot(x, rmse_pi, color="dodgerblue", label='Training set', linewidth=2)
plt.plot(x, val_rmse_pi, color="indianred", label='Testing set', linewidth=2)
plt.xlabel('epoch', fontsize=14)
plt.ylabel('RMSE', fontsize=14)
plt.ylim(-0.01, 0.13)
plt.xticks(size=12)
plt.yticks(size=12)
plt.legend()
plt.grid(True, linestyle='--', linewidth=1.5)
plt.text(50, 0.11, '(c)', fontsize=14)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)

plt.subplots_adjust(wspace=0.35, hspace=0.35)
plt.savefig(r".\figure\cnn-epoch-loss-pi-p-inference.pdf")
plt.savefig(r".\figure\cnn-epoch-loss-pi-p-inference.eps")
plt.show()
