import os
import matplotlib.pyplot as plt
import numpy as np

from utils import read_data

file_location = os.getcwd()
loss_pi = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pi-loss")
mae_pi = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pi-mae")
rmse_pi = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pi-rmse")
val_loss_pi = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pi-val_loss")  
val_mae_pi = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pi-val_mae")
val_rmse_pi = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pi-val_rmse")

loss_p = read_data(file_location=file_location + r"\data", 
    name="train-cnn-p-loss")
mae_p = read_data(file_location=file_location + r"\data", 
    name="train-cnn-p-mae")
rmse_p = read_data(file_location=file_location + r"\data", 
    name="train-cnn-p-rmse")
val_loss_p = read_data(file_location=file_location + r"\data", 
    name="train-cnn-p-val_loss")  
val_mae_p = read_data(file_location=file_location + r"\data", 
    name="train-cnn-p-val_mae")
val_rmse_p = read_data(file_location=file_location + r"\data", 
    name="train-cnn-p-val_rmse")

x = np.arange(len(loss_p))

fig = plt.figure(figsize=(14, 8))

plt.subplot(2,3,1) 
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

plt.subplot(2,3,2) 
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

plt.subplot(2,3,3) 
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

plt.subplot(2,3,4) 
plt.plot(x, loss_p, color="dodgerblue", label='Training set', linewidth=2)
plt.plot(x, val_loss_p, color="indianred", label='Testing set', linewidth=2)
plt.xlabel('epoch', fontsize=14)
plt.ylabel('MSE', fontsize=14)
plt.ylim(-0.001, 0.013)
plt.xticks(size=12)
plt.yticks(size=12)
plt.legend()
plt.grid(True, linestyle='--', linewidth=1.5)
plt.text(50, 0.011, '(d)', fontsize=14)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)

plt.subplot(2,3,5) 
plt.plot(x, mae_p, color="dodgerblue", label='Training set', linewidth=2)
plt.plot(x, val_mae_p, color="indianred", label='Testing set', linewidth=2)
plt.xlabel('epoch', fontsize=14)
plt.ylabel('MAE', fontsize=14)
plt.ylim(-0.01, 0.13)
plt.xticks(size=12)
plt.yticks(size=12)
plt.legend()
plt.grid(True, linestyle='--', linewidth=1.5)
plt.text(50, 0.11, '(e)', fontsize=14)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)

plt.subplot(2,3,6) 
plt.plot(x, rmse_p, color="dodgerblue", label='Training set', linewidth=2)
plt.plot(x, val_rmse_p, color="indianred", label='Testing set', linewidth=2)
plt.xlabel('epoch', fontsize=14)
plt.ylabel('RMSE', fontsize=14)
plt.ylim(-0.01, 0.13)
plt.xticks(size=12)
plt.yticks(size=12)
plt.legend()
plt.grid(True, linestyle='--', linewidth=1.5)
plt.text(50, 0.11, '(f)', fontsize=14)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)

plt.subplots_adjust(wspace=0.35, hspace=0.35)
plt.savefig(r".\figure\cnn-epoch-loss-pi-p.pdf")
plt.savefig(r".\figure\cnn-epoch-loss-pi-p.eps")
plt.show()

