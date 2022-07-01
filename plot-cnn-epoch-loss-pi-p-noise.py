
import os
import matplotlib.pyplot as plt
import numpy as np

from utils import read_data

file_location = os.getcwd()
loss_pi_005 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pi-loss-0.05")
mae_pi_005 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pi-mae-0.05")
rmse_pi_005 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pi-rmse-0.05")
val_loss_pi_005 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pi-val_loss-0.05")  
val_mae_pi_005 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pi-val_mae-0.05")
val_rmse_pi_005 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pi-val_rmse-0.05")
loss_pi_010 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pi-loss-0.1")
mae_pi_010 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pi-mae-0.1")
rmse_pi_010 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pi-rmse-0.1")
val_loss_pi_010 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pi-val_loss-0.1")  
val_mae_pi_010 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pi-val_mae-0.1")
val_rmse_pi_010 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pi-val_rmse-0.1")
loss_pi_020 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pi-loss-0.2")
mae_pi_020 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pi-mae-0.2")
rmse_pi_020 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pi-rmse-0.2")
val_loss_pi_020 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pi-val_loss-0.2")  
val_mae_pi_020 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pi-val_mae-0.2")
val_rmse_pi_020 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pi-val_rmse-0.2")

loss_p_005 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-p-loss-0.05")
mae_p_005 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-p-mae-0.05")
rmse_p_005 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-p-rmse-0.05")
val_loss_p_005 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-p-val_loss-0.05")  
val_mae_p_005 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-p-val_mae-0.05")
val_rmse_p_005 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-p-val_rmse-0.05")
loss_p_010 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-p-loss-0.1")
mae_p_010 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-p-mae-0.1")
rmse_p_010 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-p-rmse-0.1")
val_loss_p_010 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-p-val_loss-0.1")  
val_mae_p_010 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-p-val_mae-0.1")
val_rmse_p_010 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-p-val_rmse-0.1")
loss_p_020 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-p-loss-0.2")
mae_p_020 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-p-mae-0.2")
rmse_p_020 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-p-rmse-0.2")
val_loss_p_020 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-p-val_loss-0.2")  
val_mae_p_020 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-p-val_mae-0.2")
val_rmse_p_020 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-p-val_rmse-0.2")

x = np.arange(len(loss_pi_005))

fig = plt.figure(figsize=(14, 8))

plt.subplot(2,3,1) 
plt.plot(x, loss_pi_005, color="indianred", label='Training set with 5% noise', linewidth=2)
plt.plot(x, val_loss_pi_005, linestyle=":", color="dodgerblue", label='Testing set with 5% noise', linewidth=2)
plt.plot(x, loss_pi_010, color="m", label='Training set with 10% noise', linewidth=2)
plt.plot(x, val_loss_pi_010, linestyle=":", color="c", label='Testing set with 10% noise', linewidth=2)
plt.plot(x, loss_pi_020, color="g", label='Training set with 20% noise', linewidth=2)
plt.plot(x, val_loss_pi_020, linestyle=":", color="b", label='Testing set with 20% noise', linewidth=2)
plt.xlabel('epoch', fontsize=14)
plt.ylabel('MSE', fontsize=14)
plt.ylim(-0.001, 0.11)
plt.xticks(size=12)
plt.yticks(size=12)
plt.legend()
plt.grid(True, linestyle='--', linewidth=1.5)
plt.text(10, 0.091, '(a)', fontsize=14)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)

plt.subplot(2,3,2) 
plt.plot(x, mae_pi_005, color="indianred", label='Training set with 5% noise', linewidth=2)
plt.plot(x, val_mae_pi_005, linestyle=":", color="dodgerblue", label='Testing set with 5% noise', linewidth=2)
plt.plot(x, mae_pi_010, color="m", label='Training set with 10% noise', linewidth=2)
plt.plot(x, val_mae_pi_010, linestyle=":", color="c", label='Testing set with 10% noise', linewidth=2)
plt.plot(x, mae_pi_020, color="g", label='Training set with 20% noise', linewidth=2)
plt.plot(x, val_mae_pi_020, linestyle=":", color="b", label='Testing set with 20% noise', linewidth=2)
plt.xlabel('epoch', fontsize=14)
plt.ylabel('MAE', fontsize=14)
plt.ylim(-0.01, 0.51)
plt.xticks(size=12)
plt.yticks(size=12)
plt.legend()
plt.grid(True, linestyle='--', linewidth=1.5)
plt.text(10, 0.41, '(b)', fontsize=14)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)

plt.subplot(2,3,3) 
plt.plot(x, rmse_pi_005, color="indianred", label='Training set with 5% noise', linewidth=2)
plt.plot(x, val_rmse_pi_005, linestyle=":", color="dodgerblue", label='Testing set with 5% noise', linewidth=2)
plt.plot(x, rmse_pi_010, color="m", label='Training set with 10% noise', linewidth=2)
plt.plot(x, val_rmse_pi_010, linestyle=":", color="c", label='Testing set with 10% noise', linewidth=2)
plt.plot(x, rmse_pi_020, color="g", label='Training set with 20% noise', linewidth=2)
plt.plot(x, val_rmse_pi_020, linestyle=":", color="b", label='Testing set with 20% noise', linewidth=2)
plt.xlabel('epoch', fontsize=14)
plt.ylabel('RMSE', fontsize=14)
plt.ylim(-0.01, 0.51)
plt.xticks(size=12)
plt.yticks(size=12)
plt.legend()
plt.grid(True, linestyle='--', linewidth=1.5)
plt.text(10, 0.41, '(c)', fontsize=14)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)

plt.subplot(2,3,4) 
plt.plot(x, loss_p_005, color="indianred", label='Training set with 5% noise', linewidth=2)
plt.plot(x, val_loss_p_005, linestyle=":", color="dodgerblue", label='Testing set with 5% noise', linewidth=2)
plt.plot(x, loss_p_010, color="m", label='Training set with 10% noise', linewidth=2)
plt.plot(x, val_loss_p_010, linestyle=":", color="c", label='Testing set with 10% noise', linewidth=2)
plt.plot(x, loss_p_020, color="g", label='Training set with 20% noise', linewidth=2)
plt.plot(x, val_loss_p_020, linestyle=":", color="b", label='Testing set with 20% noise', linewidth=2)
plt.xlabel('epoch', fontsize=14)
plt.ylabel('MSE', fontsize=14)
plt.ylim(-0.001, 0.11)
plt.xticks(size=12)
plt.yticks(size=12)
plt.legend()
plt.grid(True, linestyle='--', linewidth=1.5)
plt.text(10, 0.091, '(d)', fontsize=14)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)

plt.subplot(2,3,5) 
plt.plot(x, mae_p_005, color="indianred", label='Training set with 5% noise', linewidth=2)
plt.plot(x, val_mae_p_005, linestyle=":", color="dodgerblue", label='Testing set with 5% noise', linewidth=2)
plt.plot(x, mae_p_010, color="m", label='Training set with 10% noise', linewidth=2)
plt.plot(x, val_mae_p_010, linestyle=":", color="c", label='Testing set with 10% noise', linewidth=2)
plt.plot(x, mae_p_020, color="g", label='Training set with 20% noise', linewidth=2)
plt.plot(x, val_mae_p_020, linestyle=":", color="b", label='Testing set with 20% noise', linewidth=2)
plt.xlabel('epoch', fontsize=14)
plt.ylabel('MAE', fontsize=14)
plt.ylim(-0.01, 0.51)
plt.xticks(size=12)
plt.yticks(size=12)
plt.legend()
plt.grid(True, linestyle='--', linewidth=1.5)
plt.text(10, 0.41, '(e)', fontsize=14)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)

plt.subplot(2,3,6) 
plt.plot(x, rmse_p_005, color="indianred", label='Training set with 5% noise', linewidth=2)
plt.plot(x, val_rmse_p_005, linestyle=":", color="dodgerblue", label='Testing set with 5% noise', linewidth=2)
plt.plot(x, rmse_p_010, color="m", label='Training set with 10% noise', linewidth=2)
plt.plot(x, val_rmse_p_010, linestyle=":", color="c", label='Testing set with 10% noise', linewidth=2)
plt.plot(x, rmse_p_020, color="g", label='Training set with 20% noise', linewidth=2)
plt.plot(x, val_rmse_p_020, linestyle=":", color="b", label='Testing set with 20% noise', linewidth=2)
plt.xlabel('epoch', fontsize=14)
plt.ylabel('RMSE', fontsize=14)
plt.ylim(-0.01, 0.51)
plt.xticks(size=12)
plt.yticks(size=12)
plt.legend()
plt.grid(True, linestyle='--', linewidth=1.5)
plt.text(10, 0.41, '(f)', fontsize=14)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)

plt.subplots_adjust(wspace=0.35, hspace=0.35)
# plt.savefig(r".\figure\cnn-epoch-loss-pi-p.pdf")
# plt.savefig(r".\figure\cnn-epoch-loss-pi-p.eps")
plt.savefig(r".\figure\cnn-epoch-loss-pi-p-noise.pdf")
plt.savefig(r".\figure\cnn-epoch-loss-pi-p-noise.eps")
plt.show()
