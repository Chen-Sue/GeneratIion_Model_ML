
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import time
from scipy.ndimage import measurements
import tensorflow as tf

from utils import read_data

start_time = time.time()

per = linspace(0.41, 0.8, 40)  # 0.593
Lattices = [28]  
L = 28
sweeps = 10**3
image_height = 28
p = np.linspace(0.41, 0.80, 40) # 0.593

file_location = os.getcwd() + r"\VAE"
Pi = read_data(file_location=file_location + r"\data" , name="Pi_1000")
print(Pi.shape) 
P = read_data(file_location=file_location + r"\data", name="P_1000")
print(P.shape) 
per = read_data(file_location=file_location + r"\data", name="per_1000")
print(per.shape) 

x = read_data(file_location=file_location + r"\data", 
    name="x_1000").reshape(-1, image_height, image_height, 1)
x_vae = read_data(file_location=file_location + r"\data", 
    name="vae-x_decoded").reshape(-1, image_height, image_height, 1)
x_cvae = read_data(file_location=file_location + r"\data", 
    name="cvae-x_decoded").reshape(-1, image_height, image_height, 1)

model_pi = tf.keras.models.load_model(file_location +
    r"\model_cnn_pi" +
    r"\20210111-120537" +
    r"\00662-0.000818-0.018294-0.028606-0.001806-0.021274-0.042496.h5")
model_p = tf.keras.models.load_model(file_location +
    r"\model_cnn_p" +
    r"\20201226-203238" +
    r"\00833-0.000315-0.012202-0.017740-0.000772-0.016134-0.027789.h5")

# fig = plt.figure(figsize=(6,8)) 
# ax = fig.add_subplot(211) 
fig = plt.figure(figsize=(8,4)) 
ax = fig.add_subplot(121) 
plt.plot(p, Pi, c='grey', marker="*", markersize=12, label="MC", linewidth=2)
y_pred = model_pi.predict(x_vae)
y_pred_average = [y_pred[i::40] for i in np.arange(40)]
y_pred_average = np.array(y_pred_average).reshape(-1, 1000)
y_pred_average = np.mean(y_pred_average, axis=1)
plt.plot(p, y_pred_average, c='dodgerblue', marker="o", 
    label="VAE and CNN-I", linewidth=2)
y_pred = model_pi.predict(x_cvae)
y_pred_average = [y_pred[i::40] for i in np.arange(40)]
y_pred_average = np.array(y_pred_average).reshape(-1, 1000)
y_pred_average = np.mean(y_pred_average, axis=1)
plt.plot(p, y_pred_average, c='indianred', marker="s", 
    label="cVAE and CNN-I", linewidth=2)
ax.text(0.40, 0.98, '(a)', fontsize=14)
plt.xlabel('$p$', fontsize=14)
plt.ylabel('$\Pi(p,L)$', fontsize=14)
plt.xticks(size=12)
plt.yticks(size=12)
plt.xlim(0.39, 0.81)
plt.ylim(-0.09, 1.09)
plt.legend(loc="lower right", fontsize=12)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
from pylab import *
tick_params(which='major', width=2)
tick_params(which='minor', width=1)
ax.xaxis.set_major_locator(plt.MultipleLocator(0.05))
ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
# ax = fig.add_subplot(212) 
ax = fig.add_subplot(122) 
y_pred = model_p.predict(x)
y_pred_average = [y_pred[i::40] for i in np.arange(40)]
y_pred_average = np.array(y_pred_average).reshape(-1, 1000)
y_pred_average = np.mean(y_pred_average, axis=1)
# ax.scatter(p, y_pred_average, c='', edgecolors='dodgerblue', marker="o")
plt.plot(p, P, c='grey', marker="*", markersize=12, label="MC", linewidth=2)
y_pred = model_p.predict(x_vae)
y_pred_average = [y_pred[i::40] for i in np.arange(40)]
y_pred_average = np.array(y_pred_average).reshape(-1, 1000)
y_pred_average = np.mean(y_pred_average, axis=1)
plt.plot(p, y_pred_average, c='dodgerblue', marker="o", 
    label="VAE and CNN-II", linewidth=2)
y_pred = model_p.predict(x_cvae)
y_pred_average = [y_pred[i::40] for i in np.arange(40)]
y_pred_average = np.array(y_pred_average).reshape(-1, 1000)
y_pred_average = np.mean(y_pred_average, axis=1)
plt.plot(p, y_pred_average, c='indianred', marker="s", 
    label="cVAE and CNN-II", linewidth=2)
ax.text(0.40, 0.98, '(b)', fontsize=14)
plt.xlabel('$p$', fontsize=14)
plt.ylabel('$P(p,L)$', fontsize=14)
plt.xticks(size=12)
plt.yticks(size=12)
plt.xlim(0.39, 0.81)
plt.ylim(-0.09, 1.09)
plt.legend(loc="lower right", fontsize=12)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
from pylab import *
tick_params(which='minor', width=1)
tick_params(which='major', width=2)
ax.xaxis.set_major_locator(plt.MultipleLocator(0.05))
ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
plt.subplots_adjust(left=None,bottom=None,
    right=None,top=None,wspace=0.5,hspace=None)
plt.tight_layout()
plt.savefig(file_location + r"\figure\vae-cvae-percolation-pi-p.pdf")
plt.savefig(file_location + r"\figure\vae-cvae-percolation-pi-p.eps")
plt.show()

fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(1,1,1) 
# plt.plot(per, Pi,"y--o",label='mc')
# plt.scatter(per, Pi, c='dodgerblue', marker="o")
# plt.plot(per, Pi, c='dodgerblue', marker="o")
y_pred = model.predict(x)
y_pred_average = [y_pred[i::40] for i in np.arange(40)]
y_pred_average = np.array(y_pred_average).reshape(-1, 1000)
y_pred_average = np.mean(y_pred_average, axis=1)
# ax.scatter(p, y_pred_average, c='', edgecolors='dodgerblue', marker="o")
plt.plot(p, y_pred_average, c='dodgerblue', marker="^", label="CNN")
y_pred = model.predict(x_vae)
y_pred_average = [y_pred[i::40] for i in np.arange(40)]
y_pred_average = np.array(y_pred_average).reshape(-1, 1000)
y_pred_average = np.mean(y_pred_average, axis=1)
# ax.scatter(p, y_pred_average, c='', edgecolors='indianred', marker="*")
plt.plot(p, y_pred_average, c='indianred', marker="*", label="VAE")
y_pred = model.predict(x_cvae)
y_pred_average = [y_pred[i::40] for i in np.arange(40)]
y_pred_average = np.array(y_pred_average).reshape(-1, 1000)
y_pred_average = np.mean(y_pred_average, axis=1)
# ax.scatter(p, y_pred_average, c='', edgecolors='green', marker="+")
plt.plot(p, y_pred_average, c='darkviolet', marker="+", label="cVAE")
plt.grid(True, linestyle='--')
plt.xlabel('percolation')
plt.ylabel('$\Pi$')
plt.ylim(-0.1, 1.1)
plt.legend(loc="upper left")
plt.savefig(file_location + r"\figure\cnn-percolation-pi.pdf")
plt.savefig(file_location + r"\figure\cnn-percolation-pi.eps")
plt.show()

fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(1,1,1) 
# plt.plot(per, P, c='dodgerblue', marker="o")
model = tf.keras.models.load_model(file_location +
    r"\model_cnn_p" +
    r"\20201226-120736" +
    r"\01000-0.000275-0.011431-0.016584-0.000780-0.015896-0.027922.h5")
y_pred = model.predict(x)
y_pred_average = [y_pred[i::40] for i in np.arange(40)]
y_pred_average = np.array(y_pred_average).reshape(-1, 1000)
y_pred_average = np.mean(y_pred_average, axis=1)
# ax.scatter(p, y_pred_average, c='', edgecolors='dodgerblue', marker="o")
plt.plot(p, y_pred_average, c='dodgerblue', marker="^", label="CNN")
y_pred = model.predict(x_vae)
y_pred_average = [y_pred[i::40] for i in np.arange(40)]
y_pred_average = np.array(y_pred_average).reshape(-1, 1000)
y_pred_average = np.mean(y_pred_average, axis=1)
# ax.scatter(p, y_pred_average, c='', edgecolors='indianred', marker="*")
plt.plot(p, y_pred_average, c='indianred', marker="*", label="VAE")
y_pred = model.predict(x_cvae)
y_pred_average = [y_pred[i::40] for i in np.arange(40)]
y_pred_average = np.array(y_pred_average).reshape(-1, 1000)
y_pred_average = np.mean(y_pred_average, axis=1)
# ax.scatter(p, y_pred_average, c='', edgecolors='green', marker="+")
plt.plot(p, y_pred_average, c='darkviolet', marker="+", label="cVAE")
plt.grid(True, linestyle='--')
plt.xlabel('percolation')
plt.ylabel('P(p,L)')
plt.ylim(-0.1, 1.1)
plt.legend(loc="upper left")
plt.savefig(r".\figure\cnn-percolation-p.pdf")
plt.savefig(r".\figure\cnn-percolation-p.eps")
plt.show()

end_time = time.time()
print("times = ", (end_time-start_time)/60)
