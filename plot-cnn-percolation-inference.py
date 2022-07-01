
import os
os.environ["TF_CPP_spill_LOG_LEVEL"]="3"
os.environ["CUDA_VISIBLE_DEVICES"]="0"  # gpu if "0"; cpu if "-1"
import time
import random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_virtual_device_configuration(gpus[0],
#     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4608),
#     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4608)])
tf.keras.backend.clear_session()

from utils import binary, save_data, read_data, \
    reparameter, shuffle_data

start_time = time.time()
seed = 12345
random.seed(seed)
np.random.seed(seed=seed)

p = np.linspace(0.41, 0.80, 40) # 0.593
p1 = np.linspace(0.01, 0.40, 40) 
p2 = np.linspace(0.81, 1, 20) 

file_location = os.getcwd() + r"\VAE"
image_height = 28
x = read_data(file_location=file_location + r"\data", 
    name="x_1000").reshape(-1, image_height, image_height, 1)
x_spill = read_data(file_location=file_location + r"\data", 
    name="x_min").reshape(-1, image_height, image_height, 1)
y = read_data(file_location=file_location+r"\data", 
    name="percolation_1000")
y_spill = read_data(file_location=file_location+r"\data", 
    name="percolation_min")

# model = tf.keras.models.load_model(file_location + # 100
#     r"\model_cnn_percolation" +
#     r"\20201226-200500" +
#     r"\00100-0.000102-0.007884-0.010095-0.000323-0.013819-0.017972.h5")
model = tf.keras.models.load_model(file_location + # 1000
    r"\model_cnn_percolation" +
    r"\20201226-200824" +
    r"\00968-0.000001-0.000725-0.000958-0.000189-0.010358-0.013753.h5")

y_spill_pred = model.predict(x_spill)
y_pred = model.predict(x)

y_spill_average = [y_spill_pred[i::60] for i in np.arange(60)]
y_spill_average = np.array(y_spill_average).reshape(-1, 1000)
y_pred_average = [y_pred[i::40] for i in np.arange(40)]
y_pred_average = np.array(y_pred_average).reshape(-1, 1000)

y_error = np.concatenate((y_spill_average[:40, :], 
    y_pred_average, y_spill_average[40:, :]), axis=0)
y_error_min = np.min(y_error, axis=1)
y_error_max = np.max(y_error, axis=1)
error_range = [y_error_min, y_error_max]

y_spill_average = np.mean(y_spill_average, axis=1)
y_pred_average = np.mean(y_pred_average, axis=1)

y_average = np.concatenate((y_spill_average[:40], 
    y_pred_average, y_spill_average[40:]), axis=0)
y_average = list(y_average)
error_range = [(y_error_min-y_average)/y_average, 
    (y_error_max-y_average)/y_average]

fig = plt.figure(figsize=(5, 4))
plt.scatter(y_spill, y_spill_pred, c='indianred', marker="o", s=2.5, 
    label='initial')
plt.scatter(y, y_pred, c='dodgerblue', marker="o", s=2.5, 
    label='extrapolated')
plt.plot(np.linspace(0.01, 1.00, 100), y_average, label='averaged', c='black')
plt.xlabel('$p$', fontsize=14)
plt.ylabel('$\hat{p}$ by CNN-III', fontsize=14)
plt.xlim(-0.015, 1.015)
plt.ylim(-0.08, 1.02)
plt.xticks(size=12)
plt.yticks(size=12)
plt.legend(loc='upper left', fontsize=12)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
from pylab import *
tick_params(which='major', width=2)
ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
plt.subplots_adjust(left=None,bottom=None,
    right=None,top=None,wspace=0.5,hspace=None)
plt.tight_layout()
plt.savefig(r".\figure\cnn-percolation-inference.pdf")
plt.savefig(r".\figure\cnn-percolation-inference.eps")
plt.show()


# model_vae = tf.keras.models.load_model(file_location + 
#     r"\model_vae" +
#     r"\20201227-201539" +
#     r"\00981-24.414131-19.164835-5.249296.h5")
# model_cvae = tf.keras.models.load_model(file_location + 
#     r"\model_cvae" +
#     r"\20201227-211938" +
#     r"\00930-4.828875-1.399238-3.429636.h5")
# model_vae = tf.keras.models.load_model(file_location + 
#     r"\model_cvae" +
#     r"\20201227-211938" +
#     r"\00930-4.828875-1.399238-3.429636.h5")

x = read_data(file_location=file_location + r"\data", 
    name="x_1000").reshape(-1, image_height, image_height, 1)
x_vae = read_data(file_location=file_location + r"\data", 
    name="vae-x_decoded").reshape(-1, image_height, image_height, 1)
x_cvae = read_data(file_location=file_location + r"\data", 
    name="cvae-x_decoded").reshape(-1, image_height, image_height, 1)

x_spill = read_data(file_location=file_location + r"\data", 
    name="x_min").reshape(-1, image_height, image_height, 1)
# x_spill_vae = model_vae.predict(x_spill)
# x_spill_vae = model_cvae.predict(x_spill)

y = read_data(file_location=file_location+r"\data", 
    name="percolation_1000")
y_spill = read_data(file_location=file_location+r"\data", 
    name="percolation_min")

y_pred = model.predict(x)
# y_spill_pred = model.predict(x_spill)
y_pred_vae = model.predict(x_vae)
# y_spill_pred_vae = model.predict(x_spill_vae)
y_pred_cvae = model.predict(x_cvae)
# y_spill_pred_cvae = model.predict(x_spill_cvae)

# y_spill_average = [y_spill_pred[i::60] for i in np.arange(60)]
# y_spill_average = np.array(y_spill_average).reshape(-1, 1000)
y_pred_average = [y_pred[i::40] for i in np.arange(40)]
y_pred_average = np.array(y_pred_average).reshape(-1, 1000)
# y_spill_average_vae = [y_spill_pred_vae[i::60] for i in np.arange(60)]
# y_spill_average_vae = np.array(y_spill_average_vae).reshape(-1, 1000)
y_pred_average_vae = [y_pred_vae[i::40] for i in np.arange(40)]
y_pred_average_vae = np.array(y_pred_average_vae).reshape(-1, 1000)
# y_spill_average_cvae = [y_spill_pred_cvae[i::60] for i in np.arange(60)]
# y_spill_average_cvae = np.array(y_spill_average_cvae).reshape(-1, 1000)
y_pred_average_cvae = [y_pred_cvae[i::40] for i in np.arange(40)]
y_pred_average_cvae = np.array(y_pred_average_cvae).reshape(-1, 1000)

# y_error = np.concatenate((y_spill_average[:40, :], 
#     y_pred_average, y_spill_average[40:, :]), axis=0)
# y_error_min = np.min(y_error, axis=1)
# y_error_max = np.max(y_error, axis=1)
# error_range = [y_error_min, y_error_max]

# y_spill_average = np.mean(y_spill_average, axis=1)
# y_pred_average = np.mean(y_pred_average, axis=1)
y_pred_average_vae = np.mean(y_pred_average_vae, axis=1)
y_pred_average_cvae = np.mean(y_pred_average_cvae, axis=1)

# y_average = np.concatenate((y_spill_average[:40], 
#     y_pred_average, y_spill_average[40:]), axis=0)
# y_average = list(y_average)
# error_range = [(y_error_min-y_average)/y_average, 
#     (y_error_max-y_average)/y_average]


y_average_vae = list(y_pred_average_vae)
y_average_cvae = list(y_pred_average_cvae)
# error_range = [(y_error_min-y_average)/y_average, 
#     (y_error_max-y_average)/y_average]

# fig = plt.figure(figsize=(6, 8))
# ax = fig.add_subplot(211) 
fig = plt.figure(figsize=(8, 4))
ax = fig.add_subplot(121) 
# plt.scatter(y_spill, y_spill_pred_vae, c='indianred', marker="o", s=2.5, 
#     label='raw permeability in dataset')
plt.scatter(y, y_pred_vae, c='dodgerblue', marker="o", s=2.5, 
    label='VAE and CNN-III')
plt.plot(np.linspace(0.41, 0.80, 40), y_average_vae, label='averaged', c='black')
# plt.plot(np.linspace(0.01, 1.00, 100), y_average_vae, c='black')
plt.xlabel('$p$', fontsize=14)
plt.ylabel('$\hat{p}$', fontsize=14)
# plt.xlim(-0.09, 1.09)
# plt.ylim(-0.09, 1.09)
plt.xlim(0.39, 0.82)
plt.ylim(0.33, 0.86)
plt.xticks(size=12)
plt.yticks(size=12)
ax.text(0.40, 0.8, '(a)', fontsize=14)
plt.legend(loc='lower right', fontsize=12)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
from pylab import *
tick_params(which='major', width=2)
ax.xaxis.set_major_locator(plt.MultipleLocator(0.05))
ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
# ax = fig.add_subplot(212) 
ax = fig.add_subplot(122) 
# plt.scatter(y_spill, y_spill_pred_cvae, c='indianred', marker="o", s=2.5, 
#     label='raw permeability in dataset')
plt.scatter(y, y_pred_cvae, c='dodgerblue', marker="o", s=2.5, 
    label='cVAE and CNN-III')
# plt.plot(np.linspace(0.01, 1.00, 100), y_average, c='black')
plt.plot(np.linspace(0.41, 0.80, 40), y_average_cvae, 
    label='averaged', c='black')
plt.xlabel('$p$', fontsize=14)
plt.ylabel('$\hat{p}$', fontsize=14)
# plt.xlim(-0.09, 1.09)
# plt.ylim(-0.09, 1.09)
plt.xlim(0.39, 0.82)
plt.ylim(0.33, 0.86)
plt.xticks(size=12)
plt.yticks(size=12)
plt.legend(loc='lower right', fontsize=12)
ax.text(0.40, 0.8, '(b)', fontsize=14)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
from pylab import *
tick_params(which='major', width=2)
ax.xaxis.set_major_locator(plt.MultipleLocator(0.05))
ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
plt.subplots_adjust(left=None,bottom=None,
    right=None,top=None,wspace=0.8,hspace=None)
plt.tight_layout()

plt.savefig(file_location + r"\figure\vae-cvae-percolation-inference.pdf")
plt.savefig(file_location + r"\figure\vae-cvae-percolation-inference.eps")
plt.show()