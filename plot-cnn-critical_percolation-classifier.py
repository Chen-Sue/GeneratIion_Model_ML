



import os
from tensorflow import keras
from tensorflow.keras import models
import matplotlib.pyplot as plt
import tensorflow as tf
import os
os.environ["TF_CPP_spill_LOG_LEVEL"]="3"
os.environ["CUDA_VISIBLE_DEVICES"]="0"  # gpu if "0"; cpu if "-1"
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_virtual_device_configuration(gpus[0],
#     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4608),
#     tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4608)])
tf.keras.backend.clear_session()
import matplotlib.pyplot as plt
from tensorflow import keras
import h5py
from pylab import *
import time
import datetime
import random

from sklearn.model_selection import train_test_split
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA, IncrementalPCA

from utils import binary, save_data, read_data, findIntersection

start_time = time.time()
seed = 12345
random.seed(seed)
np.random.seed(seed=seed)

p = np.linspace(0.41, 0.80, 40) # 0.593

file_location = os.getcwd() + r"\VAE"
image_height = 28
image_size = 28*28
epochs = 10**3
learning_rate = 1e-4
batch_size = 256
l2_rate = 1e-3
# l2_rate = tf.keras.optimizers.schedules.ExponentialDecay(
#     decay_steps=epochs, initial_learning_rate=l2_rate, 
#     decay_rate=0.999, staircase=False)

# fig = plt.figure(figsize=(7,10)) 
# ax = fig.add_subplot(211) 
fig = plt.figure(figsize=(10,4)) 
ax = fig.add_subplot(121) 

p = np.linspace(0.41, 0.80, 40) # 0.593
x = read_data(file_location=file_location + r"\data", 
    name="x_1000").reshape(-1, image_height, image_height, 1)

colors = ['y', 'm', 'c', 'b', 'indianred', 'DarkCyan', 
    'darkviolet', 'dodgerblue', 'grey', 'midnightblue', 'g']
markers = ['p', 'd', 'v', '^', 'x', 'o', '+', '<', '>', 
    's', '*']

# hyp = [0.60]

hyp = [0.55, 0.56, 0.57, 0.58, 0.59, 0.60, 0.61, 0.62, 0.63, 0.64, 0.65]
difference =[]
for key, value in enumerate(hyp):
    file_dir = file_location + r'\model_cnn_critical_percolation_classifier'
    file_dir = file_dir+r'\%.2f'%value +'\\'
    list = os.listdir(file_dir)
    list.sort(key=lambda fn: os.path.getmtime(file_dir+fn) if not os.path.isdir(file_dir+fn) else 0)
    model = keras.models.load_model(file_dir+str(list[-1]))
    y_pred = model.predict(x)
    y_pred_average = [y_pred[i::40] for i in np.arange(40)]
    y_pred_average = np.array(y_pred_average).reshape(-1, 1000)
    y_pred_average = np.mean(y_pred_average, axis=1)
    plt.plot(p, y_pred_average, c=colors[key], marker=markers[key], \
        label='%.2f'%value, linewidth=1.5, markersize=8)
    plt.axvline(x=value, color=colors[key], linestyle="--", linewidth=1.5)

    first_y = max(y_pred_average-0.5)+0.5
    second_y = min(y_pred_average-0.5)+0.5
    # c = abs(x-abs(Z-x).min())
    first_y = max(y_pred_average[np.where((y_pred_average-0.5)<=0)])
    second_y = min(y_pred_average[np.where((y_pred_average-0.5)>0)])
    first_x, second_x = value-0.01, value
    x1, y1, x2, y2 = 0, 0.5, 1, 0.5
    x3, y3, x4, y4 = first_x, first_y, second_x, second_y
    y_cross = np.around(findIntersection(x1,y1,x2,y2,x3,y3,x4,y4),4)[0]
    print(key+1, '假设', value, '\tANN求得', y_cross, second_y-second_x)
    # print('\t', first_x, second_x, first_y, second_y)
    # print('\t', second_y-second_x)
    difference.append(abs(second_y-second_x))

plt.xlabel('$p$', size=14)
plt.ylabel('average outputs', size=14)
plt.xlim(0.39, 0.81)
plt.axhline(y=0.5, color="black", linestyle="--", linewidth=1.5)
plt.legend(loc='lower right', fontsize=12)
plt.tick_params(labelsize=12)
ax.text(0.40, 0.98, '(a)', fontsize=14)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
from pylab import *
tick_params(which='major', width=2)
ax.xaxis.set_major_locator(plt.MultipleLocator(0.05))
ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))
plt.tight_layout()

# ax = fig.add_subplot(212) 
ax = fig.add_subplot(122) 
# plt.axvline(x=0.6, color='darkviolet', linestyle='--', linewidth=1.5)
# plt.grid(True, axis='y', linestyle='--', linewidth=1.5)
plt.plot(hyp, difference, markersize=4, linewidth=2, 
    marker='o', color='dodgerblue', label='absolute')   
plt.scatter(hyp[5], difference[5], marker='s', 
    c='indianred', label='$p_{\mathrm{critical}}$=0.595', s=100)
# for key,value in zip(hyp, difference):
#     plt.text(key,value+0.05,value) 
plt.xlabel(u'$p_{\mathrm{preset}}$', fontsize=14)                
plt.ylabel(u'$|\Delta p|$', fontsize=14)  
plt.tick_params(labelsize=12)
plt.xlim(0.545, 0.655)
plt.ylim(-0.002, 0.057)
plt.legend(loc='upper right', fontsize=12)
ax.text(0.55, 0.052, '(b)', fontsize=14)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
from pylab import *
tick_params(which='major', width=2)
ax.xaxis.set_major_locator(plt.MultipleLocator(0.01))
ax.yaxis.set_major_locator(plt.MultipleLocator(0.01))
# ax.xaxis.set_minor_locator(plt.MultipleLocator(0.01))
# ax.yaxis.set_minor_locator(plt.MultipleLocator(0.01))
plt.subplots_adjust(left=None,bottom=None,
    right=None,top=None,wspace=None,hspace=None)
plt.tight_layout()
plt.savefig(file_location + r"\figure\cnn-critical.pdf")
plt.savefig(file_location + r"\figure\cnn-critical.eps")
plt.show()