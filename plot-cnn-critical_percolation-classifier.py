



import os
import matplotlib.pyplot as plt
os.environ['TF_CPP_spill_LOG_LEVEL']='3'
os.environ['CUDA_VISIBLE_DEVICES']='0'  # gpu if '0'; cpu if '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4608),
    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4608)])
tf.keras.backend.clear_session()
import matplotlib.pyplot as plt
from tensorflow import keras
import h5py
from pylab import *
import time
import datetime
import random
import math

from sklearn.model_selection import train_test_split

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA, IncrementalPCA

from utils import binary, save_data, read_data, reparameter, colorsys, color, ncolors, findIntersection
import config

start_time = time.time()

seed = config.seed
p = config.p # 0.593
image_height = config.image_height
image_size = config.image_size
epochs = config.epochs
learning_rate = config.learning_rate
batch_size = config.batch_size
l2_rate = config.l2_rate
hyp = config.hyp

random.seed(seed)
np.random.seed(seed=seed)
file_location = os.getcwd()
x = read_data(file_location=file_location + r'\data', name='x').reshape(-1, image_height, image_height, 1)
P = read_data(file_location=file_location + r'\data', name='P')
percolation = read_data(file_location=file_location + r'\data', name='percolation')
Pi = read_data(file_location=file_location + r'\data', name='Pi')

# hyp = np.around(np.arange(0.01, 0.86-0.005, 0.01), 2)

colors = list(map(lambda x: color(tuple(x)), ncolors(len(hyp))))
markers = ['p', 'd', 'v', '^', 'x', 'o', '+', '<', '>', 's', '*']
# fig = plt.figure(figsize=(18, 8))
fig = plt.figure(figsize=(18,10)) 
ax = fig.add_subplot(211)    
# plt.grid(True, axis='x', linestyle='--', linewidth=1.5) 
# plt.axhline(y=0.5, color='black', linestyle='--', linewidth=1.5)
# plt.axvline(x=0.593, color='black', linestyle='--', linewidth=1.5)

difference =[]
for key, value in enumerate(hyp):
    file_dir = r'D:\hoffee\generate_model\VAE-v2\model_cnn_critical_percolation_classifier'
    file_dir = file_dir+r'\{}\\'.format(value)
    list = os.listdir(file_dir)
    list.sort(key=lambda fn: os.path.getmtime(file_dir+fn) if not os.path.isdir(file_dir+fn) else 0)
    model = keras.models.load_model(file_dir+str(list[-1]))
    y_pred = model.predict(x)
    y_pred_average = [y_pred[i::101] for i in np.arange(101)]
    y_pred_average = np.array(y_pred_average).reshape(-1, 1000)
    y_pred_average = np.mean(y_pred_average, axis=1)
    plt.plot(p, y_pred_average, c=colors[key], marker=markers[key%11], \
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
# np.savetxt()
plt.xlabel('$p$', size=18)
plt.ylabel('average outputs', size=18)
plt.xlim(-0.009, 1.009)
plt.ylim(-0.02, 1.02)
plt.axhline(y=0.5, color="black", linestyle="--", linewidth=1.5)
plt.legend(loc='upper left', ncol=6, fontsize=10)
plt.tick_params(labelsize=16)
ax.text(0.00, 1.05, '(a)', fontsize=18)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
from pylab import *
tick_params(which='major', width=2)
ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
ax.xaxis.set_minor_locator(plt.MultipleLocator(0.01))
# plt.savefig(r'.\figure\cnn-critical.pdf')
# plt.savefig(r'.\figure\cnn-critical.eps')
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(12, 6))
ax = fig.add_subplot(212) 
# plt.axvline(x=0.6, color='darkviolet', linestyle='--', linewidth=1.5)
# plt.grid(True, axis='y', linestyle='--', linewidth=1.5)
plt.plot(hyp, difference, markersize=4,
    linewidth=2, marker='o', color='dodgerblue', label='absolute')   
plt.scatter(hyp[59], difference[59], marker='s', 
    c='indianred', label='$p_{\mathrm{critical}}$=0.594', s=100)
# for key,value in zip(hyp, difference):
#     plt.text(key,value+0.05,value) 
plt.xlabel(u'$p_{\mathrm{preset}}$', fontsize=18)                
plt.ylabel(u'$|\Delta p|$', fontsize=18)  
plt.tick_params(labelsize=16)
plt.xlim(-0.009, 1.009)
plt.ylim(-0.02, 1.02)
plt.legend(loc='upper right', fontsize=16)
ax.text(0.00, 1.05, '(b)', fontsize=18)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
from pylab import *
tick_params(which='major', width=2)
ax.xaxis.set_major_locator(plt.MultipleLocator(0.1))
ax.yaxis.set_major_locator(plt.MultipleLocator(0.1))
ax.xaxis.set_minor_locator(plt.MultipleLocator(0.01))
plt.tight_layout()
plt.savefig(r'.\figure\cnn-diff.pdf')
plt.show()