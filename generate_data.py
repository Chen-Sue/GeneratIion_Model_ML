
from pylab import *
from scipy.ndimage import measurements
import time
import h5py
import numpy as np
import random
import os

import config
from utils import save_data

start_time = time.time()

sweeps = config.sweeps 
image_height = config.image_height
image_size = config.image_size
p = config.p

file_location = os.getcwd()
Pi = np.zeros(len(p))
P = np.zeros(len(p))
x = np.empty(shape=(0, image_size))
percolation = np.empty(shape=(0, 1))

for sweep in np.arange(1, sweeps+1, 1):       
    z = np.random.rand(image_height, image_height)
    for ip in np.arange(len(p)):
        m = z<p[ip]
        lw, num = measurements.label(m) #  label the clusters
        # check if the set of labels on the up side and 
        # the set of labels on the down side have any intersection  
        perc_x = intersect1d(lw[0,:], lw[-1,:]) 
        # If the length of the set of intersections is larger than zero, 
        # there is at least one percolating cluster, 
        # find the intersection of two arrays
        perc = perc_x[where(perc_x>0)]
        if (len(perc)>0):
            Pi[ip] = Pi[ip] + 1
            area = measurements.sum(m, lw, index=perc[0])
            P[ip] = P[ip] + area   
        percolation = np.concatenate((percolation, np.reshape(np.array(p[ip]), (1, 1))), axis=0)
        x = np.concatenate((x, np.reshape(np.array(m), (1, image_size))), axis=0)
    print('sweep=', sweep, 'x.shape=', x.shape, 'percolation.shape=', percolation.shape, 
        'Pi.shape=', Pi.shape, 'P.shape=', P.shape, )  

Pi = Pi/sweeps
P = P/(sweeps*image_size)

save_data(file_location=file_location+r'\data', name='x', value=x)    
save_data(file_location=file_location+r'\data', name='Pi', value=Pi)    
save_data(file_location=file_location+r'\data', name='P', value=P)      
save_data(file_location=file_location+r'\data', name='percolation', value=percolation)      

end_time = time.time()
print('times = ', (end_time-start_time)/60)
