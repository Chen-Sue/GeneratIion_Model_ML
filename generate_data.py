
from pylab import *
from scipy.ndimage import measurements
import time
import h5py
import numpy as np
import random

start_time = time.time()
sweeps = 10**4 # 6*10**3 
relax = 50
L = 28 #100
p = np.linspace(0.41, 0.80, 40) # 0.593
# p1 = np.linspace(0.01, 0.40, 40) # 0.593
# p2 = np.linspace(0.81, 1, 20) # 0.593
# p = np.hstack((p1, p2))
print(p)
Pi = np.zeros(len(p))
P = np.zeros(len(p))
x = np.empty(shape=(0, L*L))
pp = np.empty(shape=(0, 1))

for sweep in range(sweeps+relax):       
    print("sweep=", sweep, "x.shape=", x.shape)
    z = np.random.rand(L,L)
    for ip in range(len(p)):
        m = z<p[ip]
        lw, num = measurements.label(m) #  label the clusters
        # check if the set of labels on the up side and 
        # the set of labels on the down side have any intersection  
        perc_x = intersect1d(lw[0,:], lw[-1,:]) 
        # If the length of the set of intersections is larger than zero, 
        # there is at least one percolating cluster, 
        # find the intersection of two arrays
        perc = perc_x[where(perc_x>0)]
        if (len(perc)>0) and sweep>=relax:
            Pi[ip] = Pi[ip] + 1
            area = measurements.sum(m, lw, index=perc[0])
            P[ip] = P[ip] + area   
        if sweep>=relax: 
            pp = np.concatenate((pp, np.reshape(np.array(p[ip]), (1, 1))), axis=0)
            x = np.concatenate((x, np.reshape(np.array(m), (1, L*L))), axis=0)

Pi = Pi/sweeps
P = P/(sweeps*L*L)

def save_data(name, value):
    with h5py.File(r".\data\{}.h5".format(name),'w') as hf:
        hf.create_dataset("elem", data=value, compression="gzip", compression_opts=9)
        hf.close()

save_data(name="x", value=x)    
save_data(name="Pi", value=Pi)    
save_data(name="P", value=P)    
save_data(name="per", value=p)    
save_data(name="percolation", value=pp)  

# save_data(name="x_min", value=x)    
# save_data(name="Pi_min", value=Pi)    
# save_data(name="P_min", value=P)    
# save_data(name="per_min", value=p)    
# save_data(name="percolation_min", value=pp)       

end_time = time.time()
print("times = ", (end_time-start_time)/60)
