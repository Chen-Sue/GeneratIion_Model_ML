
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import h5py
import time
from scipy.ndimage import measurements

start_time = time.time()

p = np.linspace(0.41, 0.8, 40)  # 0.593 
L = 28
sweeps = 10**3

Pi_origin = np.empty(shape=(0, len(p)))
Pi_vae = np.empty(shape=(0, len(p)))
Pi_cvae = np.empty(shape=(0, len(p)))
P_origin = np.empty(shape=(0, len(p)))
P_vae = np.empty(shape=(0, len(p)))
P_cvae = np.empty(shape=(0, len(p)))
Pi_origin_40 = np.zeros(len(p))
Pi_vae_40 = np.zeros(len(p))
Pi_cvae_40 = np.zeros(len(p))
P_origin_40 = np.zeros(len(p))
P_vae_40 = np.zeros(len(p))
P_cvae_40 = np.zeros(len(p))

def dataset(name):
    import h5py
    data = h5py.File(r".\data\{}.h5".format(name),'r')
    data = data["elem"][:]
    return data

x = dataset(name="X")
print(x.shape) 
x_vae = dataset(name="x_vae_decoded")
print(x_vae.shape) 
x_cvae = dataset(name="x_cvae_decoded")
print(x_cvae.shape)   

for sweep in np.arange(sweeps):
    print("sweep=", sweep, "Pi_origin.shape=", Pi_origin.shape)
    for ip in np.arange(len(p)):
        z = x[ip*sweeps+ip, :].reshape(L,L)
        z_vae = x_vae[ip*sweeps+ip, :].reshape(L,L)
        z_cvae = x_cvae[ip*sweeps+ip, :].reshape(L,L)
        m = z.astype(bool)
        m_vae = z_vae.astype(bool)
        m_cvae = z_cvae.astype(bool)
        lw, num = measurements.label(m) #  label the clusters
        lw_vae, num_vae = measurements.label(m_vae) #  label the clusters
        lw_cvae, num_cvae = measurements.label(m_cvae) #  label the clusters
        # check if the set of labels on the left side and 
        # the set of labels on the right side have any intersection  
        area = measurements.sum(m, lw, index=np.arange(lw.max()+1))
        area_vae = measurements.sum(m_vae, lw_vae, index=np.arange(lw_vae.max()+1))
        area_cvae = measurements.sum(m_cvae, lw_cvae, index=np.arange(lw_cvae.max()+1))
        # Remove spanning cluster by setting its area to zero
        perc_x = intersect1d(lw[0,:], lw[-1,:]) 
        perc_x_vae = intersect1d(lw_vae[0,:], lw_vae[-1,:]) 
        perc_x_cvae = intersect1d(lw_cvae[0,:], lw_cvae[-1,:]) 
        # If the length of the set of intersections is larger than zero, 
        # there is at least one percolating cluster, find the intersection of two arrays
        perc = perc_x[where(perc_x>0)]
        perc_vae = perc_x_vae[where(perc_x_vae>0)]
        perc_cvae = perc_x_cvae[where(perc_x_cvae>0)]
        if (len(perc)>0):
            Pi_origin_40[ip] = Pi_origin_40[ip] + 1
            P_origin_40[ip] = P_origin_40[ip] + area[perc[0]]
        if (len(perc_vae)>0):
            Pi_vae_40[ip] = Pi_vae_40[ip] + 1
            P_vae_40[ip] = P_vae_40[ip] + area_vae[perc_vae[0]]
        if (len(perc_cvae)>0):
            Pi_cvae_40[ip] = Pi_cvae_40[ip] + 1
            P_cvae_40[ip] = P_cvae_40[ip] + area_cvae[perc_cvae[0]]

Pi_origin = np.concatenate((Pi_origin, (Pi_origin_40/sweeps).reshape(-1, len(p))), axis=0)
Pi_vae = np.concatenate((Pi_vae, (Pi_vae_40/sweeps).reshape(-1, len(p))), axis=0)
Pi_cvae = np.concatenate((Pi_cvae, (Pi_cvae_40/sweeps).reshape(-1, len(p))), axis=0)
P_origin = np.concatenate((P_origin, (P_origin_40.reshape(-1, len(p)))/(sweeps*L*L)), axis=0)
P_vae = np.concatenate((P_vae, (P_vae_40.reshape(-1, len(p)))/(sweeps*L*L)), axis=0)
P_cvae = np.concatenate((P_cvae, (P_cvae_40.reshape(-1, len(p)))/(sweeps*L*L)), axis=0)

Pi_origin, P_origin = Pi_origin.reshape(-1, 1), P_origin.reshape(-1, 1)
Pi_vae, P_vae = Pi_vae.reshape(-1, 1), P_vae.reshape(-1, 1)
Pi_cvae, P_cvae = Pi_cvae.reshape(-1, 1), P_cvae.reshape(-1, 1)
print(Pi_origin.shape, P_origin.shape, Pi_vae.shape, P_vae.shape, Pi_cvae.shape, P_cvae.shape)

def save_data(name, value):
    with h5py.File(r".\data\{}.h5".format(name),'w') as hf:
        hf.create_dataset("elem", data=value, compression="gzip", compression_opts=9)
        hf.close()

save_data(name="Pi_origin", value=Pi_origin)    
save_data(name="P_origin", value=P_origin)    
save_data(name="Pi_vae", value=Pi_vae)    
save_data(name="P_vae", value=P_vae)    
save_data(name="Pi_cvae", value=Pi_cvae)    
save_data(name="P_cvae", value=P_cvae)    

end_time = time.time()
print("times = ", (end_time-start_time)/60)
