
import matplotlib.pyplot as plt
import numpy as np

image_height = 28
original_dim = image_height ** 2
latent_dim = 200
seed = 12345

def dataset(name):
    import h5py
    data = h5py.File(r".\data\{}.h5".format(name),'r')
    data = data["elem"][:]
    return data

vae_loss = dataset(name="vae_loss")
print(vae_loss.shape) 
vae_reconstruction_loss = dataset(name="vae_reconstruction_loss")
print(vae_reconstruction_loss.shape) 
vae_kl_loss = dataset(name="vae_kl_loss")
print(vae_kl_loss.shape) 
cvae_loss = dataset(name="cvae_loss")
print(cvae_loss.shape) 
cvae_reconstruction_loss = dataset(name="cvae_reconstruction_loss")
print(cvae_reconstruction_loss.shape) 
cvae_kl_loss = dataset(name="cvae_kl_loss")
print(cvae_kl_loss.shape) 

x = np.arange(len(vae_reconstruction_loss))

fig = plt.figure(figsize=(14, 4))

plt.subplot(1,3,1) 
plt.plot(x, vae_loss, color="r", label='vae_loss')
plt.plot(x, cvae_loss, color="b", label='cvae_loss')
plt.xlabel('$\Pi$')
plt.ylabel('$pca$')
plt.ylim(0, 2)
plt.legend()
plt.grid(True, linestyle='--')

plt.subplot(1,3,2) 
plt.plot(x, vae_reconstruction_loss, color="r", label='vae_reconstruction_loss')
plt.plot(x, cvae_reconstruction_loss, color="b", label='cvae_reconstruction_loss')
plt.xlabel('$\Pi$')
plt.ylabel('$pca$')
plt.ylim(0, 2)
plt.legend()
plt.grid(True, linestyle='--')

plt.subplot(1,3,3) 
plt.plot(x, vae_kl_loss, color="r", label='vae_kl_loss')
plt.plot(x, cvae_kl_loss, color="b", label='cvae_kl_loss')
plt.xlabel('$\Pi$')
plt.ylim(0, 2)
plt.legend()
plt.grid(True, linestyle='--')

plt.show()