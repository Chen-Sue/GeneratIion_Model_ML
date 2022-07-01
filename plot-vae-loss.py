import os
import matplotlib.pyplot as plt
import numpy as np

from utils import read_data

file_location = os.getcwd()
loss = read_data(file_location=file_location + r"\data", 
    name="vae-loss")
reconstruction_loss = read_data(file_location=file_location + r"\data", 
    name="vae-reconstruction_loss")
kl_loss = read_data(file_location=file_location + r"\data", 
    name="vae-kl_loss")

x = np.arange(len(loss))

fig = plt.figure(figsize=(14, 4))

plt.subplot(1,3,1) 
plt.plot(x, loss, color="dodgerblue", label='loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.ylim(-5, 505)
plt.legend()
plt.grid(True, linestyle='--')

plt.subplot(1,3,2) 
plt.plot(x, reconstruction_loss, color="dodgerblue", label='mae')
plt.xlabel('epoch')
plt.ylabel('reconstruction_loss')
plt.ylim(-5, 505)
plt.legend()
plt.grid(True, linestyle='--')

plt.subplot(1,3,3) 
plt.plot(x, kl_loss, color="dodgerblue", label='rmse')
plt.xlabel('epoch')
plt.ylabel('kl_loss')
plt.ylim(4.8, 8.2)
plt.legend()
plt.grid(True, linestyle='--')

plt.subplots_adjust(wspace=0.35)
plt.savefig(r".\figure\vae-loss.pdf")
plt.savefig(r".\figure\vae-loss.eps")
plt.show()