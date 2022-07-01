import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time

from utils import read_data

start_time = time.time()
seed = 12345
L = 28

file_location = os.getcwd()
x = read_data(file_location=file_location + r"\data", 
    name="x_1000")
x_vae = read_data(file_location=file_location + r"\data", 
    name="vae-x_decoded")
x_cvae = read_data(file_location=file_location + r"\data", 
    name="cvae-x_decoded")
percolation = read_data(file_location=file_location + r"\data", 
    name="percolation_1000")

for i in np.arange(1000):
    x_vae[i::40] = np.where(x_vae[i::40]>=percolation[i], 1, x_vae[i::40])
    x_vae[i::40] = np.where(x_vae[i::40]<percolation[i], 0, x_vae[i::40])
    x_cvae[i::40] = np.where(x_cvae[i::40]>=percolation[i], 1, x_cvae[i::40])
    x_cvae[i::40] = np.where(x_cvae[i::40]<percolation[i], 0, x_cvae[i::40])
    
plt.figure(figsize=(5, 5))
ax = plt.subplot(3, 3, 1)
plt.ylabel("MC", fontsize=14)
plt.imshow(x[5].reshape(28, 28))
plt.gray()
ax = plt.gca()
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
ax = plt.subplot(3, 3, 2)
plt.imshow(x[19].reshape(28, 28))
plt.gray()
ax = plt.gca()
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
ax = plt.subplot(3, 3, 3)
plt.imshow(x[34].reshape(28, 28))
plt.gray()
ax = plt.gca()
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
ax = plt.subplot(3, 3, 4)
plt.ylabel("VAE", fontsize=14)
plt.imshow(x_vae[5].reshape(28, 28))
plt.gray()
ax = plt.gca()
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
ax = plt.subplot(3, 3, 5)
plt.imshow(x_vae[19].reshape(28, 28))
plt.gray()
ax = plt.gca()
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
ax = plt.subplot(3, 3, 6)
plt.imshow(x_vae[34].reshape(28, 28))
plt.gray()
ax = plt.gca()
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
ax = plt.subplot(3, 3, 7)
plt.xlabel("$p=0.46$", fontsize=14)
plt.ylabel("cVAE", fontsize=14)
plt.imshow(x_cvae[5].reshape(28, 28))
plt.gray()
ax = plt.gca()
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
ax = plt.subplot(3, 3, 8)
plt.xlabel("$p=0.60$", fontsize=14)
plt.imshow(x_cvae[19].reshape(28, 28))
plt.gray()
ax = plt.gca()
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
ax = plt.subplot(3, 3, 9)
plt.xlabel("$p=0.75$", fontsize=14)
plt.imshow(x_cvae[34].reshape(28, 28))
plt.gray()
ax = plt.gca()
ax.axes.xaxis.set_ticks([])
ax.axes.yaxis.set_ticks([])
plt.subplots_adjust(hspace=0.1, wspace=0.1)
plt.tight_layout()
plt.savefig(r".\figure\vae-cvae-image.eps")
plt.savefig(r".\figure\vae-cvae-image.pdf")
plt.show()

# n = 40
# plt.figure(figsize=(16, 4))
# for i in range(0, n, 4):
#     ax = plt.subplot(3, 10, int(i/4)+1)
#     plt.imshow(x[3960+i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#     # display reconstruction vae
#     ax = plt.subplot(3, 10, int(i/4)+10+1)
#     plt.imshow(x_vae[3960+i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#     # display reconstruction cvae
#     ax = plt.subplot(3, 10, int(i/4)+2*10+1)
#     plt.imshow(x_cvae[3960+i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.subplots_adjust(hspace=0.5)
# plt.savefig(r".\figure\vae-cvae-image.eps")
# plt.savefig(r".\figure\vae-cvae-image.pdf")
# plt.show()

end_time = time.time()
print("times = ", (end_time-start_time)/60)