import os
import matplotlib.pyplot as plt
import numpy as np
from brokenaxes import brokenaxes

from utils import read_data

file_location = os.getcwd()
vae_loss = read_data(file_location=file_location + r"\data", 
    name="vae-loss")
vae_reconstruction_loss = read_data(file_location=file_location + r"\data", 
    name="vae-reconstruction_loss")
vae_kl_loss = read_data(file_location=file_location + r"\data", 
    name="vae-kl_loss")
cvae_loss = read_data(file_location=file_location + r"\data", 
    name="cvae-loss")
cvae_reconstruction_loss = read_data(file_location=file_location + r"\data", 
    name="cvae-reconstruction_loss")
cvae_kl_loss = read_data(file_location=file_location + r"\data", 
    name="cvae-kl_loss")

x = np.arange(len(vae_loss))




# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
# # fig.subplots_adjust(hspace=0.05)  # adjust space between axes

# ax1.plot(vae_loss)
# ax2.plot(vae_loss)
# ax1.plot(cvae_loss)
# ax2.plot(cvae_loss)

# ax1.set_ylim(378, 778)  # outliers only
# ax2.set_ylim(-28, 108)  # most of the data

# ax1.spines['bottom'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax1.xaxis.tick_top()
# ax1.tick_params(labeltop=False)  # don't put tick labels at the top
# ax2.xaxis.tick_bottom()

# # Now, let's turn towards the cut-out slanted lines.
# # We create line objects in axes coordinates, in which (0,0), (0,1),
# # (1,0), and (1,1) are the four corners of the axes.
# # The slanted lines themselves are markers at those locations, such that the
# # lines keep their angle and position, independent of the axes size or scale
# # Finally, we need to disable clipping.

# d = 0.015  # proportion of vertical to horizontal extent of the slanted line
# # kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
# #               linestyle="none", color='k', mec='k', mew=1, clip_on=True)

# d = .85
# kwargs = dict(marker=[(-1, -d), (1, d)], markersize=20,
#               linestyle='none', color='r', mec='r', mew=1, clip_on=False)
# # ax1.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
# # ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

# ax1.plot(x, vae_loss, transform=ax1.transAxes, **kwargs)
# ax2.plot(x, vae_loss, transform=ax2.transAxes, **kwargs)
# ax1.plot(x, cvae_loss, transform=ax1.transAxes, **kwargs)
# ax2.plot(x, cvae_loss, transform=ax2.transAxes, **kwargs)

# plt.show()

fig = plt.figure(figsize=(14, 4.5))
plt.subplot(1,3,1) 
plt.plot(x, vae_loss, color="dodgerblue", label='VAE', linewidth=2)
plt.plot(x, cvae_loss, color="indianred", label='cVAE', linewidth=2)
plt.xlabel('epoch', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.xticks(size=12)
plt.yticks(size=12)
plt.ylim(-20, 580)
plt.legend()
plt.grid(True, linestyle='--', linewidth=1.5)
plt.legend(loc="upper right")
plt.text(220, 510, '(a)', fontsize=14)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)

plt.subplot(1,3,2) 
plt.plot(x, vae_reconstruction_loss, color="dodgerblue", 
    label='VAE', linewidth=2)
plt.plot(x, cvae_reconstruction_loss, color="indianred", 
    label='cVAE', linewidth=2)
plt.xlabel('epoch', fontsize=14)
plt.ylabel('Binary Cross-Entropy', fontsize=14)
plt.xticks(size=12)
plt.yticks(size=12)
plt.ylim(-20, 580)
plt.legend()
plt.grid(True, linestyle='--', linewidth=1.5)
plt.legend(loc="upper right")
plt.text(220, 510, '(b)', fontsize=14)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)

plt.subplot(1,3,3) 
plt.plot(x, vae_kl_loss, color="dodgerblue", label='VAE', linewidth=2)
plt.plot(x, cvae_kl_loss, color="indianred", label='cVAE', linewidth=2)
plt.xlabel('epoch', fontsize=14)
plt.ylabel('KL Divergence', fontsize=14)
plt.xticks(size=12)
plt.yticks(size=12)
plt.ylim(-0.5, 20.5)
plt.legend()
plt.grid(True, linestyle='--', linewidth=1.5)
plt.legend(loc="upper right")
plt.text(220, 18, '(c)', fontsize=14)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)

plt.subplots_adjust(wspace=0.35)
plt.savefig(r".\figure\vae-cvae-epoch-loss-pi-p.pdf")
plt.savefig(r".\figure\vae-cvae-epoch-loss-pi-p.eps")
plt.show()