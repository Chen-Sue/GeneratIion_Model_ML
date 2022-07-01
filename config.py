import numpy as np

sweeps = 10**3 
image_height = 28
image_size = 28*28
p = np.around(np.arange(0, 1+0.005, 0.01), 2) # 0.593
print(p)
hyp = np.around(np.arange(0.01, 1-0.005, 0.01), 2)

seed = 12345
n_clusters = 2
n_components = 2

epochs = 10**3
learning_rate = 1e-4 
batch_size = 512 #256
split_rate = 0.8

num_classes = 1
l2_rate = 0 #1e-3
filter1 = 32
filter2 = 64
fc1 = 128
dropout_rate = 0.5

