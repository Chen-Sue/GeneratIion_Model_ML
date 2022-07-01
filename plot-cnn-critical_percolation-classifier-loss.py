
import os
import matplotlib.pyplot as plt
import numpy as np

from utils import read_data

file_location = os.getcwd()
loss_55 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pc-loss-55")
ba_55 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pc-binary_accuracy-55")
val_loss_55 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pc-val_loss-55")
val_ba_55 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pc-binary_accuracy-55")  

loss_56 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pc-loss-56")
ba_56 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pc-binary_accuracy-56")
val_loss_56 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pc-val_loss-56")
val_ba_56 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pc-binary_accuracy-56")  

loss_57 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pc-loss-57")
ba_57 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pc-binary_accuracy-57")
val_loss_57 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pc-val_loss-57")
val_ba_57 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pc-binary_accuracy-57")  

loss_58 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pc-loss-58")
ba_58 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pc-binary_accuracy-58")
val_loss_58 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pc-val_loss-58")
val_ba_58 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pc-binary_accuracy-58")  

loss_59 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pc-loss-59")
ba_59 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pc-binary_accuracy-59")
val_loss_59 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pc-val_loss-59")
val_ba_59 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pc-binary_accuracy-59")  

loss_60 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pc-loss-60")
ba_60 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pc-binary_accuracy-60")
val_loss_60 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pc-val_loss-60")
val_ba_60 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pc-binary_accuracy-60")  

loss_61 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pc-loss-61")
ba_61 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pc-binary_accuracy-61")
val_loss_61 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pc-val_loss-61")
val_ba_61 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pc-val_binary_accuracy-61")  

loss_62 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pc-loss-62")
ba_62 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pc-binary_accuracy-62")
val_loss_62 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pc-val_loss-62")
val_ba_62 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pc-val_binary_accuracy-62")  

loss_63 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pc-loss-63")
ba_63 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pc-binary_accuracy-63")
val_loss_63 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pc-val_loss-63")
val_ba_63 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pc-val_binary_accuracy-63")  

loss_64 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pc-loss-64")
ba_64 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pc-binary_accuracy-64")
val_loss_64 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pc-val_loss-64")
val_ba_64 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pc-val_binary_accuracy-64")  

loss_65 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pc-loss-65")
ba_65 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pc-binary_accuracy-65")
val_loss_65 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pc-val_loss-65")
val_ba_65 = read_data(file_location=file_location + r"\data", 
    name="train-cnn-pc-val_binary_accuracy-65")  

x = np.arange(len(loss_55))

fig = plt.figure(figsize=(14, 8))

plt.subplot(1,2,1) 
plt.plot(x, loss_55, color="purple", label='Training set with the preset value of 0.55', linewidth=2)
plt.plot(x, val_loss_55, color="green", label='Testing set with the preset value of 0.55', linestyle=":", linewidth=2)
plt.plot(x, loss_56, color="blue", label='Training set with the preset value of 0.56', linewidth=2)
plt.plot(x, val_loss_56, color="pink", label='Testing set with the preset value of 0.56', linestyle=":", linewidth=2)
plt.plot(x, loss_57, color="brown", label='Training set with the preset value of 0.57', linewidth=2)
plt.plot(x, val_loss_57, color="navy", label='Testing set with the preset value of 0.57', linestyle=":", linewidth=2)
plt.plot(x, loss_58, color="teal", label='Training set with the preset value of 0.58', linewidth=2)
plt.plot(x, val_loss_58, color="orange", label='Testing set with the preset value of 0.58', linestyle=":", linewidth=2)
plt.plot(x, loss_59, color="purple", label='Training set with the preset value of 0.59', linewidth=2)
plt.plot(x, val_loss_59, color="tan", label='Testing set with the preset value of 0.59', linestyle=":", linewidth=2)
plt.plot(x, loss_60, color="aqua", label='Training set with the preset value of 0.60', linewidth=2)
plt.plot(x, val_loss_60, color="goldenrod", label='Testing set with the preset value of 0.60', linestyle=":", linewidth=2)
plt.plot(x, loss_61, color="beige", label='Training set with the preset value of 0.61', linewidth=2)
plt.plot(x, val_loss_61, color="olive", label='Testing set with the preset value of 0.61', linestyle=":", linewidth=2)
plt.plot(x, loss_62, color="blueviolet", label='Training set with the preset value of 0.62', linewidth=2)
plt.plot(x, val_loss_62, color="yellowgreen", label='Testing set with the preset value of 0.62', linestyle=":", linewidth=2)
plt.plot(x, loss_63, color="sienna", label='Training set with the preset value of 0.63', linewidth=2)
plt.plot(x, val_loss_63, color="lightgreen", label='Testing set with the preset value of 0.63', linestyle=":", linewidth=2)
plt.plot(x, loss_64, color="lime", label='Training set with the preset value of 0.64', linewidth=2)
plt.plot(x, val_loss_64, color="c", label='Testing set with the preset value of 0.64', linestyle=":", linewidth=2)
plt.plot(x, loss_65, color="seagreen", label='Training set with the preset value of 0.65', linewidth=2)
plt.plot(x, val_loss_65, color="dodgerblue", label='Testing set with the preset value of 0.65', linestyle=":", linewidth=2)
plt.xlabel('epoch', fontsize=14)
plt.ylabel('Binary Cross-Entropy', fontsize=14)
plt.ylim(-0.01, 0.81)
plt.xticks(size=12)
plt.yticks(size=12)
plt.legend(loc="upper right")
plt.grid(True, linestyle='--', linewidth=1.5)
plt.text(50, 0.76, '(a)', fontsize=14)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)

plt.subplot(1,2,2) 
plt.plot(x, ba_55, color="indianred", label='Training set with the preset value of 0.55', linewidth=2)
plt.plot(x, val_ba_55, color="indianred", label='Testing set with the preset value of 0.55', linestyle=":", linewidth=2)
plt.plot(x, ba_56, color="sienna", label='Training set with the preset value of 0.56', linewidth=2)
plt.plot(x, val_ba_56, color="sienna", label='Testing set with the preset value of 0.56', linestyle=":", linewidth=2)
plt.plot(x, ba_57, color="sandybrown", label='Training set with the preset value of 0.57', linewidth=2)
plt.plot(x, val_ba_57, color="sandybrown", label='Testing set with the preset value of 0.57', linestyle=":", linewidth=2)
plt.plot(x, ba_58, color="tan", label='Training set with the preset value of 0.58', linewidth=2)
plt.plot(x, val_ba_58, color="tan", label='Testing set with the preset value of 0.58', linestyle=":", linewidth=2)
plt.plot(x, ba_59, color="wheat", label='Training set with the preset value of 0.59', linewidth=2)
plt.plot(x, val_ba_59, color="wheat", label='Testing set with the preset value of 0.59', linestyle=":", linewidth=2)
plt.plot(x, ba_60, color="goldenrod", label='Training set with the preset value of 0.60', linewidth=2)
plt.plot(x, val_ba_60, color="goldenrod", label='Testing set with the preset value of 0.60', linestyle=":", linewidth=2)
plt.plot(x, ba_61, color="olive", label='Training set with the preset value of 0.61', linewidth=2)
plt.plot(x, val_ba_61, color="olive", label='Testing set with the preset value of 0.61', linestyle=":", linewidth=2)
plt.plot(x, ba_62, color="yellowgreen", label='Training set with the preset value of 0.62', linewidth=2)
plt.plot(x, val_ba_62, color="yellowgreen", label='Testing set with the preset value of 0.62', linestyle=":", linewidth=2)
plt.plot(x, ba_63, color="grey", label='Training set with the preset value of 0.63', linewidth=2)
plt.plot(x, val_ba_63, color="grey", label='Testing set with the preset value of 0.63', linestyle=":", linewidth=2)
plt.plot(x, ba_64, color="c", label='Training set with the preset value of 0.64', linewidth=2)
plt.plot(x, val_ba_64, color="c", label='Testing set with the preset value of 0.64', linestyle=":", linewidth=2)
plt.plot(x, ba_65, color="dodgerblue", label='Training set with the preset value of 0.65', linewidth=2)
plt.plot(x, val_ba_65, color="dodgerblue", label='Testing set with the preset value of 0.65', linestyle=":", linewidth=2)
plt.xlabel('epoch', fontsize=14)
plt.ylabel('Binary Accuracy', fontsize=14)
plt.ylim(0.795, 1.005)
plt.xticks(size=12)
plt.yticks(size=12)
plt.legend(loc="lower right")
plt.grid(True, linestyle='--', linewidth=1.5)
plt.text(50, 0.99, '(b)', fontsize=14)
ax = plt.gca()
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)

plt.subplots_adjust(wspace=0.35)
plt.savefig(r".\figure\cnn-epoch-loss-pc.pdf")
plt.savefig(r".\figure\cnn-epoch-loss-pc.eps")
plt.show()