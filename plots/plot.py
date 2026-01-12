import h5py
import matplotlib.pyplot as plt
import os
directory_data = "/gpfs/home6/scur2334/MLOps_2026/data_files/mlops_2026_pcam_data/surfdrive"
xpath = os.path.join(directory_data, "camelyonpatch_level_2_split_train_x.h5")
ypath = os.path.join(directory_data, "camelyonpatch_level_2_split_train_y.h5")

with h5py.File(xpath, "r") as filex, h5py.File(ypath, "r") as filey:
    images = filex["x"][:50] 
    labels = filey["y"][:50].squeeze()

plt.figure(figsize=(15, 5))


plt.subplot(1, 3, 1)
plt.hist(labels)
plt.title("1. Labels (0 vs 1)")

plt.subplot(1, 3, 2)
plt.hist(images.flatten(), bins=20)
plt.title("2. Pixel Ranges (0-255)")


plt.subplot(1, 3, 3)
plt.imshow(images[0].astype('uint8'))
plt.title(f"3. Sample (Label: {labels[0]})")

plt.savefig("3plots.png")