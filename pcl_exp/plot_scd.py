import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = []
data_file = "/home/kilox/kilox_ws/sc000070.scd"
with open(data_file, "r") as f:
    for line in f.readlines():
        line_list = line.strip().split(" ")
        values = []
        for num in line_list:
            values.append(float(num))

        data.append(values)
# print(data)

a = np.array(data)
print(a.shape)
plt.imshow(a)

plt.show()


