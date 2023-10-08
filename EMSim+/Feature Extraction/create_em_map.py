import numpy as np
import h5py
import matplotlib.pyplot as plt

num_probe_x_tiles = 48
num_probe_y_tiles = 48
num_input_stimuli = 1000

em_map = np.zeros((num_input_stimuli, 20, num_probe_x_tiles, num_probe_y_tiles))
cema_trace_file = '../Electromagnetic Computation/fastem_sim_traces.h5'


cema_trace_file = '../Electromagnetic Computation/fastem_sim_traces.h5'
f = h5py.File(cema_trace_file, "r")
print('--iterms: ', len(f.keys()), f.keys())
x_loc = list(f.keys())
n = len(x_loc)
for i in range(n-1):
    for j in range(n-1-i):
        if float(x_loc[j]) > float(x_loc[j+1]):
            x_loc[j], x_loc[j+1] = x_loc[j+1], x_loc[j]
x_loc1 = x_loc
print('--iterms: ', len(f[x_loc[0]].keys()), f[x_loc[0]].keys())
y_loc = list(f[x_loc[0]].keys())
m = len(y_loc)
for i in range(m-1):
    for j in range(m-1-i):
        if float(y_loc[j]) > float(y_loc[j+1]):
            y_loc[j], y_loc[j+1] = y_loc[j+1], y_loc[j]
y_loc1 = y_loc
print(x_loc1)
print(y_loc1)

for current_probe_x_tiles in range(num_probe_x_tiles):
    for current_probe_y_tiles in range(num_probe_y_tiles):
        tmp_trace = f[x_loc1[current_probe_x_tiles]][y_loc1[current_probe_y_tiles]]["traces"][:]
        em_map[: , :, current_probe_x_tiles, current_probe_y_tiles] = tmp_trace[:num_input_stimuli, :20]  # 2:2+3

#

# np.save("em_map_100um.npy", em_map)
print(np.shape(em_map))

np.save("em_map_train.npy", em_map[:750, :, :, :])
np.save("em_map_test.npy", em_map[750:, :, :, :])
