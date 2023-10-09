import numpy as np
import h5py
import argparse
import time
import sys

def main(EM_path,num_input_stimuli,num_probe_x_tiles,num_probe_y_tiles,time_steps,em_map_train,em_map_test):
    em_map = np.zeros((num_input_stimuli, 20, num_probe_x_tiles, num_probe_y_tiles))

    cema_trace_file = EM_path + 'fastem_sim_traces.h5'
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
            em_map[: , :, current_probe_x_tiles, current_probe_y_tiles] = tmp_trace[:num_input_stimuli, :20]


    print(np.shape(em_map))

    np.save(em_map_train + "em_map_train.npy", em_map[:750, :, :, :])
    np.save(em_map_test + "em_map_test.npy", em_map[750:, :, :, :])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--EM_path", type=str, default="../Electromagnetic Computation/",
                        help=" path to the simulated EM data ")
    parser.add_argument("--num_input_stimuli", type=int, default=1000,
                        help=" number of plaintexts ")
    parser.add_argument("--num_probe_x_tiles", type=int, default=48,
                        help=" number of point grid in x axial direction ")
    parser.add_argument("--num_probe_y_tiles", type=int, default=48,
                        help=" number of point grid in y axial direction ")
    parser.add_argument("--time_steps", type=int, default=20,
                        help=" simulation time for each EM trace ")
    parser.add_argument("--em_map_train", type=str, default="../GAN Model Training/",
                        help="  generate em map for GAN training ")
    parser.add_argument("--em_map_test", type=str, default="../GAN Model Training/",
                        help="  generate em current map for EM prediction ")

    args = parser.parse_args()
    EM_path = args.EM_path
    num_input_stimuli = args.num_input_stimuli
    num_probe_x_tiles = args.num_probe_x_tiles
    num_probe_y_tiles = args.num_probe_y_tiles
    time_steps = args.time_steps
    em_map_train = args.em_map_train
    em_map_test = args.em_map_test
    start_time = time.time()
    try:
        sys.exit(
            main(EM_path,num_input_stimuli,num_probe_x_tiles,num_probe_y_tiles,time_steps,em_map_train,em_map_test))
    except KeyboardInterrupt:
        sys.exit()
    finally:
        print("Running time:", time.time() - start_time, "seconds")
