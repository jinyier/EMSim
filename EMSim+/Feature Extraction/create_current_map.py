import argparse
import re
import sys
import numpy as np
import time
import math
import matplotlib.pyplot as plt


def get_cell_info(def_path):
    '''
    mathch the cell infor from parasistic file and def file
    :param def_file: design.def
    :return: hierarchical instance
    '''
    # open and read the def file
    def_file = open(def_path, 'r')

    # create an empty list
    instance_hierarchy = []
    instance_type = []
    instance_x0 = []
    instance_y0 = []
    # initialize the counter for instance
    instance_count = 0
    start_component = False
    for line in def_file:
        # the end of the component lines to check
        if re.match(r'END COMPONENTS', line):
            start_component = False
        if start_component:
            if 'FILL' in line:
                continue
            split_line = line.split(" ")
            # check if the data length > 7
            if len(split_line) > 7 and split_line[5] in ['TIMING', 'DIST']:
                instance_hierarchy.append(split_line[1])
                instance_type.append(split_line[2])
                instance_x0.append(float(split_line[9]) / 2000)
                instance_y0.append(float(split_line[10]) / 2000)
                instance_count += 1
            elif len(split_line) > 7:
                instance_hierarchy.append(split_line[1])
                instance_type.append(split_line[2])
                instance_x0.append(float(split_line[6]) / 2000)
                instance_y0.append(float(split_line[7]) / 2000)
                instance_count += 1
            else:
                continue
        # the beginning of the component lines to check
        elif re.match(r'COMPONENTS [\d]+', line):
            start_component = True
    print('Total number of logic cells:', instance_count)

    instance_y0 = np.asarray(instance_y0)
    instance_x0 = np.asarray(instance_x0)
    instance_hierarchy = np.asarray(instance_hierarchy)

    # the index of sorted data, from small to large
    sort_index_vertical = np.argsort(instance_y0)
    instance_y0_reorder = instance_y0[sort_index_vertical]
    instance_x0_reorder = instance_x0[sort_index_vertical]
    instance_hierarchy_reorder = instance_hierarchy[sort_index_vertical]
    # the unique data without repetition
    sort_unique = np.unique(instance_y0_reorder)

    # match the cell name and the hierarchy instance name
    end_num = 0
    for current_num in range(len(sort_unique)):
        start_num = end_num
        end_num = start_num + np.count_nonzero(instance_y0_reorder == sort_unique[current_num])
        tmp_x0 = instance_x0_reorder[start_num: end_num]
        tmp_instance_hierarchy = instance_hierarchy_reorder[start_num: end_num]
        sort_index_horizontal = np.argsort(tmp_x0)
        instance_x0_reorder[start_num: end_num] = tmp_x0[sort_index_horizontal]
        instance_hierarchy_reorder[start_num: end_num] = tmp_instance_hierarchy[sort_index_horizontal]

    return instance_hierarchy_reorder, instance_x0_reorder, instance_y0_reorder


def current_map_fature(current_trace, layout_min_x, layout_min_y, target_area_x, target_area_y, num_probe_x_tiles,
                       num_probe_y_tiles, num_input_stimuli, start_points, sample_points, instance_x0, instance_y0):
    '''
    extract the current map feature
    :return: hierarchical instance
    '''
    end_points = int(start_points + sample_points)
    current_map = np.zeros((num_input_stimuli, sample_points, num_probe_x_tiles, num_probe_y_tiles))
    interval_x = target_area_x / num_probe_x_tiles
    interval_y = target_area_y / num_probe_y_tiles
    for current_cell in range(np.shape(current_trace)[0]):
        tmp_x0 = int((instance_x0[current_cell] - layout_min_x) // interval_x)
        tmp_y0 = int((instance_y0[current_cell] - layout_min_y) // interval_y)
        if 0 < tmp_x0 < num_probe_x_tiles and 0 < tmp_y0 < num_probe_y_tiles:
            current_map[:, :, tmp_x0, tmp_y0] += current_trace[current_cell, : num_input_stimuli, start_points: end_points]

    return current_map


def main(def_path, current_path, layout_min_x, layout_min_y, target_area_x, target_area_y, num_probe_x_tiles,
         num_probe_y_tiles, num_input_stimuli, start_points, sample_points, current_map_train, current_map_test):
    instance_hierarchy, instance_x0, instance_y0 = get_cell_info(def_path)
    #num_current_sets = 1  #3
    #current_map = np.zeros((num_input_stimuli * num_current_sets, sample_points, num_probe_x_tiles, num_probe_y_tiles))
    current_map = np.zeros((num_input_stimuli, sample_points, num_probe_x_tiles, num_probe_y_tiles))
    current_trace = np.load(current_path + 'current_trace.npy')
    print('Current Shape', np.shape(current_trace))
    current_map = current_map_fature(current_trace, layout_min_x, layout_min_y, target_area_x, target_area_y,
                                         num_probe_x_tiles, num_probe_y_tiles, num_input_stimuli, start_points,
                                         sample_points, instance_x0, instance_y0)
    #current_map[current_idx * num_input_stimuli: (current_idx + 1) * num_input_stimuli, :, :, :] = tmp_current_map
    # for current_idx in range(num_current_sets):
    #     current_trace = np.load(current_path + str(current_idx+1) + '/Current Analysis/current_trace.npy')
    #     print('Current Index', current_idx, 'Current Shape', np.shape(current_trace))
    #     tmp_current_map = current_map_fature(current_trace, layout_min_x, layout_min_y, target_area_x, target_area_y,
    #                                      num_probe_x_tiles, num_probe_y_tiles, num_input_stimuli, start_points,
    #                                      sample_points, instance_x0, instance_y0)
    #     current_map[current_idx*num_input_stimuli: (current_idx+1)*num_input_stimuli, :, :, :] = tmp_current_map
        #plt.plot(np.sum(current_trace, axis=0)[0, 18: 18 + 20])  # 20: 20 + 3
        #plt.show()
        #print(np.shape(current_map))
    print(np.shape(current_map))
    np.save(current_map_train, current_map[:750, :, :, :])
    np.save(current_map_test, current_map[750:, :, :, :])
    #np.save("current_map_train.npy", current_map[:, :8, :, :])
    #print(current_map[:, :8, :, :].shape)
    #np.save("current_map_test.npy", current_map[750:, 3:6, :, :])

    for map_idx in range(20):
        plt.imshow(current_map[0, map_idx, :, :], vmax=np.max(current_map[0, :, :, :]))
        plt.colorbar()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--def_path", type=str, default="../Current Analysis/AES_extension.def",
                        help="Path to the def file, should end in .def")
    parser.add_argument("--current_path", type=str, default="../Current Analysis/",
                        help="Path to the simulated logic cell currents")
    parser.add_argument("--num_input_stimuli", type=int, default=1000,
                        help="Number of required plaintexts for em computation")
    parser.add_argument("--target_area_x", type=float, default=960,
                        help="Target simulated area in x axial direction")
    parser.add_argument("--target_area_y", type=float, default=960,
                        help="Target simulated area in y axial direction")
    parser.add_argument("--num_probe_x_tiles", type=int, default=48,
                        help="Number of point grid in x axial direction")
    parser.add_argument("--num_probe_y_tiles", type=int, default=48,
                        help="Number of point grid in y axial direction")
    parser.add_argument("--layout_min_x", type=float, default=0,
                        help="Reference coordinate in x axial direction")
    parser.add_argument("--layout_min_y", type=float, default=0,
                        help="Reference coordinate in y axial direction")
    parser.add_argument("--start_points", type=int, default=0,  # 20
                        help="Start of sample point for each current trace")
    parser.add_argument("--sample_points", type=int, default=20,  # 3
                        help="End of sample point for each current trace")
    parser.add_argument("--current_map_train", type=str, default="../GAN Model Training/current_map_train.npy",  # 3
                        help="generate cell current map for GAN training")
    parser.add_argument("--current_map_test", type=str, default="../EM Predition/current_map_test.npy",  # 3
                        help="generate cell current map for EM prediction")

    args = parser.parse_args()
    def_path = args.def_path
    current_path = args.current_path
    num_input_stimuli = args.num_input_stimuli
    target_area_x = args.target_area_x
    target_area_y = args.target_area_y
    num_probe_x_tiles = args.num_probe_x_tiles
    num_probe_y_tiles = args.num_probe_y_tiles
    layout_min_x = args.layout_min_x
    layout_min_y = args.layout_min_y
    start_points = args.start_points
    sample_points = args.sample_points
    current_map_train = args.current_map_train
    current_map_test = args.current_map_test

    start_time = time.time()
    try:
        sys.exit(
            main(def_path, current_path, layout_min_x, layout_min_y, target_area_x, target_area_y, num_probe_x_tiles,
                 num_probe_y_tiles, num_input_stimuli, start_points, sample_points, current_map_train, current_map_test))
    except KeyboardInterrupt:
        sys.exit()
    finally:
        print("Running time:", time.time() - start_time, "seconds")
