import numpy as np
import sys
import argparse
import matplotlib.pyplot as plt
import time


def clean_metal_paths_em_data(metal_layers):
    """
    Concatenates the EM relevant data from Step 2/Prep_For_Sim.py into a single list. Assigns each useful
    type of value (column) of the metal layer data to its own component of the instance. Has 7 outputs, each
    a vector where each element relates to a single wire
    :return: True
    """
    # init array of null size
    # must be numpy, as we will be reading in from text file
    metal_em_info = np.zeros((0, 6), dtype=np.float32)
    # load in metal path em info
    print("Loading metal layer data...")
    for metal in metal_layers:
        metal_em_info = np.concatenate(
            (metal_em_info, np.genfromtxt('./' + metal + "_em.txt", delimiter=',', dtype=np.float32)))
    print("Finished loading metal layer data!")
    print("EM metal layers shape:", metal_em_info.shape)

    # digest metal path info
    width = metal_em_info[:, 0] * 1000000
    x0 = metal_em_info[:, 2]
    y0 = metal_em_info[:, 3]
    x1 = metal_em_info[:, 4]
    y1 = metal_em_info[:, 5]
    # signed vector from x0 (y0) to x1 (y1)
    x = x1 - x0
    y = y1 - y0
    # each is num_wires x 1
    wire_widths, wire_x0, wire_x1, wire_y0, wire_y1, wire_x, wire_y = \
        width, x0, x1, y0, y1, x, y
    return wire_widths, wire_x0, wire_x1, wire_y0, wire_y1, wire_x, wire_y


def calculate_wire_distances_for_field(wire_widths, wire_x0, wire_x1, wire_y0, wire_y1, wire_x, wire_y):
    """
    Calculates the distances used in the the field function all at once, so matrix operations
    with them are possible. Assigns these as a component of the instance. Should be a 7 by the number
    of wires matrix
    :return: True
    """
    # start with everything zero
    # columns are [x_dir, y_dir, z_dir, p_x0, p_x1, p_y0, py_1]
    # rows are for each wire
    print("Calculating static wire distance matrix...")
    # these calculations can only be done in numpy
    out = np.zeros((wire_widths.shape[0], 7), dtype=np.float32)

    # set values as desired in rows where wire widths are 0
    vertical = np.asarray(wire_x == 0).nonzero()
    # dir_y in column y_dir
    out[vertical, 1] = np.sign(wire_y[vertical])
    # x0_in - W_in / 2 in column p_x0
    out[vertical, 3] = wire_x0[vertical] - wire_widths[vertical] / 2
    # x0_in + W_in / 2 in column p_x1
    out[vertical, 4] = wire_x0[vertical] + wire_widths[vertical] / 2
    # y0_in in column p_y0
    out[vertical, 5] = wire_y0[vertical]
    # y1_in in column p_y1
    out[vertical, 6] = wire_y1[vertical]

    # set values as desired where wired lengths are 0
    # dir_x
    horizontal = np.asarray(wire_y == 0).nonzero()
    out[horizontal, 0] = np.sign(wire_x[horizontal])
    # y0_in - W_in / 2 in column p_y0
    out[horizontal, 5] = wire_y0[horizontal] - wire_widths[horizontal] / 2
    # y0_in + W_in / 2 in column p_y1
    out[horizontal, 6] = wire_y0[horizontal] + wire_widths[horizontal] / 2
    # x0_in in column p_x0
    out[horizontal, 3] = wire_x0[horizontal]
    # x1_in in column p_x1
    out[horizontal, 4] = wire_x1[horizontal]

    print("Finished calculating static wire distance matrix!")
    print("Static wire distance matrix shape:", np.shape(out))
    return out


# def grid_map_feature(out, layout_min_x, layout_min_y, target_area_x, target_area_y, num_probe_x_tiles,
# num_probe_y_tiles): """ extract the grid map for each design :return: grid map """ print(np.min(out[:, 3]),
# np.max(out[:, 4])) print(np.min(out[:, 5]), np.max(out[:, 6])) grid_map = np.zeros((num_probe_x_tiles,
# num_probe_y_tiles)) interval_x = target_area_x / num_probe_x_tiles interval_y = target_area_y / num_probe_y_tiles
#
#     for current_grid in range(np.shape(out)[0]):
#         tmp_x0 = int((out[current_grid, 3] - layout_min_x) // interval_x)
#         tmp_x1 = int((out[current_grid, 4] - layout_min_x) // interval_x)
#         if tmp_x0 > tmp_x1:
#             tmp_x = np.arange(tmp_x1, tmp_x0 + 1, 1)
#         else:
#             tmp_x = np.arange(tmp_x0, tmp_x1 + 1, 1)
#         tmp_y0 = int((out[current_grid, 5] - layout_min_y) // interval_y)
#         tmp_y1 = int((out[current_grid, 6] - layout_min_y) // interval_y)
#         if tmp_y0 > tmp_y1:
#             tmp_y = np.arange(tmp_y1, tmp_y0 + 1, 1)
#         else:
#             tmp_y = np.arange(tmp_y0, tmp_y1 + 1, 1)
#
# for x_tile in range(np.shape(tmp_x)[0]): for y_tile in range(np.shape(tmp_y)[0]): if 0 <= tmp_x[x_tile] <
# num_probe_x_tiles and 0 <= tmp_y[y_tile] < num_probe_y_tiles: if grid_map[tmp_x[x_tile], tmp_y[y_tile]] == 0:
# grid_map[tmp_x[x_tile], tmp_y[y_tile]] = 1 # grid_map[tmp_x[x_tile], tmp_y[y_tile]] = (float(out[current_grid,
# 7]) + grid_map[tmp_x[x_tile], tmp_y[y_tile]])/2 # if float(out[current_grid, 7]) > grid_map[tmp_x[x_tile],
# tmp_y[y_tile]]: #     grid_map[tmp_x[x_tile], tmp_y[y_tile]] = float(out[current_grid, 7]) # grid_map[tmp_x[
# x_tile], tmp_y[y_tile]] += float(out[current_grid, 7]) return grid_map


def grid_distance_feature(out, layout_min_x, layout_min_y, target_area_x, target_area_y, num_probe_x_tiles, num_probe_y_tiles):
    """
    extract the grid map for each design
    :return: grid map
    """

    power_wire_attribute = []
    for metal in metal_layers:
        file = open('./' + metal + "_probe.txt", 'r')
        power_wire_attribute += file.readlines()

    grid_distance = np.zeros((num_probe_x_tiles, num_probe_y_tiles))
    interval_x = target_area_x / num_probe_x_tiles
    interval_y = target_area_y / num_probe_y_tiles
    VDD = np.array([[(4.3905 - layout_min_x) // interval_x, (457.311 - layout_min_y) // interval_y], [(271.8705 - layout_min_x) // interval_x, (876.671 - layout_min_y) // interval_y]])
    VSS = np.array([[(-5.4395 - layout_min_x) // interval_x, (586.351 - layout_min_y) // interval_y], [(396.2905 - layout_min_x) // interval_x, (878.111 - layout_min_y) // interval_y]])

    for current_grid in range(np.shape(out)[0]):

        if 'VDD' in power_wire_attribute[current_grid]:
            power_pad = VDD
        else:
            power_pad = VSS

        tmp_x0 = int((out[current_grid, 3] - layout_min_x) // interval_x)
        tmp_x1 = int((out[current_grid, 4] - layout_min_x) // interval_x)
        if tmp_x0 > tmp_x1:
            tmp_x = np.arange(tmp_x1, tmp_x0 + 1, 1)
        else:
            tmp_x = np.arange(tmp_x0, tmp_x1 + 1, 1)
        tmp_y0 = int((out[current_grid, 5] - layout_min_y) // interval_y)
        tmp_y1 = int((out[current_grid, 6] - layout_min_y) // interval_y)
        if tmp_y0 > tmp_y1:
            tmp_y = np.arange(tmp_y1, tmp_y0 + 1, 1)
        else:
            tmp_y = np.arange(tmp_y0, tmp_y1 + 1, 1)

        for x_tile in range(np.shape(tmp_x)[0]):
            for y_tile in range(np.shape(tmp_y)[0]):
                if 0 <= tmp_x[x_tile] < num_probe_x_tiles and 0 <= tmp_y[y_tile] < num_probe_y_tiles:
                    # Compute Euclidean Distance tmp_distance0 = np.sqrt((tmp_x[x_tile] - power_pad[0, 0])**2 + (
                    # tmp_y[y_tile] - power_pad[0, 1])**2) tmp_distance1 = np.sqrt((tmp_x[x_tile] - power_pad[1,
                    # 0]) ** 2 + (tmp_y[y_tile] - power_pad[1, 1]) ** 2) tmp_distance = 1 / (tmp_distance1 +
                    # tmp_distance0)

                    # Compute Manhattan Distance
                    tmp_distance0 = np.abs(tmp_x[x_tile] - power_pad[0, 0]) + np.abs(tmp_y[y_tile] - power_pad[0, 1])
                    tmp_distance1 = np.abs(tmp_x[x_tile] - power_pad[1, 0]) + np.abs(tmp_y[y_tile] - power_pad[1, 1])
                    tmp_distance = 1 / (tmp_distance1 + tmp_distance0 + 1)
                    grid_distance[tmp_x[x_tile], tmp_y[y_tile]] = (grid_distance[tmp_x[x_tile], tmp_y[y_tile]] + tmp_distance) / 2

    return grid_distance


def main(metal_layers, layout_min_x, layout_min_y, target_area_x, target_area_y, num_probe_x_tiles, num_probe_y_tiles):
    wire_widths, wire_x0, wire_x1, wire_y0, wire_y1, wire_x, wire_y = clean_metal_paths_em_data(metal_layers)
    out = calculate_wire_distances_for_field(wire_widths, wire_x0, wire_x1, wire_y0, wire_y1, wire_x, wire_y)
    # grid_map = grid_map_feature(out, layout_min_x, layout_min_y, target_area_x, target_area_y, num_probe_x_tiles,
    # num_probe_y_tiles) np.save('power_wire_res.npy', out[:, 7]) np.save('grid_map.npy', grid_map) plt.imshow(
    # grid_map[0:, :])  # 53 plt.colorbar() plt.show()

    grid_distance = grid_distance_feature(out, layout_min_x, layout_min_y, target_area_x, target_area_y, num_probe_x_tiles, num_probe_y_tiles)
    np.save('grid_distance_map.npy', grid_distance)
    plt.imshow(grid_distance[0:, :])
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metal_layers", type=str, default=["metal5", "metal6"],
                        help="target metal layers")
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

    args = parser.parse_args()
    metal_layers = args.metal_layers
    target_area_x = args.target_area_x
    target_area_y = args.target_area_y
    num_probe_x_tiles = args.num_probe_x_tiles
    num_probe_y_tiles = args.num_probe_y_tiles
    layout_min_x = args.layout_min_x
    layout_min_y = args.layout_min_y

    start_time = time.time()
    try:
        sys.exit(main(metal_layers, layout_min_x, layout_min_y, target_area_x, target_area_y, num_probe_x_tiles, num_probe_y_tiles))
    except KeyboardInterrupt:
        sys.exit()
    finally:
        print("Running time:", time.time() - start_time, "seconds")
