import numpy as np
import sys
import argparse
import time
import h5py
import pynvml


class EM_field_simulator():
    def __init__(self, transient_file, metal_layers, use_gpu, num_input_stimuli, clock_period_ns,
                 start_time_ns, end_time_ns, num_horizontal_tiles_per_wire, num_vertical_tiles_per_wire,
                 target_area_x, target_area_y, layout_max_z, num_probe_x_tiles, num_probe_y_tiles,
                 num_probe_z_tiles, ns_per_sample, output, layout_min_x, layout_min_y):
        """
        Creates an EM simulator instance
        :param transient_file: file with transient data
        :param metal_layers: metal layers to use in simulation
        :param qk: package, either numpy for cpu or cupy gpu, to use acceleration
        :param use_gpu: whether we are using gpu
        """
        self.metal_layers = metal_layers
        self.transient_analysis_file = transient_file
        if use_gpu:
            import cupy as qk
        else:
            import numpy as qk
        self.qk = qk
        self.use_gpu = use_gpu
        self.num_input_stimuli = num_input_stimuli
        self.clock_period_ns = clock_period_ns
        self.start_time_ns = start_time_ns
        self.end_time_ns = end_time_ns
        self.num_horizontal_tiles_per_wire = num_horizontal_tiles_per_wire
        self.num_vertical_tiles_per_wire = num_vertical_tiles_per_wire
        self.target_area_x = target_area_x
        self.target_area_y = target_area_y
        self.num_probe_x_tiles = num_probe_x_tiles
        self.num_probe_y_tiles = num_probe_y_tiles
        # note this may bring an additional dimension to the array
        self.num_probe_z_tiles = num_probe_z_tiles
        self.layout_max_z = layout_max_z
        self.ns_per_sample = ns_per_sample
        self.output = output
        self.layout_min_x = layout_min_x
        self.layout_min_y = layout_min_y

    def clean_transient_data(self):
        """
        Assigns cleans transient data using parameters set at the instance level to obtain current
        information for each wire and or input stimulus. Sets this cleaned data as a component of the instance
        :return: True
        """
        # needs to load a numpy array
        print("Loading transient data...")
        tran_file = h5py.File(self.transient_analysis_file, 'r')
        data_init = tran_file.get('spice_sim')
        data_init = np.asarray(data_init)
        tran_file.close()
        print("Finished loading transient data!")
        print("Cleaning transient data...")
        # skip time data
        all_transient = data_init[:]
        # find the desired period to collect data from
        desired_period_ns = self.end_time_ns - self.start_time_ns + 1
        # find the number of data points we can extract
        # this is the total length of the data divided by the period plus any remainder
        num_data_points = int(desired_period_ns * self.num_input_stimuli / self.ns_per_sample)
        # create an empty array that can hold the number of desired data points for each component
        transient_current_info = np.zeros((all_transient.shape[0], num_data_points), dtype=np.float32)
        print("Transient current data shape:", transient_current_info.shape)
        # for each input stimulus, get the transient data of the desired period
        print('Start point', 0 * desired_period_ns, ', end point', desired_period_ns + 0 * desired_period_ns, ', time scale', self.ns_per_sample,)
        for input_stimulus in range(self.num_input_stimuli):
            transient_current_info[:, input_stimulus * desired_period_ns:
                                      desired_period_ns + input_stimulus * desired_period_ns] = \
                all_transient[:, self.start_time_ns + input_stimulus * self.clock_period_ns:
                                 int((self.end_time_ns + 1 + input_stimulus * self.clock_period_ns)/self.ns_per_sample)]
        # should be num_wires x num_data_points (x num_stimuli?)
        print("Finished cleaning transient data!")
       
        # print(transient_current_info)
        if self.use_gpu:
            # get size of gpu
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mempool = pynvml.nvmlDeviceGetMemoryInfo(handle)
            # get size of the data
            # 32 bit float = 4 bytes
            sim_data_size = data_init.shape[0] * data_init.shape[1] * 4
            # see how many bytes are free
            print('GPU mem limit', mempool.free)   
            print('Size of data',  sim_data_size)
            # if there is not enough space for the data
            if mempool.free < sim_data_size:
                print('Transient data too large for GPU, splitting results')
                # split it into multiple subsections
                num_splits = int(sim_data_size / mempool.free) + 1
                # save the original output name
                final_output = self.output
                for split in range(num_splits-1):
                    temp_tran_current_info = transient_current_info[split*int(transient_current_info.shape[0]/num_splits):(split+1)*int(transient_current_info.shape[0]/num_splits)]
                    # write the results of each subsection into a temp file
                    self.transient_current_info = self.qk.asarray(temp_tran_current_info)
                    self.output = 'split_out_' + str(split) + '.h5'
                    assert self.clean_metal_paths_em_data()
                    assert self.calculate_wire_distances_for_field()
                    assert self.create_sub_grid_for_wires()
                    assert self.create_probe_grid()
                    assert self.compute()
                # this hold the remaineder in case of an uneven split
                sim_data_splits[num_splits-1] = transient_current_info[(num_splits-1)*int(transient_current_info.shape[0]/num_splits):]
                self.output = 'split_out_' + str(num_splits-1) + '.h5'
                assert self.clean_metal_paths_em_data()
                assert self.calculate_wire_distances_for_field()
                assert self.create_sub_grid_for_wires()
                assert self.create_probe_grid()
                assert self.compute()
                # now to recombine into the original output
                final_out = h5py.File(final_output, 'w')
                # create the file
                x_locations = [str((x_tile * self.target_area_x / self.num_probe_x_tiles + self.layout_min_x) * 1000000)
                               for x_tile in range(self.num_probe_x_tiles)]
                y_locations = [str((y_tile * self.target_area_y / self.num_probe_y_tiles + self.layout_min_y) * 1000000)
                               for y_tile in range(self.num_probe_y_tiles)]
                out_file = h5py.File(self.output, 'w')
                for x_loc in range(self.num_probe_x_tiles):
                    x_group = out_file.create_group(x_locations[x_loc])
                    for y_loc in range(self.num_probe_y_tiles):
                        group = x_group.create_group(y_locations[y_loc])
                        # create  the dataset by running through this group in all temp files
                        combined_data = []
                        for split in range(num_splits):
                            temp_file = h5py.File('split_out_' + str(split) + '.h5')
                            target_group = temp_file[x_loc][y_loc]
                            target_data = target_group.get('traces')
                            target_data = np.asarray(target_data)
                            combined_data.append(target_data)
                            temp_file.close()
                        # once we have all the data, convert to numpy and write to file
                        combined_data = np.asarray(combined_data)
                        group.create_dataset('traces', data=combined_data)
                        # we'll flush here to be safe
                        out_file.flush()
                # after all points have been combined, close the file
                out_file.close()
                # and exit the program
                sys.exit()
                    
        # convert to cupy
        self.transient_current_info = self.qk.asarray(transient_current_info)

        return True

    def clean_metal_paths_em_data(self):
        """
        Concatenates the EM relevant data from Step 2/Prep_For_Sim.py into a single list. Assigns each useful
        type of value (column) of the metal layer data to its own component of the instance. Has 7 outputs, each
        a vector where each element relates to a single wire
        :return: True
        """
        # init array of null size
        # must be numpy, as we will be reading in from text file
        metal_em_info = np.zeros((0, 6), dtype=self.qk.float32)
        # load in metal path em info
        print("Loading metal layer data...")
        for metal in self.metal_layers:
            metal_em_info = np.concatenate(
                (metal_em_info, np.genfromtxt(metal + "_em.txt", delimiter=',', dtype=np.float32)))
        if self.use_gpu:
            metal_em_info = self.qk.asarray(metal_em_info)
        print("Finished loading metal layer data!")
        print("EM metal layers shape:", metal_em_info.shape)

        # digest metal path info
        width = metal_em_info[:, 0]
        x0 = metal_em_info[:, 2] / 1000000
        y0 = metal_em_info[:, 3] / 1000000
        x1 = metal_em_info[:, 4] / 1000000
        y1 = metal_em_info[:, 5] / 1000000
        # signed vector from x0 (y0) to x1 (y1)
        x = x1 - x0
        y = y1 - y0
        # each is num_wires x 1
        self.wire_widths, self.wire_x0, self.wire_x1, self.wire_y0, self.wire_y1, self.wire_x, self.wire_y = \
            width, x0, x1, y0, y1, x, y
        return True

    def calculate_wire_distances_for_field(self):
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
        out = np.zeros((self.wire_widths.shape[0], 7), dtype=self.qk.float32)
        if self.use_gpu:
            self.wire_x = self.qk.asnumpy(self.wire_x)
            self.wire_y = self.qk.asnumpy(self.wire_y)
            self.wire_y0 = self.qk.asnumpy(self.wire_y0)
            self.wire_y1 = self.qk.asnumpy(self.wire_y1)
            self.wire_x0 = self.qk.asnumpy(self.wire_x0)
            self.wire_x1 = self.qk.asnumpy(self.wire_x1)
            self.wire_widths = self.qk.asnumpy(self.wire_widths)

        # set values as desired in rows where wire widths are 0
        vertical = np.asarray(self.wire_x == 0).nonzero()
        # dir_y in column y_dir
        out[vertical, 1] = np.sign(self.wire_y[vertical])
        # x0_in - W_in / 2 in column p_x0
        out[vertical, 3] = self.wire_x0[vertical] - self.wire_widths[vertical] / 2
        # x0_in + W_in / 2 in column p_x1
        out[vertical, 4] = self.wire_x0[vertical] + self.wire_widths[vertical] / 2
        # y0_in in column p_y0
        out[vertical, 5] = self.wire_y0[vertical]
        # y1_in in column p_y1
        out[vertical, 6] = self.wire_y1[vertical]

        # set values as desired where wired lengths are 0
        # dir_x
        horizontal = np.asarray(self.wire_y == 0).nonzero()
        out[horizontal, 0] = np.sign(self.wire_x[horizontal])
        # y0_in - W_in / 2 in column p_y0
        out[horizontal, 5] = self.wire_y0[horizontal] - self.wire_widths[horizontal] / 2
        # y0_in + W_in / 2 in column p_y1
        out[horizontal, 6] = self.wire_y0[horizontal] + self.wire_widths[horizontal] / 2
        # x0_in in column p_x0
        out[horizontal, 3] = self.wire_x0[horizontal]
        # x1_in in column p_x1
        out[horizontal, 4] = self.wire_x1[horizontal]

        # num_wires x 7
        if self.use_gpu:
            self.wire_measurement_for_field_calulations = self.qk.asarray(out)
            self.wire_x = self.qk.asarray(self.wire_x)
            self.wire_y = self.qk.asarray(self.wire_y)
            self.wire_y0 = self.qk.asarray(self.wire_y0)
            self.wire_y1 = self.qk.asarray(self.wire_y1)
            self.wire_x0 = self.qk.asarray(self.wire_x0)
            self.wire_x1 = self.qk.asarray(self.wire_x1)
            self.wire_widths = self.qk.asarray(self.wire_widths)
        else:
            self.wire_measurement_for_field_calulations = out
        print("Finished calculating static wire distance matrix!")
        print("Static wire distance matrix shape:", self.wire_measurement_for_field_calulations.shape)
        return True

    def create_sub_grid_for_wires(self):
        """
        Does all the prep calculations (not the field calculations) from field1.m.
        Saves these as components of the instance. dros = self.dros, PX = self.wire_horizontal_distances_grid,
        and PY=wire_vertical_distances_grid
        :return: True
        """
        # find the horizontal wire distance
        # num_wires x 1
        print("Calculating wire subgrid...")
        # num_wires x 1
        horizontal_wire_width = self.wire_measurement_for_field_calulations[:, 4] - \
        self.wire_measurement_for_field_calulations[:, 3]
        # find the vertical wire distance
        # num_wires x1
        vertical_wire_width = self.wire_measurement_for_field_calulations[:, 6] - \
                                self.wire_measurement_for_field_calulations[:, 5]

        # num_wires x 1
        width_per_horizontal_wire_tile = horizontal_wire_width / self.num_horizontal_tiles_per_wire
        # num_wires x 1
        width_per_vertical_wire_tile = vertical_wire_width / self.num_vertical_tiles_per_wire

        # calculate dros
        # num_wires x1
        # infinitesimal area for each wire
        self.dros = self.qk.absolute(width_per_horizontal_wire_tile * width_per_vertical_wire_tile)

        # start empty num_wires x num_horizontal_tiles_per_wire matrix
        xh = self.qk.zeros((horizontal_wire_width.shape[0], self.num_horizontal_tiles_per_wire))
        # each column will have horizontal_wire_width higher values than the last
        for i in range(self.num_horizontal_tiles_per_wire):
            xh[:, i] = width_per_horizontal_wire_tile * (0.5 + i)
        # new matrix, is is just xh repeated num_vertical_tiles_per_wire times along a new axis
        # the transpose (changes in x should be perpedicular to changes in y) is done in create_probe_grid
        # num_wires x num_horizontal_tiles_per_wire x num_vertical_tiles_per_wire
        self.wire_horizontal_distances_grid = self.qk.repeat(xh[:, :, None], self.num_vertical_tiles_per_wire, axis=2)

        # start empty num_wires x num_vertical_tiles_per_wire matrix
        yv = self.qk.zeros((horizontal_wire_width.shape[0], self.num_vertical_tiles_per_wire))
        # each column will have vertical_wire_width higher values than the last
        for i in range(self.num_vertical_tiles_per_wire):
            yv[:, i] = width_per_vertical_wire_tile * (0.5 + i)
        # new matrix, is is just yv repeated num_horizontal_tiles_per_wire times along a new axis
        # num_wires x num_vertical_tiles_per_wire x num_horizontal_tiles_per_wire
        self.wire_vertical_distances_grid = self.qk.repeat(yv[:, :, None], self.num_horizontal_tiles_per_wire, axis=2)
        print("Finished calculating wire subgrid!")
        print("Wire subgrids shapes:",
              self.wire_horizontal_distances_grid.shape,
              self.wire_vertical_distances_grid.shape)
        return True

    def create_probe_grid(self):
        """
        Recreates the results of the middle two loops from compute.m as a matrix. This is
        the most difficult/largest matrix, being a 5d array (may go to 6d once z is included)
        with all desired intermediate values being calculated, so that the actual simulation
        runs as quickly as possible this array is saved as a component of the instance.
        Final shape should be:
                              num_probe_x_tiles x num_probe_y_tiles (x num_probe_z_tiles)
                                        x num_wires x
                              num_horizontal_wire_tiles x num_vertical_wire_tiles
        :return: True
        """
        print("Calculating matrix of all possible locations and distances for field calculations")
        # 1x1
        probe_spacing_x = self.target_area_x / self.num_probe_x_tiles
        # 1x1
        probe_spacing_y = self.target_area_y / self.num_probe_y_tiles
        # 1x1
        probe_spacing_z = self.layout_max_z / self.num_probe_z_tiles

        # every possible design tile measurement location for x
        # num_probe_x_tiles x 1
        x_probe_locations = self.qk.arange(0, self.num_probe_x_tiles)
        # num_probe_x_tiles x 1
        x_probe_locations = (x_probe_locations + 0.5) * probe_spacing_x + self.layout_min_x

        # every possible design tile location for y
        # num_probe_y_tiles x 1
        y_probe_locations = self.qk.arange(0, self.num_probe_y_tiles)
        # num_probe_y_tiles x 1
        y_probe_locations = (y_probe_locations + 0.5) * probe_spacing_y + self.layout_min_y

        # every possible design tile location for z
        # num_probe_z_tiles x 1
        z_probe_locations = self.qk.arange(0, self.num_probe_z_tiles)
        # num_probe_z_tiles x 1
        z_probe_locations = (z_probe_locations + 0.5) * probe_spacing_z

        # obtain all p_x0, p_y0 locations
        # num_wires x 1
        wire_left_corners = self.wire_measurement_for_field_calulations[:, 3]
        # num_wires x 1
        wire_bottom_corners = self.wire_measurement_for_field_calulations[:, 5]

        # why do I have this shaped like this again?
        # num_wires x num_horizontal_tiles_per_wire x num_vertical_tiles_per_wire
        wire_x_distances = self.qk.zeros(self.wire_horizontal_distances_grid.shape, dtype=self.qk.float32)
        # num_wires x num_horizontal_tiles_per_wire x num_vertical_tiles_per_wire
        wire_y_distances = self.qk.zeros(self.wire_vertical_distances_grid.shape, dtype=self.qk.float32)
        # calculate the distance from wire corners to each wire measurement point
        for i in range(wire_left_corners.shape[0]):
            # num_horizontal_tiles_per_wire x num_vertical_tiles_per_wire
            wire_x_distances[i] = wire_left_corners[i] + self.qk.transpose(self.wire_horizontal_distances_grid[i])
            # num_vertical_tiles_per_wire x num_horizontal_tiles_per_wire
            wire_y_distances[i] = wire_bottom_corners[i] + self.wire_vertical_distances_grid[i]

        # change over to numpy for outer calculations
        if self.use_gpu:
            wire_x_distances = self.qk.asnumpy(wire_x_distances)
            x_probe_locations = self.qk.asnumpy(x_probe_locations)

        # add every wire tile sub-grid x offset to every probe x location
        # num_x_probe_locations x num_wires x num_horizontal_tiles_per_wire x num_vertical_tiles_per_wire
        design_x_distances = np.subtract.outer(x_probe_locations, wire_x_distances)
        if self.use_gpu:
            design_x_distances = self.qk.asarray(design_x_distances)

        # outer calcs must be done in numpy
        if self.use_gpu:
            wire_y_distances = self.qk.asnumpy(wire_y_distances)
            y_probe_locations = self.qk.asnumpy(y_probe_locations)
        # add every wire tile sub-grid y offset to every probe y location
        # num_y_probe_locations x num_wires x num_vertical_tiles_per_wire x num_horizontal_tiles_per_wire
        design_y_distances = np.subtract.outer(y_probe_locations, wire_y_distances)
        if self.use_gpu:
            design_y_distances = self.qk.asarray(design_y_distances)

        # get the total distances for every possible design x and y grid combination
        all_design_tile_total_distances = \
            self.qk.zeros((self.num_probe_x_tiles * self.num_probe_y_tiles, self.num_probe_z_tiles, design_y_distances.shape[1],
                           design_y_distances.shape[2],
                           design_y_distances.shape[3]), dtype=self.qk.float32)

        for x_tile in range(self.num_probe_x_tiles):
            for y_tile in range(self.num_probe_y_tiles):
                for z_tile in range(self.num_probe_z_tiles):
                    all_design_tile_total_distances[x_tile * self.num_probe_y_tiles + y_tile, z_tile] = self.qk.sqrt(
                        self.qk.square(design_x_distances[x_tile])
                        + self.qk.square(design_y_distances[y_tile])
                        + self.qk.square(z_probe_locations[z_tile]))
        print("all_design_tile_total_distances shape: ", all_design_tile_total_distances.shape)
        # repeat the x and y grids so that operations with the complete matrix are possible
        design_x_distances = self.qk.repeat(design_x_distances, self.num_probe_y_tiles, axis=0)
        design_y_distances = self.qk.tile(design_y_distances, (self.num_probe_x_tiles, 1, 1, 1))
        print("design_y_distances shape: ", design_y_distances.shape)

        # scale based on total x and y distance
        distance_scaler = self.qk.power(all_design_tile_total_distances, 3)
        self.scaled_x_distances = self.qk.zeros(all_design_tile_total_distances.shape)
        self.scaled_y_distances = self.qk.zeros(all_design_tile_total_distances.shape)
        # TODO there has to be a better way to do this, see the field computation method
        for z_tile in range(self.num_probe_z_tiles):
            # find the scaled distance (including different vertical values) for each horizontal value
            # in probe and wire grids
            self.scaled_x_distances[:, z_tile] = design_x_distances / distance_scaler[:, z_tile]
            # find the scaled distance (including different horizontal values) for each vertical value
            # in probe and wire grids
            self.scaled_y_distances[:, z_tile] = design_y_distances / distance_scaler[:, z_tile]
        print("scaled_x_distances.shape: ", self.scaled_x_distances.shape)
        # reshape the matrix from num_probe_x_tiles * num_probe_y_tiles to num_probe_x_tiles x num_probe_y_tiles
        self.scaled_x_distances.shape = (self.num_probe_x_tiles, self.num_probe_y_tiles, self.num_probe_z_tiles,
                                    all_design_tile_total_distances.shape[2],
                                    self.num_vertical_tiles_per_wire, self.num_horizontal_tiles_per_wire)
        # reshape the matrix from num_probe_x_tiles * num_probe_y_tiles to num_probe_x_tiles x num_probe_y_tiles
        self.scaled_y_distances.shape = (self.num_probe_x_tiles, self.num_probe_y_tiles,
                                    all_design_tile_total_distances.shape[1],all_design_tile_total_distances.shape[2],
                                    self.num_vertical_tiles_per_wire, self.num_horizontal_tiles_per_wire)
        print("Complete matrix shape:", self.scaled_y_distances.shape)
        return True

    def field(self, t):
        """
        Calculates the field at all probe locations at a certain time
        :param current: transient current for all wires at a given time
        :return: Hz, a num_probe_x_locations x num_probe_y_locations matrix
        """
        # multiply every transient current by its wire
        # num_wires x 1
        mag = self.transient_current_info[:, t] / self.wire_widths
        # calculate the current amount in the x and y directions
        # num_wires x 1
        Jx = mag * self.wire_measurement_for_field_calulations[:, 0]
        # num_wires x 1
        Jy = mag * self.wire_measurement_for_field_calulations[:, 1]
        # create empty matrix to hold all the various sub-components of the field
        total_field = self.qk.zeros((self.scaled_x_distances.shape[0], self.scaled_x_distances.shape[1],
                                     self.scaled_x_distances.shape[2]),
                                    dtype=self.qk.float32)
        # Jy * x_distances
        # multiply each member of the x matrix by the Jy of the corresponding component
        dim_array = np.ones((1, self.scaled_x_distances.ndim), int).ravel()
        dim_array[3] = -1
        component_currents_in_y_reshaped = Jy.reshape(dim_array)
        temp_x = self.scaled_x_distances * component_currents_in_y_reshaped

        # Jx * y_distances
        # multiply each member of the y matrix by the Jx of the corresponding component
        dim_array = np.ones((1, self.scaled_y_distances.ndim), int).ravel()
        dim_array[3] = -1
        component_currents_in_x_reshaped = Jx.reshape(dim_array)
        temp_y = self.scaled_y_distances * component_currents_in_x_reshaped

        # Jy * x_distances - Jx * y_distances
        temp = temp_x - temp_y
        # sum all the wire sub_grid components
        total_field = self.qk.sum(temp, axis=(4, 5))
        # multiply every component by it's dros
        Hz = total_field * self.dros / (4 * self.qk.pi)
        # sum the effects of all the wires
        # num_probe_x_locations x num_probe_y_locations
        Hz = self.qk.sum(Hz, axis=3)

        return Hz

    def compute(self):
        # create empty arrays to fill in loops
        self.Hz = self.qk.zeros((self.transient_current_info.shape[1],
                                 self.num_probe_x_tiles, self.num_probe_y_tiles, self.num_probe_z_tiles),
                                dtype=self.qk.float32)

        # track simulation and real time
        time_tracker = 1
        for sim_time in range(self.transient_current_info.shape[1]):  # time to run the simulation
            real_time = time.time()
            print("sim time is", time_tracker*self.ns_per_sample, "time points of", self.transient_current_info.shape[1]
                  * self.ns_per_sample, "time points")
            # this should return the field at each of the probe points at a given sim_time
            self.Hz[sim_time] = self.field(sim_time)
            # increment sim time
            time_tracker += 1
            # report real time
            print("Loop took", time.time() - real_time, "seconds")
        # save the Hz data to use
        x_locations = [str((x_tile*self.target_area_x/self.num_probe_x_tiles + self.layout_min_x) * 1000000) for x_tile in range(self.num_probe_x_tiles)]
        y_locations = [str((y_tile*self.target_area_y/self.num_probe_y_tiles + self.layout_min_y)* 1000000) for y_tile in range(self.num_probe_y_tiles)]
        out_file = h5py.File(self.output, 'w')
        for x_loc in range(self.num_probe_x_tiles):
            x_group = out_file.create_group(x_locations[x_loc])
            for y_loc in range(self.num_probe_y_tiles):
                group = x_group.create_group(y_locations[y_loc])
                all_fields = self.Hz[:, x_loc, y_loc, 0]
                all_fields.shape = (self.num_input_stimuli, -1)
                print(all_fields.shape)
                probe_from_field = all_fields
                if self.use_gpu:
                    probe_from_field = self.qk.asnumpy(probe_from_field)
                data = group.create_dataset('traces', data=probe_from_field)
        out_file.flush()
        out_file.close()
        return True

    def run_simulation(self):
        assert self.clean_transient_data()
        assert self.clean_metal_paths_em_data()
        assert self.calculate_wire_distances_for_field()
        assert self.create_sub_grid_for_wires()
        assert self.create_probe_grid()
        assert self.compute()


def main(metal_layers, transient_file, use_gpu, num_input_stimuli, clock_period_ns, start_time_ns, end_time_ns, target_area_x,
         target_area_y, num_probe_x_tiles, num_probe_y_tiles, num_probe_z_tiles, layout_max_z, num_horizontal_tiles_per_wire,
         num_vertical_tiles_per_wire, ns_per_sample, output, layout_min_x, layout_min_y):
    simulator = EM_field_simulator(metal_layers=metal_layers, transient_file=transient_file, use_gpu=use_gpu,
                                   num_input_stimuli=num_input_stimuli, clock_period_ns=clock_period_ns, start_time_ns=start_time_ns,
                                   end_time_ns=end_time_ns, target_area_x=target_area_x, target_area_y=target_area_y,
                                   num_probe_x_tiles=num_probe_x_tiles, num_probe_y_tiles=num_probe_y_tiles,
                                   num_probe_z_tiles=num_probe_z_tiles, layout_max_z=layout_max_z,
                                   num_horizontal_tiles_per_wire=num_horizontal_tiles_per_wire,
                                   num_vertical_tiles_per_wire=num_vertical_tiles_per_wire, ns_per_sample=ns_per_sample,
                                   output=output, layout_min_x=layout_min_x, layout_min_y=layout_min_y)
    simulator.run_simulation()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metal_layers", type=str, default=["metal5", "metal6"],
                        help="target metal layers")
    parser.add_argument("--transient_file", type=str, default="last_binary_run.h5",
                        help="Path to the transient analysis file, should be in csv form")
    parser.add_argument("--use_gpu", action='store_true', help="allow for gpu acceleration")
    parser.add_argument("--num_input_stimuli", type=int, default=10,
                        help="Number of required plaintexts for em computation")
    parser.add_argument("--clock_period_ns", type=int, default=39,
                        help="Amount of sample points per input stimuli")
    parser.add_argument("--start_time_ns", type=int, default=0,
                        help="Start time point in simulation per input stimuli")
    parser.add_argument("--end_time_ns", type=int, default=38,
                        help="End time point in simulation per input stimuli")
    parser.add_argument("--target_area_x", type=float, default=1140.52 / 1000000,
                        help="Target simulated area in x axial direction")
    parser.add_argument("--target_area_y", type=float, default=840.08 / 1000000,
                        help="Target simulated area in y axial direction")
    parser.add_argument("--num_probe_x_tiles", type=int, default=23,
                        help="Number of point grid in x axial direction")
    parser.add_argument("--num_probe_y_tiles", type=int, default=17,
                        help="Number of point grid in y axial direction")
    parser.add_argument("--num_probe_z_tiles", type=int, default=1,
                        help="Number of point grid in z axial direction")
    parser.add_argument("--layout_max_z", type=float, default=0.9e-3,
                        help="Target simulated distance in z axial direction, option value = true value x 2")
    parser.add_argument("--num_horizontal_tiles_per_wire", type=int, default=10,
                        help="Number of horizontal tiles per metal wire")
    parser.add_argument("--num_vertical_tiles_per_wire", type=int, default=10,
                        help="Number of vertical tiles per metal wire")
    parser.add_argument("--ns_per_sample", type=int, default=1,
                        help="Simulated value of time scale per sample point")
    parser.add_argument("--output", type=str, default="fastem_sim_traces.h5",
                        help="Path to write the output")
    parser.add_argument("--layout_min_x", type=float, default=20.6 / 1000000,
                        help="Reference coordinate in x axial direction")
    parser.add_argument("--layout_min_y", type=float, default=19.2 / 1000000,
                        help="Reference coordinate in y axial direction")

    args = parser.parse_args()
    metal_layers = args.metal_layers
    transient_file = args.transient_file
    use_gpu = args.use_gpu
    num_input_stimuli = args.num_input_stimuli
    clock_period_ns = args.clock_period_ns
    start_time_ns = args.start_time_ns
    end_time_ns = args.end_time_ns
    target_area_x = args.target_area_x
    target_area_y = args.target_area_y
    num_probe_x_tiles = args.num_probe_x_tiles
    num_probe_y_tiles = args.num_probe_y_tiles
    num_probe_z_tiles = args.num_probe_z_tiles
    layout_max_z = args.layout_max_z
    num_horizontal_tiles_per_wire = args.num_horizontal_tiles_per_wire
    num_vertical_tiles_per_wire = args.num_vertical_tiles_per_wire
    ns_per_sample = args.ns_per_sample
    output = args.output
    layout_min_x = args.layout_min_x
    layout_min_y = args.layout_min_y

    start_time = time.time()
    try:
        sys.exit(main(metal_layers, transient_file, use_gpu, num_input_stimuli, clock_period_ns, start_time_ns, end_time_ns, target_area_x,
                      target_area_y, num_probe_x_tiles, num_probe_y_tiles, num_probe_z_tiles, layout_max_z, num_horizontal_tiles_per_wire,
                      num_vertical_tiles_per_wire, ns_per_sample, output, layout_min_x, layout_min_y))
    except KeyboardInterrupt:
        sys.exit()
    finally:
        print("Running time:", time.time() - start_time, "seconds")