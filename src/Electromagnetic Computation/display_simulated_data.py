"""
  ______  __  __   _____  _
 |  ____||  \/  | / ____|(_)
 | |__   | \  / || (___   _  _ __ ___
 |  __|  | |\/| | \___ \ | || '_ ` _ \
 | |____ | |  | | ____) || || | | | | |
 |______||_|  |_||_____/ |_||_| |_| |_|
--------------------------------------------------------------------------------------------------
Copyright (c) 2022, Tianjin University All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted
provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this list of conditions
  and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice, this list of
  conditions and the following disclaimer in the documentation and / or other materials provided
  with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS
OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
--------------------------------------------------------------------------------------------------

Author: Haocheng Ma, Yier Jin, Max Panoff, Jiaji He, Ya Gao
 -------------------------------------------------------------------
| Name         | Affiliation           | email                      |
| ------------ | --------------------- | -------------------------- |
| Haocheng Ma  | Tianjin University    | hc_ma@tju.edu.cn           |
| Yier Jin     | University of Florida | jinyier@gmail.com          |
| Max Panoff   | University of Florida | m.panoff@ufl.edu           |
| Jiaji He     | Tianjin University    | dochejj@tju.edu.cn         |
| Ya Gao       | Tianjin University    | gaoyaya@tju.edu.cn         |
 -------------------------------------------------------------------
 Details: EMSim is designed to predict the EM emanations from ICs at the layout level.
--------------------------------------------------------------------------------------------------
"""

import argparse
import sys
import numpy as np
import h5py
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def process_simulated_data(num_probe_x_tiles, num_probe_y_tiles, num_input_stimuli, start_point, sample_points, trace_file):
    """
    Convert .tr spice results into h5 format
    :param file_num: number of required plaintexts
    :param directory: path to result files from Spice simulation
    :param metal_layers: target metal layers
    :param outfile: path to final currents across metal wires
    :return: h5 format currents
    """

    em_map = np.zeros((num_input_stimuli, sample_points, num_probe_x_tiles, num_probe_y_tiles))
    f = h5py.File(trace_file, "r")
    # process x-axis em data
    print('--iterms: ', len(f.keys()), f.keys())
    x_loc = list(f.keys())
    n = len(x_loc)
    for i in range(n - 1):
        for j in range(n - 1 - i):
            if float(x_loc[j]) > float(x_loc[j + 1]):
                x_loc[j], x_loc[j + 1] = x_loc[j + 1], x_loc[j]
    x_loc1 = x_loc
    # process y-axis em data
    print('--iterms: ', len(f[x_loc[0]].keys()), f[x_loc[0]].keys())
    y_loc = list(f[x_loc[0]].keys())
    m = len(y_loc)
    for i in range(m - 1):
        for j in range(m - 1 - i):
            if float(y_loc[j]) > float(y_loc[j + 1]):
                y_loc[j], y_loc[j + 1] = y_loc[j + 1], y_loc[j]
    y_loc1 = y_loc
    print('X-axis list', x_loc1)
    print('Y-axis list', y_loc1)
    
    for current_probe_x_tiles in range(num_probe_x_tiles):
        for current_probe_y_tiles in range(num_probe_y_tiles):
            tmp_trace = f[x_loc1[current_probe_x_tiles]][y_loc1[current_probe_y_tiles]]["traces"][:]
            em_map[:, :, current_probe_x_tiles, current_probe_y_tiles] = tmp_trace[:, start_point: start_point + sample_points]
    return em_map


def main(num_probe_x_tiles, num_probe_y_tiles, num_input_stimuli, start_point, sample_points, trace_file):
    em_map = process_simulated_data(num_probe_x_tiles, num_probe_y_tiles, num_input_stimuli, start_point, sample_points, trace_file)

    im = []
    anis = []
    Writer = animation.writers['html']
    writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800)
    for im_num in range(num_input_stimuli):
        def updatefig1(frame, im_num):
            im[im_num].set_array(em_map[im_num, frame, ...])

        print("Writing result of test: ", im_num)
        test_data_num = im_num
        fig = plt.figure()
        max_val = np.max(em_map[im_num, ...])
        min_val = np.min(em_map[im_num, ...])

        im.append(plt.imshow(em_map[im_num, 0], vmin=min_val, vmax=max_val, cmap='jet'))

        anis.append(animation.FuncAnimation(fig, updatefig1, frames=range(sample_points), fargs=(im_num,), interval=150))
        anis[im_num].save("output_plots/run_contours_%d.html" % test_data_num, writer=writer)
        anis[im_num].save("output_plots/run_contours_%d.gif" % test_data_num, writer='pillow')
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_input_stimuli", type=int, default=10,
                        help="Number of required plaintexts for em computation")
    parser.add_argument("--start_point", type=int, default=0,
                        help="Start time point in simulation per input stimuli")
    parser.add_argument("--sample_points", type=int, default=20,
                        help="Sampling time point in simulation per input stimuli")
    parser.add_argument("--num_probe_x_tiles", type=int, default=48,
                        help="Number of point grid in x axial direction")
    parser.add_argument("--num_probe_y_tiles", type=int, default=48,
                        help="Number of point grid in y axial direction")
    parser.add_argument("--trace_file", type=str, default="fastem_sim_traces.h5",
                        help="Path to read the input")

    args = parser.parse_args()
    num_input_stimuli = args.num_input_stimuli
    start_point = args.start_point
    sample_points = args.sample_points
    num_probe_x_tiles = args.num_probe_x_tiles
    num_probe_y_tiles = args.num_probe_y_tiles
    trace_file = args.trace_file

    start_time = time.time()
    try:
        sys.exit(main(num_probe_x_tiles, num_probe_y_tiles, num_input_stimuli, start_point, sample_points, trace_file))
    except KeyboardInterrupt:
        sys.exit()
    finally:
        print("Running time:", time.time() - start_time, "seconds")
