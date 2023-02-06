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
import re
import sys
import numpy as np
import time
import math
start = time.process_time()


def get_cell_pin_info(parasitic_netlist_path):
    """
    Get the pin locations of each logic cell
    :param parasitic_netlist_path: the path to the parasitic netlist used to create the file
    :return: cell name, x and y pins
    """

    parasitic_netlist_file = open(parasitic_netlist_path, 'r')
    # cell name in the parasitic netlist
    name = []
    # cell pin_x
    X = []
    # cell pin_y
    Y = []

    for line in parasitic_netlist_file.readlines():
        # search the instance pin line
        if re.search('\*|I \(X.*', line):
            # split on spaces
            split_line = line.split(" ")
            if len(split_line) > 6:
                # judge if the line has string 'VDD' or 'VSS'
                judge_false = "VDD" == split_line[3] or "VSS" == split_line[3]
                # judge if the line has string 'X..' (cell name)
                judge_true = "X" in split_line[2]
                # exclude the line with string 'VDD' and 'VSS'
                judge_true_again = "X" in split_line[4]
                # exclude the parasitic transistor
                if judge_false:
                    continue
                elif judge_true and judge_true_again:
                    # update data
                    name.append(split_line[2])
                    X.append(float(split_line[6]))
                    Y.append(float(split_line[7].strip(')\n')))
    parasitic_netlist_file.close()
    X = np.asarray(X)
    Y = np.asarray(Y)
    name_reorder = []
    X_reorder = []
    Y_reorder = []
    name_classify = list(set(name))
    # exclude the repetitive data
    for current_num_classify in range(len(name_classify)):
        index_list = [index for index, item in enumerate(name) if item == name_classify[current_num_classify]]
        # the minimum data is preserved
        X_min = np.min(X[index_list])
        Y_min = np.min(Y[index_list])
        name_reorder.append(name[index_list[0]])
        X_reorder.append(X_min)
        Y_reorder.append(Y_min)
    assert len(X_reorder) == len(Y_reorder), "Error: please check the instance reordering"

    return name_reorder, X_reorder, Y_reorder


def match_cell_info(def_path, cell_name, cell_pin_X, cell_pin_Y):
    """
    mathch the cell infor from parasistic file and def file
    :param def_file: design.def
    :param cell_name: cell name of the design
    :param cell_pin_X: x pins of all cell
    :param cell_pin_Y: y pins of all cell
    :return: hierarchical instance
    """
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
                instance_x0.append(float(split_line[9])/2000)
                instance_y0.append(float(split_line[10])/2000)
                instance_count += 1
            elif len(split_line) > 7:
                instance_hierarchy.append(split_line[1])
                instance_type.append(split_line[2])
                instance_x0.append(float(split_line[6])/2000)
                instance_y0.append(float(split_line[7])/2000)
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

    cell_name = np.asarray(cell_name)
    cell_pin_X= np.asarray(cell_pin_X)
    cell_pin_Y= np.asarray(cell_pin_Y)

    # the index of sorted data, from small to large
    sort_index_vertical = np.argsort(cell_pin_Y)
    cell_pin_Y_reorder = cell_pin_Y[sort_index_vertical]
    cell_pin_X_reorder = cell_pin_X[sort_index_vertical]
    cell_name_reorder = cell_name[sort_index_vertical]

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

        tmp_pin_X = cell_pin_X_reorder[start_num: end_num]
        tmp_cell_name = cell_name_reorder[start_num: end_num]
        sort_index_horizontal = np.argsort(tmp_pin_X)
        cell_pin_X_reorder[start_num: end_num] = tmp_pin_X[sort_index_horizontal]
        cell_name_reorder[start_num: end_num] = tmp_cell_name[sort_index_horizontal]

    return cell_name_reorder,  instance_hierarchy_reorder


def process_time_interval_use(desired_time_interval, desired_time_scale):
    """
    process the time interval to use
    :param desired_time_interval: desired time interval to use
    :param desired_time_scale: desired time scale used in power analysis
    :return: final time interval for subsequent step
    """
    if desired_time_scale < 1:
        intermediate = '%e' % desired_time_scale
        multiple = math.pow(10, int(intermediate.partition('-')[2]))
        final_time_interval = int(desired_time_interval * multiple)
        final_time_divider = int(desired_time_scale * multiple)
    else:
        final_time_interval = int(desired_time_interval)
        final_time_divider = int(desired_time_scale)
    return final_time_interval, final_time_divider


def process_power_report_use(num_plaintexts, start_time_point, desired_time_interval, power_report_init_path, power_report_path):
    """
    process the power report from power analysis
    :param num_plaintexts: the number of the plaintexts
    :param start_time_point: start time point in the initial power report
    :param desired_time_interval: final time interval for subsequent step
    :param power_report_init_path: path to the init power report file
    :param power_report_path: path to the power report file
    :return: combined power report
    """
    power_report_file = open(power_report_path, 'w', newline='\n')

    # create patterns used to extract parameters
    pattern_power = r'[\d]+  [\d]+.*'
    pattern_time_point = r'^\d+\n$'
    start_enable = True
    for current_num in range(num_plaintexts):
        power_report_init_file = open(power_report_init_path + str(current_num) + '.out', 'r')
        left_time_interval = start_time_point
        if current_num == 0:
            for line in power_report_init_file.readlines():
                if re.match(pattern_time_point, line):
                    current_time_point = int(line.strip('\n')) - left_time_interval
                    if current_time_point >= desired_time_interval:
                        start_enable = False
                        break
                    elif current_time_point < 0:
                        start_enable = False
                    else:
                        start_enable = True
                        new_time_point = current_time_point + desired_time_interval * current_num
                        power_report_file.write(str(new_time_point) + "\n")
                elif start_enable:
                    power_report_file.write(line)
        else:
            for line in power_report_init_file.readlines():
                if re.match(pattern_time_point, line):
                    current_time_point = int(line.strip('\n')) - left_time_interval
                    if current_time_point >= desired_time_interval:
                        start_enable = False
                        break
                    elif current_time_point < 0:
                        start_enable = False
                    else:
                        start_enable = True
                        new_time_point = current_time_point + desired_time_interval * current_num
                        power_report_file.write(str(new_time_point) + "\n")
                elif re.match(pattern_power, line) and start_enable:
                    power_report_file.write(line)
        power_report_init_file.close()
    power_report_file.close()


def logic_cell_modeling(top_cell, num_plaintexts, desired_time_interval, final_time_divider, power_report_path, instance_hierarchy, power_supply_voltage):
    """
    modeling the current waveform of logic cells
    :param top_cell: top cell in the design
    :param num_plaintexts: the number of the plaintexts
    :param desired_time_interval: final time interval for subsequent step
    :param final_time_divider: final time divider used for final time interval
    :param power_report_path: the path to the power_report used to create the file
    :param instance_hierarchy: instance with design hierarchy
    :param power_supply_voltage: supply voltage for logic cells
    :return: the current waveform of logic cells
    """
    power_file = open(power_report_path, 'r')
    print("Load power data, finished.")
    # create empty list for cell name, keyword and time point
    cell_hierarchy = []
    cell_keyword = []
    time_point = []
    # create patterns used to extract parameters
    pattern_keyword = r'.index Pc\(.*?\) [\d]+ Pc'
    pattern_time_point = r'^\d+\n$'
    # add matched parameters into list
    for line in power_file.readlines():
        if re.match(pattern_keyword, line):
            tmp = line.split()
            cell_hierarchy.append(tmp[1])
            cell_keyword.append(int(tmp[2]))
        elif re.match(pattern_time_point, line):
            time_point.append(int(line.strip('\n')))
    power_file.seek(0)
    # create the map list between the cell hierarchy (from .out file) and instance_hierarchy (from .dspf or .def file)
    cell_map = []
    for current_num in range(len(instance_hierarchy)):
        tmp = 'Pc(' + str(top_cell) + '/' + str(instance_hierarchy[current_num]) + ')'
        try:
            cell_map.append(cell_hierarchy.index(tmp))
        except ValueError:
            continue

    # print("The amount of the logic cells is ", len(cell_map))
    print("The maximum of the time points is ", time_point[-1])

    time_point = np.asarray(time_point)
    cell_keyword = np.asarray(cell_keyword)
    # create array for all power traces
    clock_period = desired_time_interval-0
    off_clock_cycles = 1
    start_time_point = 0
    power_trace_all = np.full((len(cell_keyword), num_plaintexts, int(clock_period/final_time_divider)), 1e-11, dtype=np.float32)
    # print("The shape of the current trace is ", np.shape(power_trace_all))
    # create pattern used to extract power traces
    pattern_power = r'[\d]+  [\d]+.*'
    tracked_time_point = 0
    num_plaintexts_recorded = 0
    # add matched power values into the array
    for line in power_file.readlines():
        if num_plaintexts_recorded > num_plaintexts:
            break
        if re.match(pattern_time_point, line):
            current_time_point = int(line.strip('\n'))
            tracked_time_point = current_time_point - start_time_point - (clock_period * off_clock_cycles * num_plaintexts_recorded)
            if current_time_point > start_time_point + num_plaintexts * off_clock_cycles * clock_period:
                break
            elif tracked_time_point >= clock_period:
                num_plaintexts_recorded += 1
                tracked_time_point = current_time_point - start_time_point - (clock_period * off_clock_cycles * num_plaintexts_recorded)
        if re.match(pattern_power, line):
            if tracked_time_point < 0:
                continue
            tmp = line.split()
            current_keyword = int(tmp[0]) - 1
            power_trace_all[current_keyword, num_plaintexts_recorded, int(tracked_time_point/final_time_divider)] = float(tmp[1])
    power_file.close()
    # map power traces for required cell
    power_trace_all = power_trace_all[cell_map, :, :]
    # post-process the power trace (replace the 0 value)
    # pool = mp.Pool(mp.cpu_count()-1)
    for current_time_point in range(np.shape(power_trace_all)[2]):
        for plaintext in range(num_plaintexts):
            vertical = np.asarray(power_trace_all[:, plaintext, current_time_point] == 1e-11).nonzero()
            if current_time_point >= 1:
                power_trace_all[vertical, plaintext, current_time_point] = power_trace_all[vertical, plaintext, current_time_point - 1]
            else:
                power_trace_all[vertical, plaintext, current_time_point] = power_trace_all[vertical, plaintext, current_time_point]
    power_trace_all = power_trace_all / power_supply_voltage
    print("The shape of the current trace is ", np.shape(power_trace_all))
    return power_trace_all


def main(top_cell, parasitic_netlist_path, def_path, power_report_path, num_plaintexts, start_time_point,
         desired_time_interval, desired_time_scale, power_report_init_path, power_supply_voltage):
    cell_name, cell_pin_X, cell_pin_Y = get_cell_pin_info(parasitic_netlist_path)
    cell_name, instance_hierarchy = match_cell_info(def_path, cell_name, cell_pin_X, cell_pin_Y)
    final_time_interval, final_time_divider = process_time_interval_use(desired_time_interval, desired_time_scale)
    process_power_report_use(num_plaintexts, start_time_point, final_time_interval, power_report_init_path, power_report_path)
    current_trace = logic_cell_modeling(top_cell, num_plaintexts, final_time_interval, final_time_divider, power_report_path, instance_hierarchy, power_supply_voltage)
    np.save('instance_hierarchy.npy', instance_hierarchy)
    np.save('cell_name.npy', cell_name)
    np.save("current_trace.npy", current_trace)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_cell", type=str, default="aes_top",
                        help="Top cell in the design")
    parser.add_argument("--parasitic_netlist_path", type=str, default="aes_top.dspf",
                        help="Path to the parasitic info file, should end in .dspf")
    parser.add_argument("--def_path", type=str, default="aes_top.def",
                        help="Path to the def file, should end in .def")
    parser.add_argument("--power_report_path", type=str, default="vcd_to_use.out",
                        help="Path to the power report file, should end in .out")
    parser.add_argument("--num_plaintexts", type=int, default=10,
                        help="Number of required plaintexts")
    parser.add_argument("--start_time_point", type=int, default=(1120 + 6880 - 1120 - 6880)*10,
                        help="Start time point in the initial power report")
    parser.add_argument("--desired_time_interval", type=int, default=40,
                        help="Desired time interval to use, timescale 1ns/1ns")
    parser.add_argument("--desired_time_scale", type=float, default=1,
                        help="Desired time scale used in power analysis")
    parser.add_argument("--power_report_init_path", type=str, default="power_reports/vcd_",
                        help="Path to the init power report file, should end in vcd_")
    parser.add_argument("--power_supply_voltage", type=float, default=1.8,
                        help="Supply voltage for logic cells")

    args = parser.parse_args()
    top_cell = args.top_cell
    parasitic_netlist_path = args.parasitic_netlist_path
    def_path = args.def_path
    power_report_path = args.power_report_path
    num_plaintexts = args.num_plaintexts
    start_time_point = args.start_time_point
    desired_time_interval = args.desired_time_interval
    desired_time_scale = args.desired_time_scale
    power_report_init_path = args.power_report_init_path
    power_supply_voltage = args.power_supply_voltage

    start_time = time.time()
    try:
        sys.exit(main(top_cell, parasitic_netlist_path, def_path, power_report_path, num_plaintexts, start_time_point,
                      desired_time_interval, desired_time_scale, power_report_init_path, power_supply_voltage))
    except KeyboardInterrupt:
        sys.exit()
    finally:
        print("Running time:", time.time() - start_time, "seconds")
