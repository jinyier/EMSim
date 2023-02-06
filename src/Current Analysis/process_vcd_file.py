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
import time


def generate_header_file(vcd_init_path, header_path):
    """
    generate header file for cutting off vcd files
    :param vcd_init_path: path to the init vcd file
    :param header_path: path to the header vcd file
    :return: the header vcd file
    """
    pattern_time = r'^#\d+\n$'
    vcd_init_file = open(vcd_init_path, "r")
    header_file = open(header_path, 'w', newline='\n')
    for line in vcd_init_file.readlines():
        match_time = re.match(pattern_time, line, flags=0)
        if match_time:
            match_line = match_time.group().strip()
            # print(match_line)
            current_time = int(match_line[1:])
            if current_time > 0:
                break
            else:
                header_file.write(line)
        else:
            header_file.write(line)
    header_file.close()


def process_vcd_file(start_time_point, num_plaintexts, desired_time_interval, off_time_interval, vcd_init_path, vcd_final_path, header_path):
    """
    cut off vcd files for power analysis, primetime px
    :param start_time_point: the start time point for power analysis
    :param num_plaintexts: amount of the required plaintexts
    :param desired_time_interval: the desired time slice for power analysis
    :param off_time_interval: the time interval between two-times power analysis
    :param vcd_init_path: path to the init vcd file
    :param vcd_final_path: path to the output vcd file
    :param header_path: path to the header vcd file
    :return: cut-off vcd files
    """

    start = False
    pattern_time = r'^#\d+\n$'
    current_num = 0
    vcd_init_file = open(vcd_init_path, "r")
    new_vcd_file = open(vcd_final_path + str(current_num) + '.vcd', 'w', newline='\n')
    header_file = open(header_path, 'r')
    new_vcd_file.write(header_file.read())
    left_time_interval = start_time_point + off_time_interval * current_num
    right_time_interval = start_time_point + desired_time_interval + off_time_interval * current_num

    for line in vcd_init_file.readlines():
        match_time = re.match(pattern_time, line, flags=0)
        if current_num == num_plaintexts:
            break
        elif match_time:
            match_line = match_time.group().strip()
            current_time = int(match_line[1:])
            if current_time > right_time_interval:
                start = False
                new_vcd_file.close()
                current_num += 1
                left_time_interval = start_time_point + off_time_interval * current_num
                right_time_interval = start_time_point + desired_time_interval + off_time_interval * current_num
                print('Current loop', current_num, 'Time interval (', left_time_interval, ',', right_time_interval, ')')
                new_vcd_file = open(vcd_final_path + str(current_num) + '.vcd', 'w', newline='\n')
                header_file = open(header_path, 'r')
                new_vcd_file.write(header_file.read())
                if current_time == left_time_interval:
                    start = True
                    tmp_line = '#' + str(current_time - left_time_interval) + '\n'
                    new_vcd_file.write(tmp_line)
            elif current_time >= left_time_interval:
                start = True
                tmp_line = '#' + str(current_time - left_time_interval) + '\n'
                new_vcd_file.write(tmp_line)
        elif start:
            new_vcd_file.write(line)


def main(start_time_point, num_plaintexts, desired_time_interval, off_time_interval, vcd_init_path, vcd_final_path, header_path):
    generate_header_file(vcd_init_path, header_path)
    process_vcd_file(start_time_point, num_plaintexts, desired_time_interval, off_time_interval, vcd_init_path, vcd_final_path, header_path)
    print("Cut off vcd files, finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vcd_init_path", type=str, default="tb_main.vcd",
                        help="Path to the init vcd file, should end in .vcd")
    parser.add_argument("--vcd_final_path", type=str, default="vcd_files/tb_main_",
                        help="Path to the output vcd file")
    parser.add_argument("--header_path", type=str, default="vcd_files/header.txt",
                        help="Path to the header vcd file")
    parser.add_argument("--start_time_point", type=int, default=1120000,
                        help="Start time point for power analysis, timescale 1ns/1ps")
    parser.add_argument("--num_plaintexts", type=int, default=10,
                        help="Amount of the required plaintexts")
    parser.add_argument("--desired_time_interval", type=int, default=40000,
                        help="Desired time slice for power analysis, timescale 1ns/1ps")
    parser.add_argument("--off_time_interval", type=int, default=6880000,
                        help="Time interval between two-times power analysis, timescale 1ns/1ps")

    args = parser.parse_args()
    vcd_init_path = args.vcd_init_path
    vcd_final_path = args.vcd_final_path
    header_path = args.header_path
    start_time_point = args.start_time_point
    num_plaintexts = args.num_plaintexts
    desired_time_interval = args.desired_time_interval
    off_time_interval = args.off_time_interval

    start_time = time.time()
    try:
        sys.exit(main(start_time_point, num_plaintexts, desired_time_interval, off_time_interval, vcd_init_path,
                     vcd_final_path, header_path))
    except KeyboardInterrupt:
        sys.exit()
    finally:
        print("Running time:", time.time() - start_time, "seconds")
