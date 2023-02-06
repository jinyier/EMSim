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

import h5py
import hspicefile as hsf
import time
import argparse
import sys
import numpy as np


def convert_trx_h5(file_num, directory, metal_layers, outfile):
    """
    Convert .tr spice results into h5 format
    :param file_num: number of required plaintexts
    :param directory: path to result files from Spice simulation
    :param metal_layers: target metal layers
    :param outfile: path to final currents across metal wires
    :return: h5 format currents
    """
    target_wires = []
    complete_results = {}
    for metal in metal_layers:
        temp_file = open(metal, 'r')
        for line in temp_file.readlines():
            try: 
                target_wire = line[15:-2]
                target_wires.append(target_wire.lower())
                complete_results[target_wire.lower()] = []
            except: 
                pass
        temp_file.close()

    for steps in range(file_num):
        spice_results_for_plaintext = hsf.hspice_read(directory + '/final.tr' + str(steps))
        for target_wire in target_wires:
            print(target_wire)
            # print(spice_results_for_plaintext[0][0][2][0]['i('+target_wire])
            complete_results[target_wire].extend(spice_results_for_plaintext[0][0][2][0]['i('+target_wire])

    out_data = np.zeros((len(target_wires), len(complete_results[target_wires[0]])))
    # print(np.shape(out_data))
    
    for wire_num, wire_result in enumerate(complete_results):
        # print(wire_num)
        out_data[wire_num] = complete_results[wire_result]
    
    print('Final h5 files, total cell', np.shape(out_data)[0], 'sample points', np.shape(out_data)[1], ',under', num_plaintexts, 'plaintexts')
    f = h5py.File(outfile, 'w')
    f.create_dataset('spice_sim', data=out_data)
    f.close()


def main(num_plaintexts, spice_result_path, metal_layers, output_file):
    convert_trx_h5(file_num=num_plaintexts, directory=spice_result_path, metal_layers=metal_layers, outfile=output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_plaintexts", type=int, default=10,
                        help="Number of required plaintexts")
    parser.add_argument("--spice_result_path", type=str, default='./trx_to_txt',
                        help="Path to result files from Spice simulation")
    parser.add_argument("--metal_layers", type=str, default=['metal5_probe.txt', 'metal6_probe.txt'],
                        help="Target metal layers")
    parser.add_argument("--output_file", type=str, default='last_binary_run.h5',
                        help="Path to final currents across metal wires")

    args = parser.parse_args()
    num_plaintexts = args.num_plaintexts
    spice_result_path = args.spice_result_path
    metal_layers = args.metal_layers
    output_file = args.output_file

    start_time = time.time()
    try:
        sys.exit(main(num_plaintexts, spice_result_path, metal_layers, output_file))
    except KeyboardInterrupt:
        sys.exit()
    finally:
        print("Running time:", time.time() - start_time, "seconds")