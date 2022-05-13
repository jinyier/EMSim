import numpy as np
import sys
import argparse
import time


def LC_inserter(lc_current_traces_filename, cell_name_path, power_port, mark_net, netlist_filename, desired_time_scale):
    """
    Replace logic cells with predefined current sources
    :param lc_current_traces_filename: path to the simulated logic cell currents
    :param power_port: VDD and VSS port in physical layout
    :param mark_net: Start and break name in the .dspf file
    :param netlist_filename: Path to the parasitic netlist
    :param desired_time_scale: Desired time scale used in power analysis
    :return: current source for Spice simulation
    """
    try:
        vdd_port = power_port[0]
        vss_port = power_port[1]
        start_net = mark_net[0]
        break_net = mark_net[1]
        netlist_file = open(netlist_filename, "r")
        new_netlist_file = open(netlist_filename[:-5] + '_lc_as_currents.dpsf', 'w+', newline='\n')
        lc_current_traces = np.load(lc_current_traces_filename)
        print("The shape of the current trace is ", lc_current_traces.shape)
        desired_line = False
        start = False
        source_set = set()
        for line in netlist_file.readlines():
            if '*|NET ' + start_net in line:
                start = True
                # print(start)
            elif '*|NET ' + break_net in line:
                break
            elif start:
                if line[0] in ['*', '\n']:
                    desired_line = False
                    
                elif vdd_port in line or vss_port in line or '.' == line[0]:
                    if line == '.ends\n':
                        continue
                    new_line = ""
                    for index in range(len(line)):
                        if line[index:index+3] in [vdd_port, vss_port]:
                            if line[index-1] == 'r' or line[index-1] == 'c':
                                new_line += line[index]
                            elif line[index-1:index+4] in [' ' + vdd_port + ' ', ' ' + vss_port + ' ']:
                                new_line += line[index]
                            else:
                                new_line += 'r' + line[index]
                        elif line[index] in ['x', 'X']:
                            new_line += 'I'
                        else:
                            new_line += line[index]

                    split = new_line.split(" ")
                    for element in split:
                        if element[0] == 'I':
                            try:
                                temp, connection = element.split(':')
                            except:
                                continue
                            if connection not in ['r' + vdd_port, 'r' + vss_port]:
                                continue
                            source_set.add(temp.upper())
                    if new_line[0] == 'r' or new_line[0] == 'c':
                        new_netlist_file.write(new_line)
                        desired_line = True
                        
                elif desired_line:
                    if line[0] in ['+', '.']:
                        if line == '.ends\n':
                            break
                        new_netlist_file.write(line)
                    desired_line = False
                    
                else:
                    continue

        cell_name_array = np.load(cell_name_path)
        cell_name_reorder = cell_name_array.tolist()
        for index, cell in enumerate(cell_name_reorder):
            cell_name_reorder[index] = cell.replace('X', 'I')
        assert set(source_set) == set(cell_name_reorder), "Error: please check the instance reordering"

        print('Read parasitic file and current source, finished.')
        for current_plaintext in range(np.shape(lc_current_traces)[1]):
            tmp_lc_current_traces = lc_current_traces[:, current_plaintext, :]
            if current_plaintext == 0:
                for index, cell in enumerate(cell_name_reorder):
                    chars_per_line = 0
                    new_line = cell + " " + cell + ':r' + vdd_port + ' ' + cell + ':r' + vss_port + ' PWL ( '
                    for time, current in enumerate(tmp_lc_current_traces[index][: -1]):
                        if chars_per_line >= 115:
                            new_line += '\n+'
                            chars_per_line = 0
                        next_point = str(round(time * desired_time_scale, 1)) + "ns " + str(current) + " "
                        chars_per_line += len(next_point)
                        new_line += next_point
                    new_netlist_file.write(new_line)
                    new_netlist_file.write(')\n\n')
                print("Current plaintext index", current_plaintext, 'with', len(cell_name_reorder), "logic cells")
            elif current_plaintext > 0:
                new_netlist_file.write('\n.alter\n')
                last_lc_current_traces = lc_current_traces[:, current_plaintext - 1, :]
                for index, cell in enumerate(cell_name_reorder):
                    if (np.array(tmp_lc_current_traces[index]) == np.array(last_lc_current_traces[index])).all():
                        continue
                    else:
                        chars_per_line = 0
                        new_line = cell + " " + cell + ':r' + vdd_port + ' ' + cell + ':r' + vss_port + ' PWL ( '
                        for time, current in enumerate(tmp_lc_current_traces[index][: -1]):
                            if chars_per_line >= 115:
                                new_line += '\n+'
                                chars_per_line = 0
                            next_point = str(round(time * desired_time_scale,1)) + "ns " + str(current) + " "
                            chars_per_line += len(next_point)
                            new_line += next_point
                        new_netlist_file.write(new_line)
                        new_netlist_file.write(')\n\n')
                print("Current plaintext index", current_plaintext, 'with', len(cell_name_reorder), "logic cells")

    except Exception as e:
        print(e)

    finally:
        netlist_file.close()
        new_netlist_file.close()


def main(lc_currents_path, cell_name_path, power_port, mark_net, netlist_path, desired_time_scale):
    LC_inserter(lc_currents_path, cell_name_path, power_port, mark_net, netlist_path, desired_time_scale)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lc_currents_path", type=str, default="current_trace.npy",
                        help="Path to the simulated logic cell currents")
    parser.add_argument("--cell_name_path", type=str, default="cell_name.npy",
                        help="Path to the Reorder cell name from dspf file")
    parser.add_argument("--power_port", type=str, default=['VDD', 'VSS'],
                        help="VDD and VSS port in physical layout")
    parser.add_argument("--mark_net", type=str, default=['VDD ', '32 '],
                        help="Start and break net in the .dspf file, should end in a blank")
    parser.add_argument("--netlist_path", type=str, default="aes_top.dspf",
                        help="Path to the parasitic netlist, should end in .dspf")
    parser.add_argument("--desired_time_scale", type=float, default=0.4,
                        help="Desired time scale used in power analysis")

    args = parser.parse_args()
    netlist_path = args.netlist_path
    cell_name_path = args.cell_name_path
    power_port = args.power_port
    mark_net = args.mark_net
    lc_currents_path = args.lc_currents_path
    desired_time_scale = args.desired_time_scale

    start_time = time.time()
    try:
        sys.exit((main(lc_currents_path, cell_name_path, power_port, mark_net, netlist_path, desired_time_scale)))
    except KeyboardInterrupt:
        sys.exit()
    except Exception as e:
        print(e)
        sys.exit()
    finally:
        print("Running time:", time.time() - start_time, "seconds")
