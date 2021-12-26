import argparse
import sys
import time


def generate_ptpx_tcl(start_time_point, num_plaintexts, desired_time_interval, ptpx_tcl_path, output_tcl_path, ptpx_run_path):
    '''
    generate ptpx tcl scripts
    :param start_time_point: start time point for power analysis
    :param num_plaintexts: amount of the required plaintexts
    :param desired_time_interval: desired time slice for power analysis
    :param ptpx_tcl_path: path to the template ptpx file
    :param output_tcl_path: path to the output ptpx file
    :param ptpx_run_path: path to the run folder of ptpx
    :return: output ptpx files
    '''

    for current_num in range(num_plaintexts):
        new_ptpx_file = open(output_tcl_path + str(current_num) + '.tcl', 'w', newline='\n')
        ptpx_tcl_file = open(ptpx_tcl_path, "r")
        left_time_interval = start_time_point
        right_time_interval = start_time_point + desired_time_interval
        for line in ptpx_tcl_file.readlines():
            if "read_vcd" in line:
                split_line = line.split()
                tmp = "read_vcd -time { " + str(left_time_interval) + " " + str(right_time_interval) + " } "
                tmp_vcd_path = ptpx_run_path + "vcd_files/tb_main_" + str(current_num) + ".vcd"
                new_line = tmp + tmp_vcd_path + " " + str(split_line[2]) + " " + str(split_line[3])
                new_ptpx_file.write(new_line)
            elif "set_power_analysis_options " in line:
                new_line = line.strip() + "_" + str(current_num) + "\n"
                new_ptpx_file.write(new_line)
            else:
                new_ptpx_file.write(line)
        ptpx_tcl_file.close()
        new_ptpx_file.close()


def main(start_time_point, num_plaintexts, desired_time_interval, ptpx_tcl_path, output_tcl_path, ptpx_run_path):
    generate_ptpx_tcl(start_time_point, num_plaintexts, desired_time_interval, ptpx_tcl_path, output_tcl_path, ptpx_run_path)
    print("Generate ptpx files, finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptpx_tcl_path", type=str, default="ptpx.tcl",
                        help="Path to the template ptpx file, should end in .tcl")
    parser.add_argument("--output_tcl_path", type=str, default="ptpx_files/ptpx_",
                        help="Path to the output ptpx file, should end in ptpx")
    parser.add_argument("--start_time_point", type=int, default=1120 - 1120,
                        help="Start time point for power analysis, timescale 1ns/1ns")
    parser.add_argument("--num_plaintexts", type=int, default=10,
                        help="Amount of the required plaintexts")
    parser.add_argument("--desired_time_interval", type=int, default=40,
                        help="Desired time slice for power analysis, timescale 1ns/1ns")
    parser.add_argument("--ptpx_run_path", type=str, default="/home/Design/AES_NIST_FAB/ptpx/",
                        help="Path to the run folder of ptpx")

    args = parser.parse_args()
    ptpx_tcl_path = args.ptpx_tcl_path
    output_tcl_path = args.output_tcl_path
    start_time_point = args.start_time_point
    num_plaintexts = args.num_plaintexts
    desired_time_interval = args.desired_time_interval
    ptpx_run_path = args.ptpx_run_path

    start_time = time.time()
    try:
        sys.exit(main(start_time_point, num_plaintexts, desired_time_interval, ptpx_tcl_path, output_tcl_path, ptpx_run_path))
    except KeyboardInterrupt:
        sys.exit()
    finally:
        print("Running time:", time.time() - start_time, "seconds")