import sys
import argparse
import re
import csv
import time


def read_metal_layer(parasitic_netlist_path, output_path, target_layer, power_port):
    """
    Creates probe, detail, and em files for a single metal layer
    :param parasitic_netlist_path: the path to the parasitic netlist used to create the file
    :param output_path: the standard output file name. The end changed to em, probe, or detail
    :param target_layer: the target metal layer
    :param power_port: VDD and VSS port in physical layout
    :return: the number of sources/lines of interest divided by 2 in the netlist
    """
    assert target_layer[:5] == "metal", "Target layer must be a metal layer"
    try:
        parasitic_netlist_file = open(parasitic_netlist_path, "r")
        probe_out_file = open(output_path + "_probe.txt", "w+", newline="")
        em_out_file = open(output_path + "_em.txt", "w+", newline="")
        detail_file = open(output_path + "_detail.txt", "w+")

        # skip first few lines
        target_line_found = False
        # we do one of two different tasks depending on what number (even/odd) line this is
        new_section = True

        # first task target data
        name = []
        width = []
        length = []
        type_1_count = 0
        # second task target data
        X1 = []
        Y1 = []
        X2 = []
        Y2 = []
        type_2_count = 0

        for line in parasitic_netlist_file.readlines():
            if re.search('(.*\/[\d]+) .*w=(.*) .*l=(.*) .*'+target_layer, line):
                # save the raw matched text to file for error checking
                detail_file.write(line)
                # split on spaces
                split_line = line.split(" ")
                # check that this is split as expected
                assert len(split_line) >= 7, "Line should be have 7 categories, got  " + str(len(split_line))
                if len(split_line) > 7:
                    extras = split_line[7:]
                # first comes the name
                temp = split_line[0]
                # we also want to create the probe command at this time
                if (power_port[1] in temp or power_port[0] in temp) is True and (split_line[4][3:] == '8e-07') is False:
                    print(temp)
                    temp = ".PROBE TRAN i(*"+temp+")\n"
                    probe_out_file.write(temp)
                name.append(temp)
                # next we want the width
                temp = split_line[4]
                assert temp[:3] == "$w=", "Expected width, found: " + temp
                width.append(temp[3:])
                # last is the length
                temp = split_line[5]
                assert temp[:3] == "$l=", "Expected length, found: " + temp
                length.append(temp[3:])
                # no longer a type 1
                new_section = False
                # we have also found the first target line
                target_line_found = True
                # increment type count
                type_1_count += 1
            elif re.search('\+.*X=(.*) .*Y=(.*) .*X2=(.*) .*Y2=(.*)', line) and new_section is False:
                # save the raw matched text to file for error checking
                detail_file.write(line)
                # split on spaces
                split_line = line.split(" ")
                # check that this is split as expected
                assert len(split_line) <= 6, "Line should be have at most 6 categories, got  " + str(len(split_line)) \
                                             + " " + str(split_line)
                if len(split_line) < 6:
                    flat_list = []
                    for item in extras:
                        flat_list.append(item)
                    for item in split_line:
                        flat_list.append(item)
                    assert len(flat_list) == 6, "Line categories do not align even after matching efforts"
                    split_line = flat_list
                # update values
                temp = split_line[2]
                assert temp[:3] == "$X=", "Expected X1, found: " + temp
                X1.append(temp[3:])

                temp = split_line[3]
                assert temp[:3] == "$Y=", "Expected Y1, found: " + temp
                Y1.append(temp[3:])

                temp = split_line[4]
                assert temp[:4] == "$X2=", "Expected X2, found: " + temp
                X2.append(temp[4:])

                temp = split_line[5]
                assert temp[:4] == "$Y2=", "Expected Y2, found: " + temp
                # skip the newline character
                Y2.append(temp[4:-1])
                # we are no longer in a type 1
                new_section = True
                # increment type count
                type_2_count += 1
            else:
                new_section = True
        # make sure sections match
        assert type_2_count == type_1_count, "Section counts do not align"

        probe_count = 0
        out_writer = csv.writer(em_out_file)
        for item in range(type_1_count):
            if (power_port[1] in name[item] or power_port[0] in name[item]) is True and (width[item] == '8e-07') is False:
                out = [width[item], length[item], X1[item], Y1[item], X2[item], Y2[item]]
                out_writer.writerow(out)
                probe_count += 1

        return probe_count

    finally:
        parasitic_netlist_file.close()
        em_out_file.close()
        detail_file.close()


def create_sp_file(template_path, output_path, parasitic_netlist_path, metal_files):
    """
    Creates Spice script for wire current analysis
    :param template_path: path to the default sp template file
    :param parasitic_netlist_path: path to the parasitic netlist used to create the file
    :param output_path: path to write the final sp to
    :param metal_files: the target metal layer
    :return: Spice script for wire current analysis
    """
    try:
        template_file = open(template_path, "r")
        final_file = open(output_path[:-3] + ".sp", "w+")
        parasitic_netlist_file = open(parasitic_netlist_path, "r")

        for line in template_file.readlines():
            final_file.write(line)
        template_file.close()

        for path in metal_files:
            metal_file = open(path + "_probe.txt", "r")
            for line in metal_file.readlines():
                final_file.write(line)
            metal_file.close()

        off_chip_pdn = "\nR1 VDD1 VDD 0.5\n"
        final_file.write(off_chip_pdn)

        for line in parasitic_netlist_file.readlines():
            final_file.write(line)
        out = "\n.end\n"
        final_file.write(out)

    finally:
        template_file.close()
        final_file.close()
        parasitic_netlist_file.close()


def main(parasitic_netlist_path, metal_layers, power_port, template_file, output_path):
    for metal in metal_layers:
        print('Layer', metal, 'wire count', read_metal_layer(parasitic_netlist_path=parasitic_netlist_path, output_path=metal, target_layer=metal, power_port=power_port))
    create_sp_file(template_path=template_file, output_path=output_path, parasitic_netlist_path=parasitic_netlist_path, metal_files=metal_layers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parasitic_netlist", type=str, default="aes_top_lc_as_currents.dpsf",
                        help="Path to the parasitic netlist, should end in .dspf")
    parser.add_argument("--metal_layers", type=str, default=["metal5", "metal6"],
                        help="target metal layers")
    parser.add_argument("--power_port", type=str, default=['VDD', 'VSS'],
                        help="VDD and VSS port in physical layout")
    parser.add_argument("--template_file", type=str, default="template.txt",
                        help="Path to the default sp template file")
    parser.add_argument("--output_file", type=str, default="final.sp",
                        help="Path to write the final sp to")

    args = parser.parse_args()
    parasitic_netlist_path = args.parasitic_netlist
    power_port = args.power_port
    metal_layers = args.metal_layers
    template_path = args.template_file
    output_path = args.output_file

    start_time = time.time()
    try:
        sys.exit(main(parasitic_netlist_path=parasitic_netlist_path, metal_layers=metal_layers, power_port=power_port, template_file=template_path, output_path=output_path))
    except KeyboardInterrupt:
        sys.exit()
    finally:
        print("Running time:", time.time() - start_time, "seconds")