import argparse
import re
import sys
import time


def extract_cell_name(def_path):
    '''
    extract hierarchical components from the def file
    :param def_file: design.def
    :return: extracted components
    '''
    # open and read the def file
    with open(def_path, 'r') as f:
        def_file = f.readlines()

    # create an empty list
    cell_name = []
    # initialize the counter for instance
    instance_count = 0
    start_component = False
    for line in def_file:
        # the end of the component lines to check
        if re.match(r'END COMPONENTS', line):
            start_component = False
        if start_component:
            split_line = line.split(" ")
            # check if the data length > 4
            if len(split_line) > 4:
                cell_name.append(split_line[2])
                instance_count += 1
            else:
                continue
        # the beginning of the component lines to check
        elif re.match(r'COMPONENTS [\d]+', line):
            start_component = True

    cell_name_classify = list(set(cell_name))
    return cell_name_classify


def generate_hcell_file(cell_name_classify, hcell_path):
    '''
    generate hcell file for gate-level parasitic extraction
    :param cell_name_classify: component types in design
    :param hcell_path: path of hcell file
    :return: hcell file
    '''
    hcell_file = open(hcell_path, 'w')
    for line in cell_name_classify:
        tmp = str(line) + '* ' + str(line) + '\n'
        hcell_file .write(tmp)
    hcell_file.close()


def generate_xcell_file(cell_name_classify, xcell_path):
    '''
    generate xcell file for gate-level parasitic extraction
    :param cell_name_classify: component types in design
    :param xcell_path: path of xcell file
    :return: xcell file
    '''
    xcell_file = open(xcell_path, 'w')
    for line in cell_name_classify:
        tmp = str(line) + '* ' + str(line) + ' -I\n'
        xcell_file .write(tmp)
    xcell_file.close()


def generate_lvs_rule(cell_name_classify, lvs_rule_path):
    '''
    generate lvs rule file for gate-level parasitic extraction
    :param cell_name_classify: component types in design
    :param lvs_rule_path: path of lvs rule
    :return: lvs rule
    '''
    lvs_rule_file = open(lvs_rule_path, 'w')
    for line in cell_name_classify:
        tmp = 'LVS BOX ' + str(line) + ' ' + str(line) + '\n'
        lvs_rule_file .write(tmp)
    lvs_rule_file.close()


def main(def_path, hcell_path, xcell_path, lvs_rule_path):
    real_time = time.time()
    cell_name_classify = extract_cell_name(def_path)
    generate_hcell_file(cell_name_classify, hcell_path)
    generate_xcell_file(cell_name_classify, xcell_path)
    generate_lvs_rule(cell_name_classify, lvs_rule_path)
    print("Generate hcell, xcell and lvs rule files, finished.")
    print("Runtime", time.time() - real_time, "seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--def_path", type=str, default="aes_top.def",
                        help="Path to the def file, should end in .def")
    parser.add_argument("--hcell_path", type=str, default="hcell",
                        help="Path to the output hcell file")
    parser.add_argument("--xcell_path", type=str, default="xcell",
                        help="Path to the output xcell file")
    parser.add_argument("--lvs_rule_path", type=str, default="lvs_rule",
                        help="Path to the output lvs_rule file")

    args = parser.parse_args()
    def_path = args.def_path
    hcell_path = args.hcell_path
    xcell_path = args.xcell_path
    lvs_rule_path = args.lvs_rule_path

    start_time = time.time()
    try:
        sys.exit(main(def_path, hcell_path, xcell_path, lvs_rule_path))
    except KeyboardInterrupt:
        sys.exit()
    finally:
        print("Running time:", time.time() - start_time, "seconds")