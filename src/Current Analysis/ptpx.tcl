####################################
#
# set the power analysis mode
#
####################################

set power_enable_analysis TRUE

set power_analysis_mode   time_based

####################################
#
# read and link the gate level netlist
#
####################################

set search_path { .\ 			
                /home/Design/AES_NIST_FAB/ptpx/frontdata\
                                                                   }

set link_library "* fast.db"

read_verilog /home/Design/AES_NIST_FAB/ptpx/frontdata/aes_top.v

current_design aes_top

link

####################################
#
# READ SDC and set transition time or annotate parasitics
#
####################################

read_sdc   /home/Design/AES_NIST/ptpx/frontdata/aes_top.sdc -echo

read_parasitics /home/Design/AES_NIST_FAB/ptpx/frontdata/aes_top.spef

####################################
#
# Check,update,or report timing
#
####################################

check_timing

update_timing

report_timing

####################################
#
# read switching activity file
#
####################################

read_vcd  /home/Design/AES_NIST_FAB/ptpx/tb_main.vcd  -strip_path  tb_main/uut

report_switching_activity -list_not_annotated

####################################

check_power

update_power

set_power_analysis_options -waveform_interval 0.4 -include all_with_leaf -waveform_format out -waveform_output vcd

report_power -hierarchy 

####################################






