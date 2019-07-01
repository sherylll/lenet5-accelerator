#################
#    HLS4ML
#################
array set opt {
  csim   1
  synth  1
  cosim  1
  export 1
}

foreach arg $::argv {
  foreach o [lsort [array names opt]] {
    regexp "$o +(\\w+)" $arg unused opt($o)
  }
}

open_project -reset hls_prj
set_top lenet5
add_files firmware/lenet5.cpp -cflags "-I[file normalize ./nnet_utils] -std=c++0x"
add_files -tb lenet5_test.cpp -cflags "-I[file normalize ./nnet_utils] -std=c++0x"
add_files -tb firmware/weights
add_files -tb test_images
#add_files -tb tb_data
open_solution -reset "solution1"
catch {config_array_partition -maximum_size 4096}
#set_part {xcku095-ffvb2104-1-c}
set_part {xcku115-flvb2014-2-e}
create_clock -period 10 -name default

if {$opt(csim)} {
  puts "***** C SIMULATION *****"
  csim_design
}

if {$opt(synth)} {
  puts "***** C/RTL SYNTHESIS *****"
  csynth_design
  if {$opt(cosim)} {
    puts "***** C/RTL SIMULATION *****"
    cosim_design -trace_level all
  }
  if {$opt(export)} {
    puts "***** EXPORT IP *****"
    export_design -format ip_catalog
  }
}

exit
