# CMake generated Testfile for 
# Source directory: /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted
# Build directory: /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test_sample "bin/test_sample")
set_tests_properties(test_sample PROPERTIES  _BACKTRACE_TRIPLES "/mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/CMakeLists.txt;30;add_test;/mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/CMakeLists.txt;0;")
add_test(test_database_statistics "bin/test_database_statistics")
set_tests_properties(test_database_statistics PROPERTIES  _BACKTRACE_TRIPLES "/mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/CMakeLists.txt;32;add_test;/mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/CMakeLists.txt;0;")
add_test(test_database_normalize "bin/test_database_normalize")
set_tests_properties(test_database_normalize PROPERTIES  _BACKTRACE_TRIPLES "/mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/CMakeLists.txt;33;add_test;/mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/CMakeLists.txt;0;")
add_test(test_distance_function "bin/test_distance_function")
set_tests_properties(test_distance_function PROPERTIES  _BACKTRACE_TRIPLES "/mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/CMakeLists.txt;35;add_test;/mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/CMakeLists.txt;0;")
add_test(test_get_bmu "bin/test_get_bmu")
set_tests_properties(test_get_bmu PROPERTIES  _BACKTRACE_TRIPLES "/mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/CMakeLists.txt;36;add_test;/mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/CMakeLists.txt;0;")
add_test(test_neighborhood_function "bin/test_neighborhood_function")
set_tests_properties(test_neighborhood_function PROPERTIES  _BACKTRACE_TRIPLES "/mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/CMakeLists.txt;37;add_test;/mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/CMakeLists.txt;0;")
add_test(test_get_neighbors "bin/test_get_neighbors")
set_tests_properties(test_get_neighbors PROPERTIES  _BACKTRACE_TRIPLES "/mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/CMakeLists.txt;38;add_test;/mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/CMakeLists.txt;0;")
add_test(test_update_cell "bin/test_update_cell")
set_tests_properties(test_update_cell PROPERTIES  _BACKTRACE_TRIPLES "/mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/CMakeLists.txt;39;add_test;/mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/CMakeLists.txt;0;")
subdirs("src")
subdirs("test")
