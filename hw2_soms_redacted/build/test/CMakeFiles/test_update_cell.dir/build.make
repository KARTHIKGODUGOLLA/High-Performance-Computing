# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw2_soms_redacted

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw2_soms_redacted/build

# Include any dependencies generated for this target.
include test/CMakeFiles/test_update_cell.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include test/CMakeFiles/test_update_cell.dir/compiler_depend.make

# Include the progress variables for this target.
include test/CMakeFiles/test_update_cell.dir/progress.make

# Include the compile flags for this target's objects.
include test/CMakeFiles/test_update_cell.dir/flags.make

test/CMakeFiles/test_update_cell.dir/test_update_cell.cxx.o: test/CMakeFiles/test_update_cell.dir/flags.make
test/CMakeFiles/test_update_cell.dir/test_update_cell.cxx.o: ../test/test_update_cell.cxx
test/CMakeFiles/test_update_cell.dir/test_update_cell.cxx.o: test/CMakeFiles/test_update_cell.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw2_soms_redacted/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object test/CMakeFiles/test_update_cell.dir/test_update_cell.cxx.o"
	cd /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw2_soms_redacted/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT test/CMakeFiles/test_update_cell.dir/test_update_cell.cxx.o -MF CMakeFiles/test_update_cell.dir/test_update_cell.cxx.o.d -o CMakeFiles/test_update_cell.dir/test_update_cell.cxx.o -c /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw2_soms_redacted/test/test_update_cell.cxx

test/CMakeFiles/test_update_cell.dir/test_update_cell.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_update_cell.dir/test_update_cell.cxx.i"
	cd /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw2_soms_redacted/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw2_soms_redacted/test/test_update_cell.cxx > CMakeFiles/test_update_cell.dir/test_update_cell.cxx.i

test/CMakeFiles/test_update_cell.dir/test_update_cell.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_update_cell.dir/test_update_cell.cxx.s"
	cd /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw2_soms_redacted/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw2_soms_redacted/test/test_update_cell.cxx -o CMakeFiles/test_update_cell.dir/test_update_cell.cxx.s

test/CMakeFiles/test_update_cell.dir/test_common.cxx.o: test/CMakeFiles/test_update_cell.dir/flags.make
test/CMakeFiles/test_update_cell.dir/test_common.cxx.o: ../test/test_common.cxx
test/CMakeFiles/test_update_cell.dir/test_common.cxx.o: test/CMakeFiles/test_update_cell.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw2_soms_redacted/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object test/CMakeFiles/test_update_cell.dir/test_common.cxx.o"
	cd /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw2_soms_redacted/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT test/CMakeFiles/test_update_cell.dir/test_common.cxx.o -MF CMakeFiles/test_update_cell.dir/test_common.cxx.o.d -o CMakeFiles/test_update_cell.dir/test_common.cxx.o -c /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw2_soms_redacted/test/test_common.cxx

test/CMakeFiles/test_update_cell.dir/test_common.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_update_cell.dir/test_common.cxx.i"
	cd /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw2_soms_redacted/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw2_soms_redacted/test/test_common.cxx > CMakeFiles/test_update_cell.dir/test_common.cxx.i

test/CMakeFiles/test_update_cell.dir/test_common.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_update_cell.dir/test_common.cxx.s"
	cd /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw2_soms_redacted/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw2_soms_redacted/test/test_common.cxx -o CMakeFiles/test_update_cell.dir/test_common.cxx.s

# Object files for target test_update_cell
test_update_cell_OBJECTS = \
"CMakeFiles/test_update_cell.dir/test_update_cell.cxx.o" \
"CMakeFiles/test_update_cell.dir/test_common.cxx.o"

# External object files for target test_update_cell
test_update_cell_EXTERNAL_OBJECTS =

bin/test_update_cell: test/CMakeFiles/test_update_cell.dir/test_update_cell.cxx.o
bin/test_update_cell: test/CMakeFiles/test_update_cell.dir/test_common.cxx.o
bin/test_update_cell: test/CMakeFiles/test_update_cell.dir/build.make
bin/test_update_cell: lib/libsom.a
bin/test_update_cell: test/CMakeFiles/test_update_cell.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw2_soms_redacted/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable ../bin/test_update_cell"
	cd /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw2_soms_redacted/build/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_update_cell.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/CMakeFiles/test_update_cell.dir/build: bin/test_update_cell
.PHONY : test/CMakeFiles/test_update_cell.dir/build

test/CMakeFiles/test_update_cell.dir/clean:
	cd /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw2_soms_redacted/build/test && $(CMAKE_COMMAND) -P CMakeFiles/test_update_cell.dir/cmake_clean.cmake
.PHONY : test/CMakeFiles/test_update_cell.dir/clean

test/CMakeFiles/test_update_cell.dir/depend:
	cd /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw2_soms_redacted/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw2_soms_redacted /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw2_soms_redacted/test /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw2_soms_redacted/build /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw2_soms_redacted/build/test /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw2_soms_redacted/build/test/CMakeFiles/test_update_cell.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/CMakeFiles/test_update_cell.dir/depend

