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
CMAKE_SOURCE_DIR = /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/build

# Include any dependencies generated for this target.
include test/CMakeFiles/test_get_neighbors.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include test/CMakeFiles/test_get_neighbors.dir/compiler_depend.make

# Include the progress variables for this target.
include test/CMakeFiles/test_get_neighbors.dir/progress.make

# Include the compile flags for this target's objects.
include test/CMakeFiles/test_get_neighbors.dir/flags.make

test/CMakeFiles/test_get_neighbors.dir/test_get_neighbors.cxx.o: test/CMakeFiles/test_get_neighbors.dir/flags.make
test/CMakeFiles/test_get_neighbors.dir/test_get_neighbors.cxx.o: ../test/test_get_neighbors.cxx
test/CMakeFiles/test_get_neighbors.dir/test_get_neighbors.cxx.o: test/CMakeFiles/test_get_neighbors.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object test/CMakeFiles/test_get_neighbors.dir/test_get_neighbors.cxx.o"
	cd /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT test/CMakeFiles/test_get_neighbors.dir/test_get_neighbors.cxx.o -MF CMakeFiles/test_get_neighbors.dir/test_get_neighbors.cxx.o.d -o CMakeFiles/test_get_neighbors.dir/test_get_neighbors.cxx.o -c /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/test/test_get_neighbors.cxx

test/CMakeFiles/test_get_neighbors.dir/test_get_neighbors.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_get_neighbors.dir/test_get_neighbors.cxx.i"
	cd /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/test/test_get_neighbors.cxx > CMakeFiles/test_get_neighbors.dir/test_get_neighbors.cxx.i

test/CMakeFiles/test_get_neighbors.dir/test_get_neighbors.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_get_neighbors.dir/test_get_neighbors.cxx.s"
	cd /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/test/test_get_neighbors.cxx -o CMakeFiles/test_get_neighbors.dir/test_get_neighbors.cxx.s

test/CMakeFiles/test_get_neighbors.dir/test_common.cxx.o: test/CMakeFiles/test_get_neighbors.dir/flags.make
test/CMakeFiles/test_get_neighbors.dir/test_common.cxx.o: ../test/test_common.cxx
test/CMakeFiles/test_get_neighbors.dir/test_common.cxx.o: test/CMakeFiles/test_get_neighbors.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object test/CMakeFiles/test_get_neighbors.dir/test_common.cxx.o"
	cd /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT test/CMakeFiles/test_get_neighbors.dir/test_common.cxx.o -MF CMakeFiles/test_get_neighbors.dir/test_common.cxx.o.d -o CMakeFiles/test_get_neighbors.dir/test_common.cxx.o -c /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/test/test_common.cxx

test/CMakeFiles/test_get_neighbors.dir/test_common.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_get_neighbors.dir/test_common.cxx.i"
	cd /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/test/test_common.cxx > CMakeFiles/test_get_neighbors.dir/test_common.cxx.i

test/CMakeFiles/test_get_neighbors.dir/test_common.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_get_neighbors.dir/test_common.cxx.s"
	cd /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/test/test_common.cxx -o CMakeFiles/test_get_neighbors.dir/test_common.cxx.s

# Object files for target test_get_neighbors
test_get_neighbors_OBJECTS = \
"CMakeFiles/test_get_neighbors.dir/test_get_neighbors.cxx.o" \
"CMakeFiles/test_get_neighbors.dir/test_common.cxx.o"

# External object files for target test_get_neighbors
test_get_neighbors_EXTERNAL_OBJECTS =

bin/test_get_neighbors: test/CMakeFiles/test_get_neighbors.dir/test_get_neighbors.cxx.o
bin/test_get_neighbors: test/CMakeFiles/test_get_neighbors.dir/test_common.cxx.o
bin/test_get_neighbors: test/CMakeFiles/test_get_neighbors.dir/build.make
bin/test_get_neighbors: lib/libsom.a
bin/test_get_neighbors: test/CMakeFiles/test_get_neighbors.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable ../bin/test_get_neighbors"
	cd /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/build/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_get_neighbors.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/CMakeFiles/test_get_neighbors.dir/build: bin/test_get_neighbors
.PHONY : test/CMakeFiles/test_get_neighbors.dir/build

test/CMakeFiles/test_get_neighbors.dir/clean:
	cd /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/build/test && $(CMAKE_COMMAND) -P CMakeFiles/test_get_neighbors.dir/cmake_clean.cmake
.PHONY : test/CMakeFiles/test_get_neighbors.dir/clean

test/CMakeFiles/test_get_neighbors.dir/depend:
	cd /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/test /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/build /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/build/test /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/build/test/CMakeFiles/test_get_neighbors.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/CMakeFiles/test_get_neighbors.dir/depend

