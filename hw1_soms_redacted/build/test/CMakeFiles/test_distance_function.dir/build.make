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
include test/CMakeFiles/test_distance_function.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include test/CMakeFiles/test_distance_function.dir/compiler_depend.make

# Include the progress variables for this target.
include test/CMakeFiles/test_distance_function.dir/progress.make

# Include the compile flags for this target's objects.
include test/CMakeFiles/test_distance_function.dir/flags.make

test/CMakeFiles/test_distance_function.dir/test_distance_function.cxx.o: test/CMakeFiles/test_distance_function.dir/flags.make
test/CMakeFiles/test_distance_function.dir/test_distance_function.cxx.o: ../test/test_distance_function.cxx
test/CMakeFiles/test_distance_function.dir/test_distance_function.cxx.o: test/CMakeFiles/test_distance_function.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object test/CMakeFiles/test_distance_function.dir/test_distance_function.cxx.o"
	cd /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT test/CMakeFiles/test_distance_function.dir/test_distance_function.cxx.o -MF CMakeFiles/test_distance_function.dir/test_distance_function.cxx.o.d -o CMakeFiles/test_distance_function.dir/test_distance_function.cxx.o -c /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/test/test_distance_function.cxx

test/CMakeFiles/test_distance_function.dir/test_distance_function.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_distance_function.dir/test_distance_function.cxx.i"
	cd /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/test/test_distance_function.cxx > CMakeFiles/test_distance_function.dir/test_distance_function.cxx.i

test/CMakeFiles/test_distance_function.dir/test_distance_function.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_distance_function.dir/test_distance_function.cxx.s"
	cd /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/test/test_distance_function.cxx -o CMakeFiles/test_distance_function.dir/test_distance_function.cxx.s

test/CMakeFiles/test_distance_function.dir/test_common.cxx.o: test/CMakeFiles/test_distance_function.dir/flags.make
test/CMakeFiles/test_distance_function.dir/test_common.cxx.o: ../test/test_common.cxx
test/CMakeFiles/test_distance_function.dir/test_common.cxx.o: test/CMakeFiles/test_distance_function.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object test/CMakeFiles/test_distance_function.dir/test_common.cxx.o"
	cd /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT test/CMakeFiles/test_distance_function.dir/test_common.cxx.o -MF CMakeFiles/test_distance_function.dir/test_common.cxx.o.d -o CMakeFiles/test_distance_function.dir/test_common.cxx.o -c /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/test/test_common.cxx

test/CMakeFiles/test_distance_function.dir/test_common.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_distance_function.dir/test_common.cxx.i"
	cd /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/test/test_common.cxx > CMakeFiles/test_distance_function.dir/test_common.cxx.i

test/CMakeFiles/test_distance_function.dir/test_common.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_distance_function.dir/test_common.cxx.s"
	cd /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/build/test && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/test/test_common.cxx -o CMakeFiles/test_distance_function.dir/test_common.cxx.s

# Object files for target test_distance_function
test_distance_function_OBJECTS = \
"CMakeFiles/test_distance_function.dir/test_distance_function.cxx.o" \
"CMakeFiles/test_distance_function.dir/test_common.cxx.o"

# External object files for target test_distance_function
test_distance_function_EXTERNAL_OBJECTS =

bin/test_distance_function: test/CMakeFiles/test_distance_function.dir/test_distance_function.cxx.o
bin/test_distance_function: test/CMakeFiles/test_distance_function.dir/test_common.cxx.o
bin/test_distance_function: test/CMakeFiles/test_distance_function.dir/build.make
bin/test_distance_function: lib/libsom.a
bin/test_distance_function: test/CMakeFiles/test_distance_function.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable ../bin/test_distance_function"
	cd /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/build/test && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_distance_function.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/CMakeFiles/test_distance_function.dir/build: bin/test_distance_function
.PHONY : test/CMakeFiles/test_distance_function.dir/build

test/CMakeFiles/test_distance_function.dir/clean:
	cd /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/build/test && $(CMAKE_COMMAND) -P CMakeFiles/test_distance_function.dir/cmake_clean.cmake
.PHONY : test/CMakeFiles/test_distance_function.dir/clean

test/CMakeFiles/test_distance_function.dir/depend:
	cd /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/test /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/build /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/build/test /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/build/test/CMakeFiles/test_distance_function.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/CMakeFiles/test_distance_function.dir/depend

