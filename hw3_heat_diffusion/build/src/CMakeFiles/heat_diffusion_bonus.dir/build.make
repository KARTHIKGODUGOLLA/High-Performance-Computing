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
CMAKE_SOURCE_DIR = /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw3_heat_diffusion/hw3_heat_diffusion

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw3_heat_diffusion/hw3_heat_diffusion/build

# Include any dependencies generated for this target.
include src/CMakeFiles/heat_diffusion_bonus.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/CMakeFiles/heat_diffusion_bonus.dir/compiler_depend.make

# Include the progress variables for this target.
include src/CMakeFiles/heat_diffusion_bonus.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/heat_diffusion_bonus.dir/flags.make

src/CMakeFiles/heat_diffusion_bonus.dir/heat_diffusion_bonus.cxx.o: src/CMakeFiles/heat_diffusion_bonus.dir/flags.make
src/CMakeFiles/heat_diffusion_bonus.dir/heat_diffusion_bonus.cxx.o: ../src/heat_diffusion_bonus.cxx
src/CMakeFiles/heat_diffusion_bonus.dir/heat_diffusion_bonus.cxx.o: src/CMakeFiles/heat_diffusion_bonus.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw3_heat_diffusion/hw3_heat_diffusion/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/heat_diffusion_bonus.dir/heat_diffusion_bonus.cxx.o"
	cd /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw3_heat_diffusion/hw3_heat_diffusion/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/heat_diffusion_bonus.dir/heat_diffusion_bonus.cxx.o -MF CMakeFiles/heat_diffusion_bonus.dir/heat_diffusion_bonus.cxx.o.d -o CMakeFiles/heat_diffusion_bonus.dir/heat_diffusion_bonus.cxx.o -c /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw3_heat_diffusion/hw3_heat_diffusion/src/heat_diffusion_bonus.cxx

src/CMakeFiles/heat_diffusion_bonus.dir/heat_diffusion_bonus.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/heat_diffusion_bonus.dir/heat_diffusion_bonus.cxx.i"
	cd /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw3_heat_diffusion/hw3_heat_diffusion/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw3_heat_diffusion/hw3_heat_diffusion/src/heat_diffusion_bonus.cxx > CMakeFiles/heat_diffusion_bonus.dir/heat_diffusion_bonus.cxx.i

src/CMakeFiles/heat_diffusion_bonus.dir/heat_diffusion_bonus.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/heat_diffusion_bonus.dir/heat_diffusion_bonus.cxx.s"
	cd /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw3_heat_diffusion/hw3_heat_diffusion/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw3_heat_diffusion/hw3_heat_diffusion/src/heat_diffusion_bonus.cxx -o CMakeFiles/heat_diffusion_bonus.dir/heat_diffusion_bonus.cxx.s

# Object files for target heat_diffusion_bonus
heat_diffusion_bonus_OBJECTS = \
"CMakeFiles/heat_diffusion_bonus.dir/heat_diffusion_bonus.cxx.o"

# External object files for target heat_diffusion_bonus
heat_diffusion_bonus_EXTERNAL_OBJECTS =

bin/heat_diffusion_bonus: src/CMakeFiles/heat_diffusion_bonus.dir/heat_diffusion_bonus.cxx.o
bin/heat_diffusion_bonus: src/CMakeFiles/heat_diffusion_bonus.dir/build.make
bin/heat_diffusion_bonus: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi_cxx.so
bin/heat_diffusion_bonus: /usr/lib/x86_64-linux-gnu/openmpi/lib/libmpi.so
bin/heat_diffusion_bonus: src/CMakeFiles/heat_diffusion_bonus.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw3_heat_diffusion/hw3_heat_diffusion/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ../bin/heat_diffusion_bonus"
	cd /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw3_heat_diffusion/hw3_heat_diffusion/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/heat_diffusion_bonus.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/heat_diffusion_bonus.dir/build: bin/heat_diffusion_bonus
.PHONY : src/CMakeFiles/heat_diffusion_bonus.dir/build

src/CMakeFiles/heat_diffusion_bonus.dir/clean:
	cd /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw3_heat_diffusion/hw3_heat_diffusion/build/src && $(CMAKE_COMMAND) -P CMakeFiles/heat_diffusion_bonus.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/heat_diffusion_bonus.dir/clean

src/CMakeFiles/heat_diffusion_bonus.dir/depend:
	cd /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw3_heat_diffusion/hw3_heat_diffusion/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw3_heat_diffusion/hw3_heat_diffusion /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw3_heat_diffusion/hw3_heat_diffusion/src /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw3_heat_diffusion/hw3_heat_diffusion/build /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw3_heat_diffusion/hw3_heat_diffusion/build/src /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw3_heat_diffusion/hw3_heat_diffusion/build/src/CMakeFiles/heat_diffusion_bonus.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/heat_diffusion_bonus.dir/depend

