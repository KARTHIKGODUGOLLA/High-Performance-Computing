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
include src/CMakeFiles/som.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include src/CMakeFiles/som.dir/compiler_depend.make

# Include the progress variables for this target.
include src/CMakeFiles/som.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/som.dir/flags.make

src/CMakeFiles/som.dir/dataset.cxx.o: src/CMakeFiles/som.dir/flags.make
src/CMakeFiles/som.dir/dataset.cxx.o: ../src/dataset.cxx
src/CMakeFiles/som.dir/dataset.cxx.o: src/CMakeFiles/som.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/som.dir/dataset.cxx.o"
	cd /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/som.dir/dataset.cxx.o -MF CMakeFiles/som.dir/dataset.cxx.o.d -o CMakeFiles/som.dir/dataset.cxx.o -c /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/src/dataset.cxx

src/CMakeFiles/som.dir/dataset.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/som.dir/dataset.cxx.i"
	cd /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/src/dataset.cxx > CMakeFiles/som.dir/dataset.cxx.i

src/CMakeFiles/som.dir/dataset.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/som.dir/dataset.cxx.s"
	cd /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/src/dataset.cxx -o CMakeFiles/som.dir/dataset.cxx.s

src/CMakeFiles/som.dir/sample.cxx.o: src/CMakeFiles/som.dir/flags.make
src/CMakeFiles/som.dir/sample.cxx.o: ../src/sample.cxx
src/CMakeFiles/som.dir/sample.cxx.o: src/CMakeFiles/som.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object src/CMakeFiles/som.dir/sample.cxx.o"
	cd /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/som.dir/sample.cxx.o -MF CMakeFiles/som.dir/sample.cxx.o.d -o CMakeFiles/som.dir/sample.cxx.o -c /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/src/sample.cxx

src/CMakeFiles/som.dir/sample.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/som.dir/sample.cxx.i"
	cd /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/src/sample.cxx > CMakeFiles/som.dir/sample.cxx.i

src/CMakeFiles/som.dir/sample.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/som.dir/sample.cxx.s"
	cd /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/src/sample.cxx -o CMakeFiles/som.dir/sample.cxx.s

src/CMakeFiles/som.dir/som.cxx.o: src/CMakeFiles/som.dir/flags.make
src/CMakeFiles/som.dir/som.cxx.o: ../src/som.cxx
src/CMakeFiles/som.dir/som.cxx.o: src/CMakeFiles/som.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object src/CMakeFiles/som.dir/som.cxx.o"
	cd /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/som.dir/som.cxx.o -MF CMakeFiles/som.dir/som.cxx.o.d -o CMakeFiles/som.dir/som.cxx.o -c /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/src/som.cxx

src/CMakeFiles/som.dir/som.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/som.dir/som.cxx.i"
	cd /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/src/som.cxx > CMakeFiles/som.dir/som.cxx.i

src/CMakeFiles/som.dir/som.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/som.dir/som.cxx.s"
	cd /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/src/som.cxx -o CMakeFiles/som.dir/som.cxx.s

src/CMakeFiles/som.dir/cell.cxx.o: src/CMakeFiles/som.dir/flags.make
src/CMakeFiles/som.dir/cell.cxx.o: ../src/cell.cxx
src/CMakeFiles/som.dir/cell.cxx.o: src/CMakeFiles/som.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object src/CMakeFiles/som.dir/cell.cxx.o"
	cd /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT src/CMakeFiles/som.dir/cell.cxx.o -MF CMakeFiles/som.dir/cell.cxx.o.d -o CMakeFiles/som.dir/cell.cxx.o -c /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/src/cell.cxx

src/CMakeFiles/som.dir/cell.cxx.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/som.dir/cell.cxx.i"
	cd /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/src/cell.cxx > CMakeFiles/som.dir/cell.cxx.i

src/CMakeFiles/som.dir/cell.cxx.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/som.dir/cell.cxx.s"
	cd /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/build/src && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/src/cell.cxx -o CMakeFiles/som.dir/cell.cxx.s

# Object files for target som
som_OBJECTS = \
"CMakeFiles/som.dir/dataset.cxx.o" \
"CMakeFiles/som.dir/sample.cxx.o" \
"CMakeFiles/som.dir/som.cxx.o" \
"CMakeFiles/som.dir/cell.cxx.o"

# External object files for target som
som_EXTERNAL_OBJECTS =

lib/libsom.a: src/CMakeFiles/som.dir/dataset.cxx.o
lib/libsom.a: src/CMakeFiles/som.dir/sample.cxx.o
lib/libsom.a: src/CMakeFiles/som.dir/som.cxx.o
lib/libsom.a: src/CMakeFiles/som.dir/cell.cxx.o
lib/libsom.a: src/CMakeFiles/som.dir/build.make
lib/libsom.a: src/CMakeFiles/som.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CXX static library ../lib/libsom.a"
	cd /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/build/src && $(CMAKE_COMMAND) -P CMakeFiles/som.dir/cmake_clean_target.cmake
	cd /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/som.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/som.dir/build: lib/libsom.a
.PHONY : src/CMakeFiles/som.dir/build

src/CMakeFiles/som.dir/clean:
	cd /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/build/src && $(CMAKE_COMMAND) -P CMakeFiles/som.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/som.dir/clean

src/CMakeFiles/som.dir/depend:
	cd /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/src /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/build /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/build/src /mnt/c/Users/Karthik_Godugolla/RIT/HPDS/hw1_soms_redacted/hw1_soms_redacted/build/src/CMakeFiles/som.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/som.dir/depend

