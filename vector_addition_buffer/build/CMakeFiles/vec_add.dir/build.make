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
CMAKE_SOURCE_DIR = /home/lcrisci/workspace/librert/opencl/vector_addition

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/lcrisci/workspace/librert/opencl/vector_addition/build

# Include any dependencies generated for this target.
include CMakeFiles/vec_add.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/vec_add.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/vec_add.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/vec_add.dir/flags.make

CMakeFiles/vec_add.dir/vec_add.cpp.o: CMakeFiles/vec_add.dir/flags.make
CMakeFiles/vec_add.dir/vec_add.cpp.o: ../vec_add.cpp
CMakeFiles/vec_add.dir/vec_add.cpp.o: CMakeFiles/vec_add.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lcrisci/workspace/librert/opencl/vector_addition/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/vec_add.dir/vec_add.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/vec_add.dir/vec_add.cpp.o -MF CMakeFiles/vec_add.dir/vec_add.cpp.o.d -o CMakeFiles/vec_add.dir/vec_add.cpp.o -c /home/lcrisci/workspace/librert/opencl/vector_addition/vec_add.cpp

CMakeFiles/vec_add.dir/vec_add.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vec_add.dir/vec_add.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lcrisci/workspace/librert/opencl/vector_addition/vec_add.cpp > CMakeFiles/vec_add.dir/vec_add.cpp.i

CMakeFiles/vec_add.dir/vec_add.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vec_add.dir/vec_add.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lcrisci/workspace/librert/opencl/vector_addition/vec_add.cpp -o CMakeFiles/vec_add.dir/vec_add.cpp.s

# Object files for target vec_add
vec_add_OBJECTS = \
"CMakeFiles/vec_add.dir/vec_add.cpp.o"

# External object files for target vec_add
vec_add_EXTERNAL_OBJECTS =

vec_add: CMakeFiles/vec_add.dir/vec_add.cpp.o
vec_add: CMakeFiles/vec_add.dir/build.make
vec_add: /usr/lib/x86_64-linux-gnu/libOpenCL.so
vec_add: CMakeFiles/vec_add.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/lcrisci/workspace/librert/opencl/vector_addition/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable vec_add"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/vec_add.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/vec_add.dir/build: vec_add
.PHONY : CMakeFiles/vec_add.dir/build

CMakeFiles/vec_add.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/vec_add.dir/cmake_clean.cmake
.PHONY : CMakeFiles/vec_add.dir/clean

CMakeFiles/vec_add.dir/depend:
	cd /home/lcrisci/workspace/librert/opencl/vector_addition/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lcrisci/workspace/librert/opencl/vector_addition /home/lcrisci/workspace/librert/opencl/vector_addition /home/lcrisci/workspace/librert/opencl/vector_addition/build /home/lcrisci/workspace/librert/opencl/vector_addition/build /home/lcrisci/workspace/librert/opencl/vector_addition/build/CMakeFiles/vec_add.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/vec_add.dir/depend

