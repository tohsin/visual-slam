# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.23

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
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.23.2/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.23.2/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/emma/dev/visual-slam/cpp_setup

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/emma/dev/visual-slam/cpp_setup/build

# Include any dependencies generated for this target.
include CMakeFiles/helloSLAM.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/helloSLAM.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/helloSLAM.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/helloSLAM.dir/flags.make

CMakeFiles/helloSLAM.dir/main.cpp.o: CMakeFiles/helloSLAM.dir/flags.make
CMakeFiles/helloSLAM.dir/main.cpp.o: ../main.cpp
CMakeFiles/helloSLAM.dir/main.cpp.o: CMakeFiles/helloSLAM.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/emma/dev/visual-slam/cpp_setup/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/helloSLAM.dir/main.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/helloSLAM.dir/main.cpp.o -MF CMakeFiles/helloSLAM.dir/main.cpp.o.d -o CMakeFiles/helloSLAM.dir/main.cpp.o -c /Users/emma/dev/visual-slam/cpp_setup/main.cpp

CMakeFiles/helloSLAM.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/helloSLAM.dir/main.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/emma/dev/visual-slam/cpp_setup/main.cpp > CMakeFiles/helloSLAM.dir/main.cpp.i

CMakeFiles/helloSLAM.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/helloSLAM.dir/main.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/emma/dev/visual-slam/cpp_setup/main.cpp -o CMakeFiles/helloSLAM.dir/main.cpp.s

# Object files for target helloSLAM
helloSLAM_OBJECTS = \
"CMakeFiles/helloSLAM.dir/main.cpp.o"

# External object files for target helloSLAM
helloSLAM_EXTERNAL_OBJECTS =

helloSLAM: CMakeFiles/helloSLAM.dir/main.cpp.o
helloSLAM: CMakeFiles/helloSLAM.dir/build.make
helloSLAM: CMakeFiles/helloSLAM.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/emma/dev/visual-slam/cpp_setup/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable helloSLAM"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/helloSLAM.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/helloSLAM.dir/build: helloSLAM
.PHONY : CMakeFiles/helloSLAM.dir/build

CMakeFiles/helloSLAM.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/helloSLAM.dir/cmake_clean.cmake
.PHONY : CMakeFiles/helloSLAM.dir/clean

CMakeFiles/helloSLAM.dir/depend:
	cd /Users/emma/dev/visual-slam/cpp_setup/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/emma/dev/visual-slam/cpp_setup /Users/emma/dev/visual-slam/cpp_setup /Users/emma/dev/visual-slam/cpp_setup/build /Users/emma/dev/visual-slam/cpp_setup/build /Users/emma/dev/visual-slam/cpp_setup/build/CMakeFiles/helloSLAM.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/helloSLAM.dir/depend

