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
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.22.2/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.22.2/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/bobschreiner/Desktop/ETH/Semester4/Lectures/IML/introML

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/bobschreiner/Desktop/ETH/Semester4/Lectures/IML/introML/build_gcc

# Include any dependencies generated for this target.
include homeworks/task1a/CMakeFiles/task1a_cpp.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include homeworks/task1a/CMakeFiles/task1a_cpp.dir/compiler_depend.make

# Include the progress variables for this target.
include homeworks/task1a/CMakeFiles/task1a_cpp.dir/progress.make

# Include the compile flags for this target's objects.
include homeworks/task1a/CMakeFiles/task1a_cpp.dir/flags.make

homeworks/task1a/CMakeFiles/task1a_cpp.dir/cpp/main.cc.o: homeworks/task1a/CMakeFiles/task1a_cpp.dir/flags.make
homeworks/task1a/CMakeFiles/task1a_cpp.dir/cpp/main.cc.o: ../homeworks/task1a/cpp/main.cc
homeworks/task1a/CMakeFiles/task1a_cpp.dir/cpp/main.cc.o: homeworks/task1a/CMakeFiles/task1a_cpp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/bobschreiner/Desktop/ETH/Semester4/Lectures/IML/introML/build_gcc/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object homeworks/task1a/CMakeFiles/task1a_cpp.dir/cpp/main.cc.o"
	cd /Users/bobschreiner/Desktop/ETH/Semester4/Lectures/IML/introML/build_gcc/homeworks/task1a && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT homeworks/task1a/CMakeFiles/task1a_cpp.dir/cpp/main.cc.o -MF CMakeFiles/task1a_cpp.dir/cpp/main.cc.o.d -o CMakeFiles/task1a_cpp.dir/cpp/main.cc.o -c /Users/bobschreiner/Desktop/ETH/Semester4/Lectures/IML/introML/homeworks/task1a/cpp/main.cc

homeworks/task1a/CMakeFiles/task1a_cpp.dir/cpp/main.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/task1a_cpp.dir/cpp/main.cc.i"
	cd /Users/bobschreiner/Desktop/ETH/Semester4/Lectures/IML/introML/build_gcc/homeworks/task1a && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/bobschreiner/Desktop/ETH/Semester4/Lectures/IML/introML/homeworks/task1a/cpp/main.cc > CMakeFiles/task1a_cpp.dir/cpp/main.cc.i

homeworks/task1a/CMakeFiles/task1a_cpp.dir/cpp/main.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/task1a_cpp.dir/cpp/main.cc.s"
	cd /Users/bobschreiner/Desktop/ETH/Semester4/Lectures/IML/introML/build_gcc/homeworks/task1a && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/bobschreiner/Desktop/ETH/Semester4/Lectures/IML/introML/homeworks/task1a/cpp/main.cc -o CMakeFiles/task1a_cpp.dir/cpp/main.cc.s

homeworks/task1a/CMakeFiles/task1a_cpp.dir/cpp/task1a.cc.o: homeworks/task1a/CMakeFiles/task1a_cpp.dir/flags.make
homeworks/task1a/CMakeFiles/task1a_cpp.dir/cpp/task1a.cc.o: ../homeworks/task1a/cpp/task1a.cc
homeworks/task1a/CMakeFiles/task1a_cpp.dir/cpp/task1a.cc.o: homeworks/task1a/CMakeFiles/task1a_cpp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/Users/bobschreiner/Desktop/ETH/Semester4/Lectures/IML/introML/build_gcc/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object homeworks/task1a/CMakeFiles/task1a_cpp.dir/cpp/task1a.cc.o"
	cd /Users/bobschreiner/Desktop/ETH/Semester4/Lectures/IML/introML/build_gcc/homeworks/task1a && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT homeworks/task1a/CMakeFiles/task1a_cpp.dir/cpp/task1a.cc.o -MF CMakeFiles/task1a_cpp.dir/cpp/task1a.cc.o.d -o CMakeFiles/task1a_cpp.dir/cpp/task1a.cc.o -c /Users/bobschreiner/Desktop/ETH/Semester4/Lectures/IML/introML/homeworks/task1a/cpp/task1a.cc

homeworks/task1a/CMakeFiles/task1a_cpp.dir/cpp/task1a.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/task1a_cpp.dir/cpp/task1a.cc.i"
	cd /Users/bobschreiner/Desktop/ETH/Semester4/Lectures/IML/introML/build_gcc/homeworks/task1a && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/bobschreiner/Desktop/ETH/Semester4/Lectures/IML/introML/homeworks/task1a/cpp/task1a.cc > CMakeFiles/task1a_cpp.dir/cpp/task1a.cc.i

homeworks/task1a/CMakeFiles/task1a_cpp.dir/cpp/task1a.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/task1a_cpp.dir/cpp/task1a.cc.s"
	cd /Users/bobschreiner/Desktop/ETH/Semester4/Lectures/IML/introML/build_gcc/homeworks/task1a && /usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/bobschreiner/Desktop/ETH/Semester4/Lectures/IML/introML/homeworks/task1a/cpp/task1a.cc -o CMakeFiles/task1a_cpp.dir/cpp/task1a.cc.s

# Object files for target task1a_cpp
task1a_cpp_OBJECTS = \
"CMakeFiles/task1a_cpp.dir/cpp/main.cc.o" \
"CMakeFiles/task1a_cpp.dir/cpp/task1a.cc.o"

# External object files for target task1a_cpp
task1a_cpp_EXTERNAL_OBJECTS =

homeworks/task1a/task1a_cpp: homeworks/task1a/CMakeFiles/task1a_cpp.dir/cpp/main.cc.o
homeworks/task1a/task1a_cpp: homeworks/task1a/CMakeFiles/task1a_cpp.dir/cpp/task1a.cc.o
homeworks/task1a/task1a_cpp: homeworks/task1a/CMakeFiles/task1a_cpp.dir/build.make
homeworks/task1a/task1a_cpp: homeworks/task1a/CMakeFiles/task1a_cpp.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/Users/bobschreiner/Desktop/ETH/Semester4/Lectures/IML/introML/build_gcc/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable task1a_cpp"
	cd /Users/bobschreiner/Desktop/ETH/Semester4/Lectures/IML/introML/build_gcc/homeworks/task1a && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/task1a_cpp.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
homeworks/task1a/CMakeFiles/task1a_cpp.dir/build: homeworks/task1a/task1a_cpp
.PHONY : homeworks/task1a/CMakeFiles/task1a_cpp.dir/build

homeworks/task1a/CMakeFiles/task1a_cpp.dir/clean:
	cd /Users/bobschreiner/Desktop/ETH/Semester4/Lectures/IML/introML/build_gcc/homeworks/task1a && $(CMAKE_COMMAND) -P CMakeFiles/task1a_cpp.dir/cmake_clean.cmake
.PHONY : homeworks/task1a/CMakeFiles/task1a_cpp.dir/clean

homeworks/task1a/CMakeFiles/task1a_cpp.dir/depend:
	cd /Users/bobschreiner/Desktop/ETH/Semester4/Lectures/IML/introML/build_gcc && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/bobschreiner/Desktop/ETH/Semester4/Lectures/IML/introML /Users/bobschreiner/Desktop/ETH/Semester4/Lectures/IML/introML/homeworks/task1a /Users/bobschreiner/Desktop/ETH/Semester4/Lectures/IML/introML/build_gcc /Users/bobschreiner/Desktop/ETH/Semester4/Lectures/IML/introML/build_gcc/homeworks/task1a /Users/bobschreiner/Desktop/ETH/Semester4/Lectures/IML/introML/build_gcc/homeworks/task1a/CMakeFiles/task1a_cpp.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : homeworks/task1a/CMakeFiles/task1a_cpp.dir/depend

