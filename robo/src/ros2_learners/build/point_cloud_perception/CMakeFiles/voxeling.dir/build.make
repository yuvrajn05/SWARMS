# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.31

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
CMAKE_COMMAND = /home/vatsan/.local/lib/python3.10/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /home/vatsan/.local/lib/python3.10/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/vatsan/robo/src/ros2_learners/point_cloud_perception

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/vatsan/robo/src/ros2_learners/build/point_cloud_perception

# Include any dependencies generated for this target.
include CMakeFiles/voxeling.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/voxeling.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/voxeling.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/voxeling.dir/flags.make

CMakeFiles/voxeling.dir/codegen:
.PHONY : CMakeFiles/voxeling.dir/codegen

CMakeFiles/voxeling.dir/src/voxelizing.cpp.o: CMakeFiles/voxeling.dir/flags.make
CMakeFiles/voxeling.dir/src/voxelizing.cpp.o: /home/vatsan/robo/src/ros2_learners/point_cloud_perception/src/voxelizing.cpp
CMakeFiles/voxeling.dir/src/voxelizing.cpp.o: CMakeFiles/voxeling.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/vatsan/robo/src/ros2_learners/build/point_cloud_perception/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/voxeling.dir/src/voxelizing.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/voxeling.dir/src/voxelizing.cpp.o -MF CMakeFiles/voxeling.dir/src/voxelizing.cpp.o.d -o CMakeFiles/voxeling.dir/src/voxelizing.cpp.o -c /home/vatsan/robo/src/ros2_learners/point_cloud_perception/src/voxelizing.cpp

CMakeFiles/voxeling.dir/src/voxelizing.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/voxeling.dir/src/voxelizing.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/vatsan/robo/src/ros2_learners/point_cloud_perception/src/voxelizing.cpp > CMakeFiles/voxeling.dir/src/voxelizing.cpp.i

CMakeFiles/voxeling.dir/src/voxelizing.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/voxeling.dir/src/voxelizing.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/vatsan/robo/src/ros2_learners/point_cloud_perception/src/voxelizing.cpp -o CMakeFiles/voxeling.dir/src/voxelizing.cpp.s

# Object files for target voxeling
voxeling_OBJECTS = \
"CMakeFiles/voxeling.dir/src/voxelizing.cpp.o"

# External object files for target voxeling
voxeling_EXTERNAL_OBJECTS =

voxeling: CMakeFiles/voxeling.dir/src/voxelizing.cpp.o
voxeling: CMakeFiles/voxeling.dir/build.make
voxeling: /usr/lib/x86_64-linux-gnu/libpcl_apps.so
voxeling: /usr/lib/x86_64-linux-gnu/libpcl_outofcore.so
voxeling: /usr/lib/x86_64-linux-gnu/libpcl_people.so
voxeling: /usr/lib/libOpenNI.so
voxeling: /usr/lib/x86_64-linux-gnu/libusb-1.0.so
voxeling: /usr/lib/x86_64-linux-gnu/libOpenNI2.so
voxeling: /usr/lib/x86_64-linux-gnu/libusb-1.0.so
voxeling: /usr/lib/x86_64-linux-gnu/libflann_cpp.so
voxeling: /usr/lib/x86_64-linux-gnu/libpcl_surface.so
voxeling: /usr/lib/x86_64-linux-gnu/libpcl_keypoints.so
voxeling: /usr/lib/x86_64-linux-gnu/libpcl_tracking.so
voxeling: /usr/lib/x86_64-linux-gnu/libpcl_recognition.so
voxeling: /usr/lib/x86_64-linux-gnu/libpcl_registration.so
voxeling: /usr/lib/x86_64-linux-gnu/libpcl_stereo.so
voxeling: /usr/lib/x86_64-linux-gnu/libpcl_segmentation.so
voxeling: /usr/lib/x86_64-linux-gnu/libpcl_features.so
voxeling: /usr/lib/x86_64-linux-gnu/libpcl_filters.so
voxeling: /usr/lib/x86_64-linux-gnu/libpcl_sample_consensus.so
voxeling: /usr/lib/x86_64-linux-gnu/libpcl_ml.so
voxeling: /usr/lib/x86_64-linux-gnu/libpcl_visualization.so
voxeling: /usr/lib/x86_64-linux-gnu/libpcl_search.so
voxeling: /usr/lib/x86_64-linux-gnu/libpcl_kdtree.so
voxeling: /usr/lib/x86_64-linux-gnu/libpcl_io.so
voxeling: /usr/lib/x86_64-linux-gnu/libpcl_octree.so
voxeling: /usr/lib/x86_64-linux-gnu/libpng.so
voxeling: /usr/lib/x86_64-linux-gnu/libz.so
voxeling: /usr/lib/libOpenNI.so
voxeling: /usr/lib/x86_64-linux-gnu/libusb-1.0.so
voxeling: /usr/lib/x86_64-linux-gnu/libOpenNI2.so
voxeling: /usr/lib/x86_64-linux-gnu/libvtkChartsCore-9.1.so.9.1.0
voxeling: /usr/lib/x86_64-linux-gnu/libvtkInteractionImage-9.1.so.9.1.0
voxeling: /usr/lib/x86_64-linux-gnu/libvtkIOGeometry-9.1.so.9.1.0
voxeling: /usr/lib/x86_64-linux-gnu/libjsoncpp.so
voxeling: /usr/lib/x86_64-linux-gnu/libvtkIOPLY-9.1.so.9.1.0
voxeling: /usr/lib/x86_64-linux-gnu/libvtkRenderingLOD-9.1.so.9.1.0
voxeling: /usr/lib/x86_64-linux-gnu/libvtkViewsContext2D-9.1.so.9.1.0
voxeling: /usr/lib/x86_64-linux-gnu/libvtkViewsCore-9.1.so.9.1.0
voxeling: /usr/lib/x86_64-linux-gnu/libvtkGUISupportQt-9.1.so.9.1.0
voxeling: /usr/lib/x86_64-linux-gnu/libvtkInteractionWidgets-9.1.so.9.1.0
voxeling: /usr/lib/x86_64-linux-gnu/libvtkFiltersModeling-9.1.so.9.1.0
voxeling: /usr/lib/x86_64-linux-gnu/libvtkInteractionStyle-9.1.so.9.1.0
voxeling: /usr/lib/x86_64-linux-gnu/libvtkFiltersExtraction-9.1.so.9.1.0
voxeling: /usr/lib/x86_64-linux-gnu/libvtkIOLegacy-9.1.so.9.1.0
voxeling: /usr/lib/x86_64-linux-gnu/libvtkIOCore-9.1.so.9.1.0
voxeling: /usr/lib/x86_64-linux-gnu/libvtkRenderingAnnotation-9.1.so.9.1.0
voxeling: /usr/lib/x86_64-linux-gnu/libvtkRenderingContext2D-9.1.so.9.1.0
voxeling: /usr/lib/x86_64-linux-gnu/libvtkRenderingFreeType-9.1.so.9.1.0
voxeling: /usr/lib/x86_64-linux-gnu/libfreetype.so
voxeling: /usr/lib/x86_64-linux-gnu/libvtkImagingSources-9.1.so.9.1.0
voxeling: /usr/lib/x86_64-linux-gnu/libvtkIOImage-9.1.so.9.1.0
voxeling: /usr/lib/x86_64-linux-gnu/libvtkImagingCore-9.1.so.9.1.0
voxeling: /usr/lib/x86_64-linux-gnu/libvtkRenderingOpenGL2-9.1.so.9.1.0
voxeling: /usr/lib/x86_64-linux-gnu/libvtkRenderingUI-9.1.so.9.1.0
voxeling: /usr/lib/x86_64-linux-gnu/libvtkRenderingCore-9.1.so.9.1.0
voxeling: /usr/lib/x86_64-linux-gnu/libvtkCommonColor-9.1.so.9.1.0
voxeling: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeometry-9.1.so.9.1.0
voxeling: /usr/lib/x86_64-linux-gnu/libvtkFiltersSources-9.1.so.9.1.0
voxeling: /usr/lib/x86_64-linux-gnu/libvtkFiltersGeneral-9.1.so.9.1.0
voxeling: /usr/lib/x86_64-linux-gnu/libvtkCommonComputationalGeometry-9.1.so.9.1.0
voxeling: /usr/lib/x86_64-linux-gnu/libvtkFiltersCore-9.1.so.9.1.0
voxeling: /usr/lib/x86_64-linux-gnu/libvtkCommonExecutionModel-9.1.so.9.1.0
voxeling: /usr/lib/x86_64-linux-gnu/libvtkCommonDataModel-9.1.so.9.1.0
voxeling: /usr/lib/x86_64-linux-gnu/libvtkCommonMisc-9.1.so.9.1.0
voxeling: /usr/lib/x86_64-linux-gnu/libvtkCommonTransforms-9.1.so.9.1.0
voxeling: /usr/lib/x86_64-linux-gnu/libvtkCommonMath-9.1.so.9.1.0
voxeling: /usr/lib/x86_64-linux-gnu/libvtkkissfft-9.1.so.9.1.0
voxeling: /usr/lib/x86_64-linux-gnu/libGLEW.so
voxeling: /usr/lib/x86_64-linux-gnu/libX11.so
voxeling: /usr/lib/x86_64-linux-gnu/libQt5OpenGL.so.5.15.3
voxeling: /usr/lib/x86_64-linux-gnu/libQt5Widgets.so.5.15.3
voxeling: /usr/lib/x86_64-linux-gnu/libQt5Gui.so.5.15.3
voxeling: /usr/lib/x86_64-linux-gnu/libQt5Core.so.5.15.3
voxeling: /usr/lib/x86_64-linux-gnu/libvtkCommonCore-9.1.so.9.1.0
voxeling: /usr/lib/x86_64-linux-gnu/libtbb.so.12.5
voxeling: /usr/lib/x86_64-linux-gnu/libvtksys-9.1.so.9.1.0
voxeling: /usr/lib/x86_64-linux-gnu/libpcl_common.so
voxeling: /usr/lib/x86_64-linux-gnu/libboost_system.so.1.74.0
voxeling: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so.1.74.0
voxeling: /usr/lib/x86_64-linux-gnu/libboost_date_time.so.1.74.0
voxeling: /usr/lib/x86_64-linux-gnu/libboost_iostreams.so.1.74.0
voxeling: /usr/lib/x86_64-linux-gnu/libboost_serialization.so.1.74.0
voxeling: /usr/lib/x86_64-linux-gnu/libqhull_r.so.8.0.2
voxeling: CMakeFiles/voxeling.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/vatsan/robo/src/ros2_learners/build/point_cloud_perception/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable voxeling"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/voxeling.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/voxeling.dir/build: voxeling
.PHONY : CMakeFiles/voxeling.dir/build

CMakeFiles/voxeling.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/voxeling.dir/cmake_clean.cmake
.PHONY : CMakeFiles/voxeling.dir/clean

CMakeFiles/voxeling.dir/depend:
	cd /home/vatsan/robo/src/ros2_learners/build/point_cloud_perception && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/vatsan/robo/src/ros2_learners/point_cloud_perception /home/vatsan/robo/src/ros2_learners/point_cloud_perception /home/vatsan/robo/src/ros2_learners/build/point_cloud_perception /home/vatsan/robo/src/ros2_learners/build/point_cloud_perception /home/vatsan/robo/src/ros2_learners/build/point_cloud_perception/CMakeFiles/voxeling.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/voxeling.dir/depend

