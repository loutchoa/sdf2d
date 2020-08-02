cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
project(cpp_eikonal_project VERSION 0.1 LANGUAGES CXX)

# Find torch
find_package(Torch REQUIRED)

# Include custom FindMatlab.cmake
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}")
set(MATLAB_ROOT "C:\\Program Files (x86)\\MATLAB\\R2020a")
find_package(Matlab REQUIRED)

# Build the shared library
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS YES CACHE BOOL "Export all symbols")
add_library(project_lib SHARED src/minHeap.cpp include/minHeap.h)
target_compile_features(project_lib PUBLIC cxx_std_11)
target_link_libraries(project_lib "${TORCH_LIBRARIES}")
target_include_directories(project_lib PUBLIC include)

# Build the MATLAB Mex Target
matlab_add_mex(
  NAME deep_eikonal_solver_c
  SHARED
  SRC src/fmm2d.cpp
  LINK_TO project_lib)

# For Windows one needs to copy the .dll to where the target is being built since there is no such thing
# as rpath on windows.
# Note: As of know, while running the mex file,
# you have the be in the same folder as the .mex and all the dll's are, otherwise it does not work ...
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET project_lib
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:project_lib>)
endif (MSVC)