# Minimum required version
cmake_minimum_required(VERSION 3.10)

# Project name
project(ProgettoAspromonte)

# Packages
find_package(OpenCV REQUIRED)

# Headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Executable
add_executable(Aspromonte main.cpp)

# Link between OpenCV and the project
target_link_libraries(Aspromonte ${OpenCV_LIBS})