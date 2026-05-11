# Minimum required version
cmake_minimum_required(VERSION 3.20)

# Project name
project(ProgettoAspromonte LANGUAGES CXX)

# C++ CONFIGURATION #

# Enforce ISO C++23 standard (disable GNU extensions for portability)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# BUILD TYPE CONFIGURATION #

# Default to 'Release' (enables GCC -O3 optimization) if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type options: None, Debug, Release" FORCE)
endif()

# ARCHITECTURE & FEATURE OPTIONS #

option(USE_NATIVE_INSTRUCTIONS "Enable -march=native for CPU vectorization (AVX2/AVX-512)" OFF)
option(ENABLE_GUI "Enable OpenCV GUI windows for debugging" OFF)
option(ENABLE_PROFILING "Enable execution time logging" OFF)

# DEPENDENCIES #

find_package(OpenCV REQUIRED)

# EXECUTABLE TARGET #

add_executable(Aspromonte main.cpp)

# TARGET CONFIGURATIONS & OPTIMIZATIONS #

# Strict warnings and Link-Time Optimization (LTO)
target_compile_options(Aspromonte PRIVATE
    -Wall -Wextra -Wpedantic
    -flto
)
# Enable LTO for the linker
target_link_options(Aspromonte PRIVATE -flto)

# Apply hardware-specific optimizations if requested
if(USE_NATIVE_INSTRUCTIONS)
    message(STATUS "Native optimization ENABLED: Binary will be hardware-specific.")
    target_compile_options(Aspromonte PRIVATE -march=native)
else()
    message(STATUS "Native optimization DISABLED: Binary is portable.")
endif()

# Inject preprocessor macros based on selected features
if(ENABLE_GUI)
    message(STATUS "GUI ENABLED: OpenCV windows will be compiled.")
    target_compile_definitions(Aspromonte PRIVATE USE_GUI)
endif()

if(ENABLE_PROFILING)
    message(STATUS "Profiling ENABLED: Execution times will be logged.")
    target_compile_definitions(Aspromonte PRIVATE ENABLE_PROFILING)
endif()

# LINKING #
target_include_directories(Aspromonte PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(Aspromonte PRIVATE ${OpenCV_LIBS})