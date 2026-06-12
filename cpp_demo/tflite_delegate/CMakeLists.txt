cmake_minimum_required(VERSION 3.14)
project(yolov5_lite_tflite VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ------------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------------

# TensorFlow Lite
# Point to your TensorFlow Lite installation:
#   cmake -DTFLite_DIR=/path/to/tensorflow-lite ...
find_package(TFLite REQUIRED)
# If find_package doesn't work, use:
# set(TFLITE_INCLUDE_DIRS "/path/to/tensorflow" CACHE PATH "")
# set(TFLITE_LIBRARIES "/path/to/libtensorflow-lite.so" CACHE FILEPATH "")

# OpenCV (for image I/O)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

# ------------------------------------------------------------------
# Library target
# ------------------------------------------------------------------
add_library(yolov5_lite_tflite STATIC
    yolov5_lite_tflite.h
    yolov5_lite_tflite.cpp
)

target_include_directories(yolov5_lite_tflite PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${TFLITE_INCLUDE_DIRS}
)

target_link_libraries(yolov5_lite_tflite PUBLIC
    ${TFLITE_LIBRARIES}
)

# ------------------------------------------------------------------
# Demo executable
# ------------------------------------------------------------------
add_executable(demo main.cpp)

target_link_libraries(demo PRIVATE
    yolov5_lite_tflite
    ${OpenCV_LIBS}
)

# ------------------------------------------------------------------
# Compiler options
# ------------------------------------------------------------------
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(yolov5_lite_tflite PRIVATE
        -Wall -Wextra -Wpedantic
        -march=native -mtune=native
        -O3 -DNDEBUG
    )
endif()

# ------------------------------------------------------------------
# Install
# ------------------------------------------------------------------
install(TARGETS demo DESTINATION bin)
install(TARGETS yolov5_lite_tflite DESTINATION lib)
