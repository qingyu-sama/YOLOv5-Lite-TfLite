# Makefile for YOLOv5-Lite TFLite inference (Linux)
#
# Prerequisites:
#   sudo apt install libopencv-dev
#   # Download TensorFlow Lite C library from:
#   # https://github.com/nicktajzsoldos/libtensorflowlite-c-linux
#   # Or build from source: https://www.tensorflow.org/lite/guide/build_cmake
#
# Usage:
#   make TFLITE_DIR=/path/to/tensorflow-lite
#   ./demo --model best_full_integer_quant.tflite --image test.jpg

CXX      := g++
CXXFLAGS := -std=c++17 -O3 -march=native -mtune=native -DNDEBUG \
            -Wall -Wextra -Wpedantic

# --- TensorFlow Lite paths (customize these) ---
TFLITE_DIR ?= /usr/local/lib/tensorflow-lite
TFLITE_INC ?= $(TFLITE_DIR)/include
TFLITE_LIB ?= $(TFLITE_DIR)/lib

# --- OpenCV (auto-detect via pkg-config) ---
OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4 2>/dev/null)
OPENCV_LIBS   := $(shell pkg-config --libs opencv4 2>/dev/null)

# --- Targets ---
.PHONY: all clean

all: demo

demo: main.cpp yolov5_lite_tflite.cpp yolov5_lite_tflite.h
	$(CXX) $(CXXFLAGS) \
	    -I$(TFLITE_INC) \
	    $(OPENCV_CFLAGS) \
	    -o $@ \
	    main.cpp yolov5_lite_tflite.cpp \
	    -L$(TFLITE_LIB) -ltensorflow-lite \
	    $(OPENCV_LIBS) \
	    -lpthread -ldl

clean:
	rm -f demo
