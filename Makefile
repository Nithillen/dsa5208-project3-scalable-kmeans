CXX ?= clang++
MPICXX ?= $(shell command -v mpicxx 2>/dev/null || echo /opt/homebrew/bin/mpicxx)
NVCC ?= $(shell command -v nvcc 2>/dev/null)
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    OMP_FLAGS = -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp
    METAL_TARGET = bin/kmeans_metal_gpu
else
    CXX = g++
    OMP_FLAGS = -fopenmp
    METAL_TARGET =
endif

ifneq ($(NVCC),)
    CUDA_TARGET = bin/kmeans_cuda
else
    CUDA_TARGET =
endif

TARGETS = bin/kmeans_serial bin/kmeans_mpi bin/kmeans_omp bin/kmeans_omp_standalone $(METAL_TARGET) $(CUDA_TARGET)

all: $(TARGETS)

bin/kmeans_serial: src/kmeans_serial.cpp src/kmeans_common.hpp
	mkdir -p bin results
	$(CXX) $(CXXFLAGS) $< -o $@

bin/kmeans_mpi: src/kmeans_mpi.cpp src/kmeans_common.hpp
	mkdir -p bin results
	$(MPICXX) $(CXXFLAGS) $< -o $@

bin/kmeans_omp: src/kmeans_omp.cpp src/kmeans_common.hpp
	mkdir -p bin results
	$(MPICXX) $(CXXFLAGS) $(OMP_FLAGS) $< -o $@

bin/kmeans_omp_standalone: src/kmeans_omp.cpp src/kmeans_common.hpp
	mkdir -p bin results
	$(CXX) $(CXXFLAGS) $(OMP_FLAGS) -DSTANDALONE_OMP $< -o $@

bin/kmeans_metal_gpu: src/kmeans_metal_gpu.mm src/kmeans_common.hpp
	mkdir -p bin results
	$(MPICXX) $(CXXFLAGS) $(OMP_FLAGS) -fobjc-arc -framework Metal -framework Foundation $< -o $@

bin/kmeans_cuda: src/kmeans_cuda.cu src/kmeans_common.hpp
	mkdir -p bin results
	$(NVCC) -O3 -std=c++17 -ccbin $(MPICXX) $< -o $@

test: all
	python3 scripts/test_correctness.py

# Safe clean: only removes compiled binaries and build objects, never deletes results/*.txt or results/*.csv!
clean:
	rm -rf bin/*.o bin/*.dSYM bin/kmeans_*
