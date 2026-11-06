CXX = clang++
MPICXX ?= $(shell command -v mpicxx 2>/dev/null || echo /opt/homebrew/bin/mpicxx)
NVCC ?= $(shell command -v nvcc 2>/dev/null)
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    OMP_FLAGS = -Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include -L/opt/homebrew/opt/libomp/lib -lomp
    METAL_TARGET = bin/kmeans_metal_gpu
else
    OMP_FLAGS = -fopenmp
    METAL_TARGET =
endif

ifneq ($(NVCC),)
    CUDA_TARGET = bin/kmeans_cuda
else
    CUDA_TARGET =
endif

TARGETS = bin/kmeans_serial bin/kmeans_mpi bin/kmeans_omp $(METAL_TARGET) $(CUDA_TARGET)

all: $(TARGETS)

bin/kmeans_serial: src/kmeans_serial.cpp
	mkdir -p bin results
	$(CXX) $(CXXFLAGS) $< -o $@

bin/kmeans_mpi: src/kmeans_mpi.cpp
	mkdir -p bin results
	$(MPICXX) $(CXXFLAGS) $< -o $@

bin/kmeans_omp: src/kmeans_omp.cpp
	mkdir -p bin results
	$(MPICXX) $(CXXFLAGS) $(OMP_FLAGS) $< -o $@

bin/kmeans_metal_gpu: src/kmeans_metal_gpu.mm
	mkdir -p bin results
	$(MPICXX) $(CXXFLAGS) $(OMP_FLAGS) -fobjc-arc -framework Metal -framework Foundation $< -o $@

bin/kmeans_cuda: src/kmeans_cuda.cu
	mkdir -p bin results
	$(NVCC) -O3 -std=c++17 -ccbin $(MPICXX) $< -o $@

clean:
	rm -rf bin results/*.txt
