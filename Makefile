CXX = clang++
MPICXX ?= $(shell command -v mpicxx 2>/dev/null || echo /opt/homebrew/bin/mpicxx)
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra

TARGETS = bin/kmeans_serial bin/kmeans_mpi

all: $(TARGETS)

bin/kmeans_serial: src/kmeans_serial.cpp
	mkdir -p bin results
	$(CXX) $(CXXFLAGS) $< -o $@

bin/kmeans_mpi: src/kmeans_mpi.cpp
	mkdir -p bin results
	$(MPICXX) $(CXXFLAGS) $< -o $@

clean:
	rm -rf bin results/*.txt
