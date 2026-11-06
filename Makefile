CXX = clang++
CXXFLAGS = -O3 -std=c++17 -Wall -Wextra

TARGET = bin/kmeans_serial

all: $(TARGET)

$(TARGET): src/kmeans_serial.cpp
	mkdir -p bin results
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -rf bin results/*.txt
