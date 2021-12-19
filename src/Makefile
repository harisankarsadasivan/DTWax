NVCC= nvcc
NVCCFLAGS= -O3 -std=c++11 -arch=sm_61
CXXFLAGS= -Xcompiler="-fopenmp -march=native"

all: main

main: main.cu
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) main.cu -o main

clean:
	rm -f main
