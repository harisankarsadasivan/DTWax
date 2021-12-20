NVCC= nvcc
NVCCFLAGS= -std=c++11 --default-stream per-thread -g -O3 -arch=sm_80 -lineinfo -gencode=arch=compute_80,code=sm_80 --compiler-options -Wall #-DNV_DEBUG
CXXFLAGS= -Xcompiler="-fopenmp -march=native" -O3 #-DNV_DEBUG

all: main

main: main.cu
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) main.cu -o main
format:
	clang-format -i main.cu include/*.hpp include/*.cu
clean:
		rm -rf main
