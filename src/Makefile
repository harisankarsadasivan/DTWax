NVCC= nvcc
NVCCFLAGS= -std=c++11 --default-stream per-thread -O2 -lineinfo -gencode=arch=compute_86,code=sm_86 --compiler-options -Wall -DNV_DEBUG -DFP16
CXXFLAGS= -Xcompiler="-fopenmp -march=native" -O2 #-DNV_DEBUG

all: main

main: main.cu
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) main.cu -o main
format:
	clang-format -i main.cu include/*.hpp include/*.cu
clean:
		rm -rf main
