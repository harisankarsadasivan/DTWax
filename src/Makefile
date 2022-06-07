NVCC= nvcc
# LIBS = -lz -lhdf5 -I/home/hsadasivan/anaconda2/include -L/home/hsadasivan/anaconda2/lib -lcudnn
LIBS = -lz -lcudnn
NVCCFLAGS= -std=c++11 -g -G --default-stream per-thread  -O2 -lineinfo -gencode=arch=compute_86,code=sm_86 --compiler-options -Wall -DNV_DEBUG -DINCLUDE_NORMALIZER #-DFP16 #-DFAST5
CXXFLAGS= -Xcompiler="-fopenmp -march=native" -O2#-DNV_DEBUG

all: main

main: main.cu	
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) $(LIBS) main.cu -o main
format:
	clang-format -i main.cu include/*.hpp include/*.cu
clean:
		rm -rf main
