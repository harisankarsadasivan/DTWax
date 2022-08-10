NVCC= nvcc
LIBS = -lz -lhdf5 -I/usr/include/hdf5/serial -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lcudnn 
# HDF5_LIB=-L/home/linuxbrew/.linuxbrew/lib
# HDF5_INCLUDE=-I/home/linuxbrew/.linuxbrew/include

# LIBS = -lhdf5 -lcudnn -I/usr/include/hdf5/serial -L/usr/lib -L/usr/lib/x86_64-linux-gnu/hdf5/serial -I/usr/local/include -L/usr/local/lib -L/usr/local/lib 
NVCCFLAGS= -std=c++11 --default-stream per-thread  -O2 -lineinfo -gencode=arch=compute_86,code=sm_86 --compiler-options -Wall -DSDTW #-DNV_DEBUG###-DDEBUG_FAST5_IO#-DFP16
CXXFLAGS= -Xcompiler="-fopenmp -march=native" -O2#-DNV_DEBUG
export HDF5_PLUGIN_PATH=/home/hsadasivan/.local/hdf5/lib/plugin
all: main

main: main.cu
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) $(LIBS) main.cu -o main
format:
	clang-format -i main.cu include/*.hpp include/*.cu include/*cpp
clean:
		rm -rf main 
$(LIBHDF5):
	cd ../hdf5 && \
	./configure --enable-threadsafe --disable-hl --prefix=`pwd` --enable-shared=no --with-pic=yes || exit 255
	make -j 8 -C ../hdf5 && make -C ../hdf5 install
