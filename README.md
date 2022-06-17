# Adapted from cuDTW++
cuDTW++: Ultra-Fast Dynamic Time Warping on CUDA-enabled GPUs

Efficient CUDA implementation of the Dynamic Time Warp (DTW) algorithm. Details can be found in the paper "cuDTW++: Ultra-Fast Dynamic Time Warping on CUDA-enabled GPUs", by Bertil Schmidt and Christian Hundt (Euro-Par 2020).

Nanopore pre-requisites:
sudo apt-get install libhdf5-dev zlib1g-dev
-install hdf5 from h5fc
add hdf5 lib path to LD_LIBRARY_PATH
install hdf5 plugin from f5c and add export HDF5_PLUGIN_PATH="/home/hsadasivan/.local/hdf5/lib/plugin"
