# Install CUDA toolkit and CUDNN:
These two steps may require creating an account and logging-in.


Install NVIDIA toolkit 11.5+ preferably from a local run file. Please follow the guide here: https://developer.nvidia.com/cuda-downloads.


Install CUDNN from https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html.

# ONT pre-requisites:

sudo apt-get install libhdf5-dev zlib1g-dev

sudo apt install hdf5-tools

export HDF5_PLUGIN_PATH=/usr/local/hdf5/lib/plugin


# Clone repo:
git clone --recursive https://github.com/hsadasivan/nv_cudtw.git -b FAST5

# Build and run the program:

make -j 1000;
./main <fast5_folder>  ref/dna_kmer_model.txt <path_to_reference.fasta> > output_log;

# Acknowledgement:
cuDTW++: Ultra-Fast Dynamic Time Warping on CUDA-enabled GPUs: https://github.com/asbschmidt/cuDTW.
Efficient CUDA implementation of the Dynamic Time Warp (DTW) algorithm. Details can be found in the paper "cuDTW++: Ultra-Fast Dynamic Time Warping on CUDA-enabled GPUs", by Bertil Schmidt and Christian Hundt (Euro-Par 2020).

**Stating changes:** cuDTW++ is used as a starting framework. Underlying DTW algorithm is modified to sDTW algorithm. Further, the code is scaled, analyzed, optimized and re-purposed to support ONT Read Until on A100 and other Ampere cards.

Nanopolish:https://github.com/jts/nanopolish/blob/master/LICENSE

