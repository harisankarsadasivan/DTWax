# Nanopore and cudnn pre-requisites:
sudo apt install hdf5-tools
sudo apt-get install libhdf5-dev zlib1g-dev zlib1g

#Install CUDA toolkit and CUDNN:
Following 2 steps may require creating an account and logging-in.
Install NVIDIA toolkit 11.5+ preferably from local run file. Follow guide here: https://developer.nvidia.com/cuda-downloads.
Install CUDNN:install cudnn from https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html;

# Clone repo:
git clone --recursive https://github.com/hsadasivan/nv_cudtw.git -b FAST5

# Build and run the program:

make -j 1000;
./main <fast5_folder>  ref/dna_kmer_model.txt <path_to_reference.fasta> > output_log;


**Acknowledgement:**


# Acknowledgement:
cuDTW++: Ultra-Fast Dynamic Time Warping on CUDA-enabled GPUs

Efficient CUDA implementation of the Dynamic Time Warp (DTW) algorithm. Details can be found in the paper "cuDTW++: Ultra-Fast Dynamic Time Warping on CUDA-enabled GPUs", by Bertil Schmidt and Christian Hundt (Euro-Par 2020).
