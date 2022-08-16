# Adapted from cuDTW++
cuDTW++: Ultra-Fast Dynamic Time Warping on CUDA-enabled GPUs

Efficient CUDA implementation of the Dynamic Time Warp (DTW) algorithm. Details can be found in the paper "cuDTW++: Ultra-Fast Dynamic Time Warping on CUDA-enabled GPUs", by Bertil Schmidt and Christian Hundt (Euro-Par 2020).



Nanopore and cudnn pre-requisites:
sudo apt install hdf5-tools
sudo apt-get install libhdf5-dev zlib1g-dev zlib1g

Following 2 steps may require creating an account and logging-in.
Install NVIDIA drivers: cudnn--- intstall toolkit 11.6 and then do this from nvidia doc for 11.6.
Install CUDNN:install cudnn from https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html; deb file cudnn-local-repo-ubuntu1804-8.5.0.96_1.0-1_amd64.deb;

Clone repo:
git clone --recursive https://github.com/hsadasivan/nv_cudtw.git -b FAST5

Build and run the program:

make -j 1000;
./main <fast5_folder>  ref/dna_kmer_model.txt <path_to_reference.fasta> > output_log;
