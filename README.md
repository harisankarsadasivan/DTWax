# CUDA Pre-requisites:
These two steps may require creating an account and logging-in.

Install NVIDIA toolkit 11.5+ preferably from a local run file. Please follow the guide here: https://developer.nvidia.com/cuda-downloads.

Install CUDNN from https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html.

# ONT pre-requisites:
```
sudo apt-get install libhdf5-dev zlib1g-dev

sudo apt install hdf5-tools

export HDF5_PLUGIN_PATH=$HOME/nv_cudtw/src:$HDF5_PLUGIN_PATH
```


# Clone repo:
```
git clone --recursive https://github.com/hsadasivan/nv_cudtw.git -b FAST5
cd src/;
```
# [Optional] Tune for threshold of classification [may vary based on wet-lab protocols]
```
make -j 1000;

### Run DTWax on dataset of reads belonging only to the target
./main <fast5_folder>  ref/dna_kmer_model.txt <path_to_reference.fasta> > dtwax_microbial_log;

### Run DTWax on dataset of reads belonging only to the non-target
./main <fast5_folder>  ref/dna_kmer_model.txt <path_to_reference.fasta> > dtwax_nontarget_log;

### Find threshold of classification
python calc_classification_threshold.py dtwax_microbial_log dtwax_nontarget_log
```
# Build and run the program (Offline: FAST5 input):
```


###Review HDF5 plugin export path in Makefile; change compute_* and sm_* to your GPU's compute capability

make -j 1000;

./main <fast5_folder>  ref/dna_kmer_model.txt <path_to_reference.fasta> > output_log;
```
# Build and run the program (Online Read Until via API) [[LIVE DEMO](https://youtu.be/E5XDGLGTH-M)]
```
###Update source and Makefile to enable Read Until
cp -rf Makefile_readuntil Makefile
cp -rf ru_main.cu main.cu

###Review HDF5 plugin export path in Makefile; change compute_* and sm_* to your GPU's compute capability
make clean; make -j 1000;

###Install Read Until python API
cd ReadUntil/
chmod -R 777 install_ru.sh
./install_ru.sh

### Start MinION sequencing
### Start Read Until and dump DTWax's output to a log
python3 ont_readuntil.py >> dtwax_log


### Launch DTWatch to visualize virus vs non-target reads identified in real-time
python3 plot_realtime.py dtwax_log <CLASSIFICATION_THRESHOLD>



```
# Acknowledgement:
cuDTW++: Ultra-Fast Dynamic Time Warping on CUDA-enabled GPUs: https://github.com/asbschmidt/cuDTW.
Efficient CUDA implementation of the Dynamic Time Warp (DTW) algorithm. Details can be found in the paper "cuDTW++: Ultra-Fast Dynamic Time Warping on CUDA-enabled GPUs", by Bertil Schmidt and Christian Hundt (Euro-Par 2020).

**Stating changes:** cuDTW++ is used as a starting framework. Underlying DTW algorithm is modified to sDTW algorithm. Further, the code is scaled, analyzed, optimized and re-purposed to support ONT Read Until on A100 and other Ampere cards.

Nanopolish:https://github.com/jts/nanopolish/blob/master/LICENSE

