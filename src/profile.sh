sudo LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.local/hdf5-1.10.4/lib HDF5_PLUGIN_PATH=~/.local/hdf5/lib/plugin  /usr/local/cuda-11.5/nsight-compute-2021.3.1/target/linux-desktop-glibc_2_11_3-x64/ncu --export "/home/hsadasivan_nvidia_com/nv_cudtw/src/fp16_16blocks" --force-overwrite --target-processes all --kernel-name-base function --launch-skip-before-match 0 --section ComputeWorkloadAnalysis --section InstructionStats --section LaunchStats --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Chart --section MemoryWorkloadAnalysis_Tables --section Occupancy --section SchedulerStats --section SourceCounters --section SpeedOfLight --section SpeedOfLight_RooflineChart --section WarpStateStats --sampling-interval auto --sampling-max-passes 5 --sampling-buffer-size 33554432 --profile-from-start 1 --cache-control all --clock-control base --apply-rules yes --import-source no --check-exit-code yes ./main ../data/ ref/dna_kmer_model.txt ref/reference.fasta