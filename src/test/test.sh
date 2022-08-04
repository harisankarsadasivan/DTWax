for i in {1..32}
do
    sed -i "34s/.*/#define SEGMENT_SIZE $i/" include/common.hpp ;
    make clean;make -j 1000;./main test/ ref/dna_kmer_model.txt ref/reference.fasta >> log;
done
