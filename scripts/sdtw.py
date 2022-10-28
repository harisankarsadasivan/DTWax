#!/usr/bin/env python

# manage imports 
import math
from sklearn import metrics
from itertools import repeat
from numba import njit
from glob import glob
from scipy import stats
import random, h5py, re, os
import numpy as np
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D  
import multiprocessing as mp
# import seaborn as sns
# import pandas as pd         
import ont_fast5_api
import pickle


#####6mer model mean and var
model_mean=90.208270
model_stdv=12.86865
model_var=165.602358
#######################
#constant
sqrt_2pi=2.506628
sqrt_2pi_std=22.217231
spatial_mu=10.391752
spatial_var=78.559737

######################

virus = "lambda"
other = "human"

# user-defined filepaths
data_dir = "/home/hariss/DTWax/test"
img_dir = "img"
os.makedirs(img_dir, exist_ok=True)
kmer_model_fn,k= "/home/hariss/DTWax/src/ref/dna_kmer_model.txt", 6 # 6-mer model
ref_fn = "/home/hariss/lambda/reference.fasta"
virus_fast5_dir = f"{data_dir}/"
other_fast5_dir = f"{data_dir}/"
results_dir = f"scripts/results/artifact-eval"

# user-defined analysis params
virus_max_reads = 1
other_max_reads = 1
R= (47 * 1024 * 2)
N=4096
prefix_lengths = np.array(range(N,N+1,500))
# prefix_lengths = np.array(range(500,3001,500))
nprefixes = len(prefix_lengths)
int_sig=0
NBITS=8


def get_fasta(fasta_fn):
    ''' Get base sequence from FASTA filename. '''
    with open(fasta_fn, 'r') as fasta:
        return ''.join(fasta.read().split('\n')[1:])

    
def rev_comp(bases):
    ''' Get reverse complement of sequence. '''
    return bases.replace('A','t').replace('T','a')        .replace('G','c').replace('C','g').upper()[::-1]


def load_model(kmer_model_fn):
    ''' Load k-mer model file into Python dict. '''
    kmer_model = {}
    var_model = {}
    with open(kmer_model_fn, 'r') as model_file:
        for line in model_file:
            #print(line)
            kmer, current, stddev = line.split()
            kmer_model[kmer] = float(current)
            var_model[kmer] = float(stddev)*float(stddev)  
            if(var_model[kmer]==0):
                print("ERROR in variance model")
    return kmer_model, var_model


def discrete_normalize(seq, bits=NBITS, minval=-4, maxval=4):
    ''' Approximate normalization which converts signal to integer of desired precision. '''
    if(int_sig):
        mean = int(np.mean(seq))
        mean_avg_dev = int(np.mean(np.abs(seq - mean)))
        norm_seq = (seq - mean) / mean_avg_dev
        norm_seq[norm_seq < minval] = minval # threshold
        norm_seq[norm_seq > maxval] = maxval 
        norm_seq = ( (norm_seq - minval) * (2**(bits)/(maxval-minval)) ).astype(int)
        return norm_seq
    else:
        mean = (np.mean(seq))
        mean_avg_dev = np.power(np.mean(np.power(seq - mean,2)),0.5)
        return [(i-mean)/mean_avg_dev for i in seq]

def ref_signal(fasta, kmer_model,var_model):
    ''' Convert reference FASTA to expected reference signal (approximate z-scores). '''
    if(int_sig):
        signal = np.zeros(len(fasta)-k)
        var = np.zeros(len(fasta)-k)
        for kmer_start in range(len(fasta)-k):
            signal[kmer_start] = kmer_model[fasta[kmer_start:kmer_start+k]]
            var[kmer_start] = var_model[fasta[kmer_start:kmer_start+k]]
        return discrete_normalize(signal*100),discrete_normalize(var*100) # increase dist between floats before rounding
    else:
        signal = np.zeros(len(fasta)-k+1)
        var = np.zeros(len(fasta)-k+1)
        for kmer_start in range(len(fasta)-k+1):
            signal[kmer_start] = kmer_model[fasta[kmer_start:kmer_start+k]]
            var[kmer_start] = var_model[fasta[kmer_start:kmer_start+k]]
        return discrete_normalize(signal[:R//2]),discrete_normalize(var[:R//2]) # increase dist between floats before rounding

class Read():
    ''' Store FAST5 read data. '''
    def __init__(self, signal, read_id, offset=0, scaling=1.0):                                                                                                                                                                               
        self.signal = signal                                                                                                                                                                                                                  
        self.read_id = read_id                                                                                                                                                                                                                
        self.total_samples = len(signal)                                                                                                                                                                                                      
        self.daq_offset = offset                                                                                                                                                                                                              
        self.daq_scaling = scaling                                                                                                                                                                                                            
        self.read_tag = random.randint(0, int(2**32 - 1))  

        
def ba_preprocess_read(uuid, length):
    ''' Extract read data from FAST5 file for basecalling. '''
    readname = f"read_{uuid}"
    fast5_file = h5py.File(full_index[uuid], 'r')
    signal = np.array(fast5_file[readname]['Raw']['Signal'][:], dtype=np.int16)
    signal, trimmed = trim(signal)
    if len(signal) < max(prefix_lengths): return None
    signal_dig = fast5_file[readname]['channel_id'].attrs['digitisation']
    signal_offset = fast5_file[readname]['channel_id'].attrs['offset']
    signal_range = fast5_file[readname]['channel_id'].attrs['range']
    signal_scaling = signal_range / signal_dig
    return Read(signal, readname, offset=signal_offset, scaling=signal_scaling)

    
def preprocess_read(uuid):
    ''' Return preprocessed read from specified FAST5 file. '''
    readname = f"read_{uuid}"
    #readname = f"{uuid}"
    fast5_file = h5py.File(full_index[uuid], 'r')
    signal = np.array(fast5_file[readname]['Raw']['Signal'][:], dtype=np.int16)
    length = signal.shape[0]

    new_signal=discrete_normalize(signal[1000:1000+max(prefix_lengths)])
    return new_signal,0, length


def get_index(index_filename):
    ''' Read index data structure from file. '''
    index_file = open(index_filename, 'r')
    index = {}
    for line in index_file:
        uuid, fname = re.split(r'\t+', line)
        index[uuid] = fname.rstrip()
    index_file.close()
    return index


def create_index(fast5_dir, force=False):
    ''' Create file which stores read FAST5 to UUID mappings. '''

    # return existing index if possible
    index_fn = f'{fast5_dir}/index.db'
    if not force and os.path.exists(index_fn):
        return get_index(index_fn)

    # remove existing index
    if os.path.exists(index_fn):
        os.remove(index_fn)

    # create new index    
    index_file = open(index_fn, 'w')

    # iterate through all FAST5 files in directory
    for subdir, dirs, files in os.walk(fast5_dir):
        for filename in files:
            ext = os.path.splitext(filename)[-1].lower()
            if ext == ".fast5":

                # print read uuid and filename to index
                fast5_file = h5py.File(os.path.join(subdir, filename), 'r')
                if 'Raw' in fast5_file: # single-FAST5
                    for readname in fast5_file['Raw']['Reads']:
                        uuid = fast5_file['Raw']['Reads'][readname].attrs['read_id']
                        print('{}\t{}'.format(uuid.decode('utf-8'),os.path.join(subdir, filename)), file=index_file)
                else: # multi-FAST5
                    for readname in fast5_file:
                        uuid = readname[5:] # remove 'read_' naming prefix
                        print('{}\t{}'.format(uuid,os.path.join(subdir, filename)), file=index_file)

    # cleanup and return results
    index_file.close()
    return get_index(index_fn)

#================================================scoring schemes==============#


@njit()
def sdtw_default(seq):
    ''' Returns minimum alignment score for subsequence DTW. '''
    
    # initialize column vectors
    cost_mat = np.zeros((len(seq), len(ref)))
    cost_mat[0, 0] = pow(seq[0]-ref[0],2)
    for i in range(1, len(seq)):
        cost_mat[i, 0] = cost_mat[i-1, 0] + pow(seq[i]-ref[0],2)
    for i in range(1, len(ref)):
        cost_mat[0,i] = pow(seq[0]-ref[i],2)
    # compute entire cost matrix
    for i in range(1, len(seq)):
        for j in range(1, len(ref)):
            cost_mat[i, j] = pow(seq[i]-ref[j],2) + min(cost_mat[i-1, j-1], cost_mat[i-1, j])
    for i in range(len(seq)):
        # for j in range(1, len(ref)):
        print("i=",i,"query=",seq[i],"val[0]=",cost_mat[i][0],"val[1]=",cost_mat[i][1],"val[31]=",cost_mat[i][31],"val[32]=",cost_mat[i][32],"val[R-2]=",cost_mat[i][R-2],"val[R-1]=",cost_mat[i][R-1])
        # print("i=",i,"q=",seq[i],"ref[1]=",ref[1],"val[0]=",cost_mat[i, 0],"val[1]=",cost_mat[i, 1],"val[30]=",cost_mat[i, N-2],"val[31]=",cost_mat[i, N-1] )       
    # print(np.array(min(cost_mat[len(seq)-1,:])))     
    return np.array(min(cost_mat[len(seq)-1,:]))

#============================================== floating point kernels============#
#int_sig=0 #######################################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

ref_fasta = get_fasta(ref_fn)
kmer_model,var_model = load_model(kmer_model_fn)
fwd_ref_sig,fwd_var = ref_signal(ref_fasta, kmer_model,var_model)
rev_ref_sig,rev_var = ref_signal(rev_comp(ref_fasta), kmer_model,var_model)
ref = np.concatenate((fwd_ref_sig[:R//2], rev_ref_sig[:R//2]))
var = np.concatenate((fwd_var[:R//2], rev_var[:R//2]))
# ref = np.array(fwd_ref_sig)
# var = np.array(fwd_var)


# create read UUID -> FAST5 filename mapping
virus_index = create_index(virus_fast5_dir)
other_index = create_index(other_fast5_dir)
full_index = {**virus_index, **other_index}

# select random subset of reads
if virus == "covid":
    # COVID reads were reverse-transcribed RNA->DNA, so most are short.
    # To ensure we have enough long reads for evaluating accuracy at longer sDTW read lengths, use a lot.
    random.seed(42)
    virus_readnames = random.choices(list(virus_index.keys()), k=virus_max_reads*1000)
else: 
    random.seed(7)
    virus_readnames = random.choices(list(virus_index.keys()), k=virus_max_reads*2)
other_readnames = random.choices(list(other_index.keys()), k=other_max_reads*2)

# trim all reads
with mp.Pool() as pool:
    virus_reads, virus_trims, virus_lengths =         list(map(list, zip(*filter(None, pool.map(preprocess_read, virus_readnames)))))
    other_reads, other_trims, other_lengths =         list(map(list, zip(*filter(None, pool.map(preprocess_read, other_readnames)))))

# warn user if not enough long reads for accuracy analyses
if len(virus_reads) < virus_max_reads:
    print(f'ERROR: only {len(virus_reads)} virus reads long enough, requested {virus_max_reads}')
if len(other_reads) < other_max_reads:
    print(f'ERROR: only {len(other_reads)} other reads long enough, requested {other_max_reads}')
    
# keep only 'max_reads' for further analysis
virus_reads, virus_trims, virus_lengths = virus_reads[:virus_max_reads],     virus_trims[:virus_max_reads], virus_lengths[:virus_max_reads]
other_reads, other_trims, other_lengths = other_reads[:other_max_reads],     other_trims[:other_max_reads], other_lengths[:other_max_reads]

vf="sdtw_default_virus"
hf="sdtw_default_human"
# for i in range(len(virus_reads[0])):
#     print(virus_reads[0][i])
for i in [0]:
    rtype=i
    with mp.Pool() as pool:
        print(f'rtype i= {rtype} Aligning {virus} reads...sdtw_default', flush=True)
        virus_scores_list = pool.map(sdtw_default, np.array(virus_reads))
print("min scores=",virus_scores_list)