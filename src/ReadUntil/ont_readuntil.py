import numpy
from read_until import ReadUntilClient
import os
import time
import struct
import threading
import random
import subprocess
import sysv_ipc

# Define pipe name and size
PIPE_NAME = "pipe"
# Define semaphore name and value
SEM_KEY= 7
SEM_VALUE = 0

RDS = 10
batch_size=RDS
chunk_size=4000
SAMPLES = 4000*batch_size

# Create named pipe
try:
    os.mkfifo(PIPE_NAME)
except OSError as e:
    if e.errno != os.errno.EEXIST:
        raise

def analysis(client, pipe_fd,*args, **kwargs):    
    
    # keep track of the unique reads we have seen
    unique_reads = set()
    count=0
    global batch_size
    pack_data=[]
    global chunk_size
    while client.is_running:
        for channel, read in client.get_read_chunks(batch_size=batch_size, last=True):
            if (read.id not in unique_reads):
                unique_reads.add(read.id)
                count=count+1
                raw_data = numpy.fromstring(read.raw_data, client.signal_dtype)    
                client.stop_receiving_read(channel, read.number)
                client.unblock_read(channel, read.number)
                pack_data=pack_data+raw_data.tolist()
                #print(pack_data)
                if(count%2==0):
                    pipe_fd = os.open(PIPE_NAME, os.O_WRONLY)
                    os.write(pipe_fd, read.raw_data[-chunk_size*4:])        
                    time.sleep(0.125)
                    pack_data=[]
                # print(count,read.chunk_classifications)      
                #print(count)
                if count==batch_size:
                    #print("batch")
                    count=0
                    unique_reads = set()
            else:
                self.client.stop_receiving_read(channel,read.number);                    

read_until_client = ReadUntilClient()
read_until_client.run()

cpp_process = subprocess.Popen(["../main", "../ref/dna_kmer_model.txt", "../ref/reference.fasta",PIPE_NAME])


# Open named pipe for writing
pipe_fd = os.open(PIPE_NAME, os.O_WRONLY)

# Start thread to write to pipe
thread = threading.Thread(target=analysis, args=(read_until_client,pipe_fd,))
thread.start()



while cpp_process.poll() is None:
    time.sleep(5)

print("completed!!!!!!!!!!")
# Wait for the thread to finish
thread.join()

# Close named pipe
os.close(pipe_fd)
os.unlink(PIPE_NAME)



