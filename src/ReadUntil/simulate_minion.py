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

RDS = 512
SAMPLES = 2048*512

# Create named pipe
try:
    os.mkfifo(PIPE_NAME)
except OSError as e:
    if e.errno != os.errno.EEXIST:
        raise

# Define function to generate random samples
def generate_samples():
    samples = []
    for i in range(SAMPLES):
        samples.append(random.randint(0,255))
    return samples

# Define function to write samples to pipe
def write_to_pipe(pipe_fd):
    while True:
        # print("before pipe open")
        pipe_fd = os.open(PIPE_NAME, os.O_WRONLY)
        # print("after pipe open")
        samples = generate_samples()
        data = struct.pack("{}B".format(len(samples)), *samples)
        # print("after packing")
        # Wait for C++ program to consume the data and release the semaphore
        #semaphore.acquire()
        # print("before pipe write")
        os.write(pipe_fd, data)        
        time.sleep(0.5)
        # print("after pipe write")
        # Signal semaphore to notify C++ subprocess that new data is available
        #semaphore.release()
     
        #break

# try:
#     # Try to open the semaphore
#     semaphore = sysv_ipc.Semaphore(SEM_KEY, sysv_ipc.IPC_CREAT, initial_value=SEM_VALUE)
#     print("Semaphore already exists")
# except sysv_ipc.ExistentialError:
# If it doesn't exist, create it with the specified initial value
#semaphore = sysv_ipc.Semaphore(SEM_KEY, sysv_ipc.IPC_CREAT, initial_value=SEM_VALUE)
#print("Semaphore created")

# Start C++ program as child process
cpp_process = subprocess.Popen(["../main", "../ref/dna_kmer_model.txt", "../ref/reference.fasta",PIPE_NAME])


# Open named pipe for writing
pipe_fd = os.open(PIPE_NAME, os.O_WRONLY)

# Start thread to write to pipe
thread = threading.Thread(target=write_to_pipe, args=(pipe_fd,))
thread.start()



while cpp_process.poll() is None:
    time.sleep(5)

print("completed!!!!!!!!!!")
# Wait for the thread to finish
thread.join()

# Close named pipe
os.close(pipe_fd)
# Remove named semaphore and named pipe
#semaphore.remove()
os.unlink(PIPE_NAME)
