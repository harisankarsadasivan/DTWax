import pickle
import numpy as np
import sys


# prefix_lengths = np.array(range(100,9501,100))

if(int(sys.argv[1])==32):
    with open(str(sys.argv[2])) as f:
        virus_scores = []
        for line in f:
            virus_scores.append(float(line.strip()))
    f.close()
    with open(str(sys.argv[3])) as f:
        other_scores = []
        for line in f:
            other_scores.append(float(line.strip()))
    f.close()
else:
    with open(str(sys.argv[2])+".fwd") as f:
        virus_scores_f = []
        for line in f:
            virus_scores_f.append(float(line.strip()))
    f.close()

    with open(str(sys.argv[2])+".rev") as f:
        virus_scores_r = []
        for line in f:
            virus_scores_r.append(float(line.strip()))
    f.close()
    with open(str(sys.argv[3])+".fwd") as f:
        other_scores_f = []
        for line in f:
            other_scores_f.append(float(line.strip()))
    f.close()

    with open(str(sys.argv[3])+".rev") as f:
        other_scores_r = []
        for line in f:
            other_scores_r.append(float(line.strip()))
    f.close()

    virus_scores,other_scores=[],[]
    for i,j in zip(virus_scores_f,virus_scores_r):
        virus_scores.append(min(i,j))
    for i,j in zip(other_scores_f,other_scores_r):
        other_scores.append(min(i,j))
    

# prefix_lengths = np.array(range(4096,4500,500))
# nprefixes = len(prefix_lengths)
# dtw_indices = [np.where(prefix_lengths == t)[0] for t in prefix_lengths]

# virus_scores_list=pickle.load(open(vf+".pickle","rb"))
# other_scores_list=pickle.load(open(hf+".pickle","rb"))
# virus_scores = np.zeros((nprefixes, len(virus_scores_list)))
# for idx, scores in enumerate(virus_scores_list):
#     for i in range(nprefixes):
#         virus_scores[i,idx] = scores[i]
# other_scores = np.zeros((nprefixes, len(other_scores_list)))

# for idx, scores in enumerate(other_scores_list):
#     for i in range(nprefixes):
#         other_scores[i,idx] = scores[i]
# # virus_scores = np.sort(virus_scores)
# # other_scores = np.sort(other_scores)
# print(virus_scores,other_scores)
# for i, l in zip(dtw_indices, prefix_lengths):

minval = min(np.min(virus_scores), np.min(other_scores))-1
maxval = max(np.max(virus_scores), np.max(other_scores))+1
print(minval,maxval)
print(np.mean(virus_scores),np.mean(other_scores))
thresholds = np.linspace(minval, maxval, num=1000)

other_discard_rate, virus_discard_rate, f1 = [], [],[]
for t in thresholds:
    virus_discard_rate.append(np.sum(virus_scores > t) / np.count_nonzero(virus_scores)) #FN
    other_discard_rate.append(np.sum(other_scores > t) / np.count_nonzero(other_scores)) #TN
    f1.append((1-virus_discard_rate[-1])/(1-virus_discard_rate[-1]+0.5*(1+virus_discard_rate[-1]-other_discard_rate[-1])))
print(1-virus_discard_rate[f1.index(max(f1))],other_discard_rate[f1.index(max(f1))],max(f1)) 