import pickle
import numpy as np

prefix_lengths = np.array(range(500,9501,500))
nprefixes = len(prefix_lengths)
dtw_indices = [np.where(prefix_lengths == t)[0] for t in prefix_lengths]

virus_scores_list=pickle.load(open("sdtw_linear_bonus_compressed_virus.pickle","rb"))
other_scores_list=pickle.load(open("sdtw_linear_bonus_compressed_human.pickle","rb"))
virus_scores = np.zeros((nprefixes, len(virus_scores_list)))
for idx, scores in enumerate(virus_scores_list):
    for i in range(nprefixes):
        virus_scores[i,idx] = scores[i]
other_scores = np.zeros((nprefixes, len(other_scores_list)))

for idx, scores in enumerate(other_scores_list):
    for i in range(nprefixes):
        other_scores[i,idx] = scores[i]
# virus_scores = np.sort(virus_scores)
# other_scores = np.sort(other_scores)

for i, l in zip(dtw_indices, prefix_lengths):
    i = int(i)
    minval = min(np.min(virus_scores[i]), np.min(other_scores[i]))-1
    maxval = max(np.max(virus_scores[i]), np.max(other_scores[i]))+1
    thresholds = np.linspace(minval, maxval, num=100)

    other_discard_rate, virus_discard_rate, f1 = [], [],[]
    for t in thresholds:
        virus_discard_rate.append(np.sum(virus_scores[i] > t) / np.count_nonzero(virus_scores[i])) #FN
        other_discard_rate.append(np.sum(other_scores[i] > t) / np.count_nonzero(other_scores[i])) #TN
        f1.append((1-virus_discard_rate[-1])/(1-virus_discard_rate[-1]+0.5*(1+virus_discard_rate[-1]-other_discard_rate[-1])))
    print(i,l,1-virus_discard_rate[f1.index(max(f1))],other_discard_rate[f1.index(max(f1))])   