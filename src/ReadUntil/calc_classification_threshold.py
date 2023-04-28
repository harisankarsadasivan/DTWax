import argparse
import numpy as np

# Define a function to parse the microbial classification file
def parse_microbial_file(file_path):
    with open(file_path) as f:
        lines = f.readlines()

 
    dtw_scores = []
    for i in range(0, len(lines), 1):
            
            score_line = lines[i].split()
            if (score_line[0]=="Read_ID"):
                continue
            dtw_scores.append(min(float(score_line[4]),float(score_line[5])))

    return np.array(dtw_scores)

# Define a function to parse the human classification file
def parse_human_file(file_path):
    with open(file_path) as f:
        lines = f.readlines()

    # Skip the first line and read every line after that
    dtw_scores = []
    for i in range(0, len(lines), 1):
            
            score_line = lines[i].split()
            if (score_line[0]=="Read_ID"):
                continue
            dtw_scores.append(min(float(score_line[4]),float(score_line[5])))

    return np.array(dtw_scores)

# Parse the command-line arguments
parser = argparse.ArgumentParser(description='Calculate DTW classification accuracy thresholds')
parser.add_argument('microbial_file', help='path to the microbial classification file')
parser.add_argument('human_file', help='path to the human classification file')
args = parser.parse_args()

# Parse the microbial and human files
human_scores = parse_human_file(args.human_file)
microbial_scores = parse_microbial_file(args.microbial_file)

print(len(microbial_scores))
print(len(human_scores))
# Calculate the optimal classification threshold
thresholds = np.linspace(0, max(max(microbial_scores),max(human_scores)), num=100000)
accuracy_scores = []
for threshold in thresholds:
    microbial_correct = np.sum(microbial_scores < threshold)
    human_correct = np.sum(human_scores >= threshold)
    accuracy = (microbial_correct + human_correct)*100.0 / (len(microbial_scores) + len(human_scores))
    accuracy_scores.append(accuracy)

best_threshold = thresholds[np.argmax(accuracy_scores)]

# Print the optimal classification threshold
print(f'Best classification threshold: {best_threshold}, accuracy: {max(accuracy_scores)}')
