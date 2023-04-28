import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

parser = argparse.ArgumentParser(description='Reads a file and plots a bar chart based on DTW mapping scores.')
parser.add_argument('filename', type=str, help='Name of file to read')
parser.add_argument('threshold', type=float, help='Threshold for DTW mapping score')
args = parser.parse_args()

# Set up initial plot
fig, ax = plt.subplots()
ax.set_xlabel('Category', fontweight='bold', fontsize=18)
ax.set_ylabel('Number of classified DNA reads', fontweight='bold', fontsize=18)
ax.set_xticks([0, 1])
ax.set_xticklabels(['microbe (target)', 'Human (non-target)'], fontweight='bold', fontsize=18)
ax.tick_params(axis='both', labelsize=18)

# Define colors for the bars
colors = ['red', 'grey']

# Set up the bars with the colors
bars = ax.bar([0, 1], [0, 0], color=colors)

# Set up the bar labels
bar_labels = [ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(int(bar.get_height())),
                      ha='center', va='bottom', fontweight='bold', fontsize=18) for bar in bars]

# Define animate function to update plot
def animate(i):
    with open(args.filename, 'r') as f:
        # Seek to last position
        f.seek(animate.pos)

        # Read new lines
        for line in f:

            line = line.split()
            if len(line) == 6:
                if (line[0] == "Read_ID"):
                    continue
                score = min(float(line[4]), float(line[5]))
                if score > args.threshold:
                    bars[1].set_height(bars[1].get_height() + 1)                    
                else:
                    bars[0].set_height(bars[0].get_height() + 1)
                # Update bar colors
                for i, bar in enumerate(bars):
                    bar.set_color(colors[i])

                # Update bar labels
                for i, bar_label in enumerate(bar_labels):
                    bar_label.set_position((bars[i].get_x() + bars[i].get_width() / 2, bars[i].get_height()))
                    bar_label.set_text(str(int(bars[i].get_height())))
                
                ax.relim()
                ax.autoscale_view()

        # Update file position
        animate.pos = f.tell()

# Set initial file position
with open(args.filename, 'r') as f:
    animate.pos = f.tell()

# Set up plot to call animate() function periodically
ani = animation.FuncAnimation(fig, animate, interval=1000)
fig.suptitle('DTWatch', fontsize=30, fontweight='bold', color='#004c9a')

# Display plot in Visual Studio Code
plt.show()
