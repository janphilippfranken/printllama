import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import FancyBboxPatch

def get_fancy_bbox(bb, boxstyle, color, background=False, mutation_aspect=3):
    """
    Creates a fancy bounding box for the bar plots. Adapted from Eric's function.
    """
    if background:
        height = bb.height - 2
    else:
        height = bb.height
    if background:
        base = bb.ymin # - 0.2
    else:
        base = bb.ymin
    return FancyBboxPatch(
        (bb.xmin, base),
        abs(bb.width), height,
        boxstyle=boxstyle,
        ec="none", fc=color,
        mutation_aspect=mutation_aspect, # change depending on ylim
        zorder=2
    )


sns.set_theme(style="darkgrid")
plt.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.size'] = 30

colors = sns.palettes.color_palette("colorblind", 10)
color = colors[0]

# Data
data = {
    "Option 1": [0.1, 0.2, 0.15, 0.15, 0.19],
    "Option 2": [0.03, 0.06, 0.1, 0.05, 0.05],
    "Option 3": [0.41, 0.31, 0.34, 0.25, 0.31],
    "Option 4": [0.35, 0.33, 0.31, 0.32, 0.38]
}

# Convert to DataFrame
df_accuracy = pd.DataFrame(data)

# Calculate mean and SEM (Standard Error of the Mean)
mean_accuracy = df_accuracy.mean()
sem_accuracy = df_accuracy.sem()



# Plotting
fig, ax = plt.subplots(figsize=(10, 5))
plt.bar(x=[0, 1, 2, 3], height=mean_accuracy, yerr=sem_accuracy.values, capsize=.2, color=color)
plt.title('Average Accuracy ± SEM for Each Option')
plt.ylabel('Average Accuracy')
plt.xlabel('Options')

plt.xticks(ticks=[0, 1, 2, 3], labels=['no print', 'print index', 'print A', 'print A slice'])
plt.ylim(-0.05, 1.05)
plt.yticks(ticks=[0, 0.25, 0.5, 0.75, 1.0])

for patch in ax.patches:
    bb = patch.get_bbox()
    color = patch.get_facecolor()
    p_bbox = get_fancy_bbox(bb, "round,pad=-0.005,rounding_size=0.02", color, mutation_aspect=0.1)
    patch.remove()
    ax.add_patch(p_bbox)


plt.tight_layout()
plt.savefig('accuracy.png', dpi=1000)

# Data for number evaluated
data_evaluated = {
    "Option 1": [16, 25, 16, 16, 26],
    "Option 2": [20, 22, 23, 24, 25],
    "Option 3": [47, 46, 43, 37, 47],
    "Option 4": [42, 44, 44, 46, 48]
}

# Convert to DataFrame
df_evaluated = pd.DataFrame(data_evaluated)

# Calculate mean and SEM
mean_evaluated = df_evaluated.mean()
sem_evaluated = df_evaluated.sem()

# Plotting
plt.figure(figsize=(10, 5))
plt.bar(x=[0, 1, 2, 3], height=mean_evaluated, yerr=sem_evaluated.values, capsize=.2, color=color)
plt.title('Average Number Evaluated ± SEM for Each Option')
plt.ylabel('Average Number Evaluated')
plt.xlabel('Options')
plt.xticks(ticks=[0, 1, 2, 3], labels=['no print', 'print index', 'print A', 'print A slice'])
plt.ylim(-2, 105)
plt.yticks(ticks=[0, 25, 50, 75, 100])
plt.tight_layout()
plt.savefig('number_evaluated.png', dpi=1000)
