import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []

# File paths
file_paths = [
    ["metrics/humaneval-patch-control/mistral-7b-instruct-v02-hf/seed1.json",
    # "metrics/humaneval-patch-control/huggingfaceh4-zephyr-7b-beta-hf/seed1.json",
    "metrics/humaneval-patch-control/mixtral_7b_instruct/seed1.json",],
    ["metrics/humaneval-patch-print/mistral-7b-instruct-v02-hf/seed1.json",
    # "metrics/humaneval-patch-print/huggingfaceh4-zephyr-7b-beta-hf/seed1.json",
    "metrics/humaneval-patch-print/mixtral_7b_instruct/seed1.json",],
    ["metrics/humaneval-py-mutants/mistral-7b-instruct-v02-hf/seed1.json",
    "metrics/humaneval-py-mutants/huggingfaceh4-zephyr-7b-beta-hf/seed1.json",]
    #"metrics/humaneval-py-mutants/mixtral_7b_instruct/seed1.json",
]
categories = ['humaneval-patch-control', 'humaneval-patch-print', 'humaneval-py-mutants']

# Loading data
datasets = [[load_data(path) for path in category] for category in file_paths]




# Setting the theme and font
sns.set_theme(style="darkgrid")
plt.rcParams['font.family'] = 'Avenir'
plt.figure(figsize=(10, 5))
colors = sns.palettes.color_palette("colorblind", 10)

# Calculate means and standard errors using list comprehensions
means = [np.mean(data) for category in datasets for data in category]
std_errs = [1.95 * (np.std(data) / np.sqrt(len(data))) for category in datasets for data in category]


print(len(means))
print(len(std_errs))
# Dataset names
#dataset_names = ['patch-control (N = 492)', 'patch-control (N = 30)', 'patch-expert-print (N = 30)', 'py-mutants (N = 600)']
dataset_names = ['patch-control (N = 316)', 'patch-control (N = 30)', 'patch-expert-print (N = 30)', 'py-mutants (N = 600)']


# Plotting
# Calculate the positions of each bar dynamically
num_categories = len(datasets)
bar_width = 0.15
gap_between_groups = 0.1  # Adjust this gap as needed
opacity = 0.8
# Initialize the starting position
current_bar_pos = 0
bar_positions = []

for category in datasets:
    category_positions = []
    for _ in category:
        category_positions.append(current_bar_pos)
        current_bar_pos += bar_width
    # Add gap between groups
    current_bar_pos += gap_between_groups
    bar_positions.extend(category_positions)

# Flatten the dataset names if necessary
dataset_names = ['Category {} (N = {})'.format(i+1, len(category)) for i, category in enumerate(datasets)]
flattened_dataset_names = [item for sublist in datasets for item in sublist]

# Plotting
fig, ax = plt.subplots(figsize=(10, 5))

colors = sns.color_palette("colorblind", len(flattened_dataset_names))

for i, (mean, std_err, pos) in enumerate(zip(means, std_errs, bar_positions)):
    ax.bar(pos, mean, bar_width, alpha=opacity, color=colors[i % len(colors)], yerr=std_err, label=f'List {i+1}')

# Adding labels, title, and custom x-axis ticks
ax.set_xlabel('Dataset')
ax.set_ylabel('Accuracy')
ax.set_title('HumanEvals Accuracy by Dataset and Model')

# Set the position of the x-ticks for each group (category)
ax.set_xticks([np.mean(bar_positions[i:i+len(category)]) for i, category in enumerate(datasets)])

# Set the labels for the x-ticks
ax.set_xticklabels(dataset_names)

# If you want a legend, you can create custom labels for each category list
handles = [plt.Rectangle((0,0),1,1, color=colors[i % len(colors)]) for i in range(len(flattened_dataset_names))]
ax.legend(handles, [f'List {i+1}' for i in range(len(flattened_dataset_names))])


# Show plot
plt.tight_layout()
plt.savefig("/sailhome/andukuri/research_projects/printllama/experiments/humaneval-patch/REALTEST.png")
plt.show()
