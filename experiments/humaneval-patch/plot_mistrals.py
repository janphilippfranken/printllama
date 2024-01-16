import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textwrap import wrap

def load_data(file_path):
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []


file_paths = [
    "metrics/humaneval-py-mutants/mistral-7b-instruct-v02-hf/seed1.json",
    "metrics/humaneval-py-mutants/mixtral_7b_instruct/seed1.json",
    "metrics/humaneval-patch-control/mistral-7b-instruct-v02-hf/seed1.json",
    "metrics/humaneval-patch-control/mixtral_7b_instruct/seed1.json",
    "metrics/humaneval-patch-print/mistral-7b-instruct-v02-hf/seed1.json",
    "metrics/humaneval-patch-print/mixtral_7b_instruct/seed1.json",
    "metrics/humaneval-patch-011224-temp07-gpt4prints-exploded-selected-prints-gpt4/mistral-7b-instruct-v02-hf/seed1.json",
    "metrics/humaneval-patch-011224-temp07-gpt4prints-exploded-selected-prints-gpt4/mixtral_7b_instruct/seed1.json"
]


# Loading data
datasets = [load_data(path) for path in file_paths]

# Setting the theme and fon
sns.set_theme(style="darkgrid")
plt.rcParams['font.family'] = 'Avenir'
plt.figure(figsize=(5, 5))
colors = sns.palettes.color_palette("colorblind", 10)

# Calculate means and standard errors using list comprehensions
means = [np.mean(dataset) for dataset in datasets]
std_errs = [1.95 * (np.std(dataset) / np.sqrt(len(dataset))) for dataset in datasets]
print(file_paths)
print(means)
print(std_errs)


# Dataset names
dataset_names = [ 'humaneval-py-mutants (N = 600)', 'humaneval-patch-control (N = 316)', 'humaneval-patch-print-\nmixtralselect (N = 316)', 'humaneval-patch-print-\ngpt4select (N = 316)']
dataset_names = ['\n'.join(name.split(' ', 1)) for name in dataset_names]

# Plotting
fig, ax = plt.subplots(figsize=(12, 5))
bar_width = 0.35
opacity = 0.8

# Bar positions
bar_pos_mistral = np.arange(len(dataset_names))
bar_pos_mixtral = [x + bar_width for x in bar_pos_mistral]

# Bars for Mistral
ax.bar(bar_pos_mistral, means[0::2], bar_width, alpha=opacity, color=colors[0], yerr=std_errs[0::2], label='mistral-7b-instruct')

# Bars for mixtral
ax.bar(bar_pos_mixtral, means[1::2], bar_width, alpha=opacity, color=colors[1], yerr=std_errs[1::2], label='mixtral-8x7b-instruct')



# Labels, Title and Custom x-axis
ax.set_xlabel('Dataset')
ax.set_ylabel('Accuracy')
ax.set_title('humaneval-patch Accuracies by Dataset and Model')
ax.set_xticks([r + bar_width / 2 for r in range(len(dataset_names))])
ax.set_xticklabels(dataset_names)

ax.legend()
plt.ylim([0.0, 1.0])

# Show plot
plt.tight_layout()
plt.savefig('figures/mistral-and-mixtral-bothselections.pdf')
plt.savefig('figures/mistral-and-mixtral-bothselections.png')