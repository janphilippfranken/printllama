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
    "metrics/humaneval-patch-control/mistral-7b-instruct-v02-hf/seed1.json",
    "metrics/humaneval-patch-control/huggingfaceh4-zephyr-7b-beta-hf/seed1.json",
    "metrics/humaneval-patch-control/mixtral_7b_instruct/seed1.json",
    "metrics/humaneval-patch-print/mistral-7b-instruct-v02-hf/seed1.json",
    "metrics/humaneval-patch-print/huggingfaceh4-zephyr-7b-beta-hf/seed1.json",
    "metrics/humaneval-patch-print/mixtral_7b_instruct/seed1.json",
    "metrics/humaneval-py-mutants/mistral-7b-instruct-v02-hf/seed1.json",
    "metrics/humaneval-py-mutants/huggingfaceh4-zephyr-7b-beta-hf/seed1.json",
    "metrics/humaneval-py-mutants/mixtral_7b_instruct/seed1.json",
]


# Loading data
datasets = [load_data(path) for path in file_paths]
datasets = [list() for i in range(9)]



# Setting the theme and fon
sns.set_theme(style="darkgrid")
plt.rcParams['font.family'] = 'Avenir'
plt.figure(figsize=(10, 5))
colors = sns.palettes.color_palette("colorblind", 10)

# Calculate means and standard errors using list comprehensions
means = [np.mean(dataset) for dataset in datasets]
std_errs = [1.95 * (np.std(dataset) / np.sqrt(len(dataset))) for dataset in datasets]


print(len(means))
print(len(std_errs))
# Dataset names
dataset_names = ['patch-control (N = 316)', 'patch-print (N = 316)', 'py-mutants (N = 600)']


# Plotting
fig, ax = plt.subplots(figsize=(10, 5))
bar_width = 0.15
opacity = 0.8

# Bar positions
bar_pos_mistral = np.arange(len(dataset_names))
bar_pos_zephyr = [x + bar_width for x in bar_pos_mistral]
bar_pos_gpt = [x + 2 * bar_width for x in bar_pos_mistral]

# Bars for Mistral
ax.bar(bar_pos_mistral, means[0::3], bar_width, alpha=opacity, color=colors[0], yerr=std_errs[0::3], label='mistral-7b-instruct')

# Bars for Zephyr
ax.bar(bar_pos_zephyr, means[1::3], bar_width, alpha=opacity, color=colors[1], yerr=std_errs[1::3], label='zephyr-7b-beta')


ax.bar(bar_pos_gpt, means[2::3], bar_width, alpha=opacity, color=colors[2], yerr=std_errs[2::3], label='mixtral-8x7b-instruct')


# Labels, Title and Custom x-axis
ax.set_xlabel('Dataset')
ax.set_ylabel('Accuracy')
ax.set_title('HumanEvals Accuracy by Dataset and Model')
ax.set_xticks([r + bar_width / 2 for r in range(len(dataset_names))])
ax.set_xticklabels(dataset_names)
ax.legend()

# Show plot
plt.tight_layout()
plt.savefig('TESTINGTESTING123.pdf')
plt.savefig('TESTINGTESTING123.png')