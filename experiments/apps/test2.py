
from datasets import load_dataset
import matplotlib.pyplot as plt
ds = load_dataset('json', data_files="../../data/apps_100_llama_prints.json")
ds


def get_accuracies(ds):
    accuracies = {feature: [] for feature in ds.features if 'accuracy' in feature}
    for i in range(len(ds)):
        for feature in accuracies:
            accuracies[feature].append(ds[i][feature])
   
    return accuracies

accuracies = get_accuracies(ds['train'])
# plt.hist(accuracies['baseline_repaired_solution_accuracy'])
print(accuracies)

import matplotlib.pyplot as plt
import numpy as np

def plot_accuracies(accuracies):

    plt.bar(range(len(accuracies)), [np.mean(accuracies[feature]) for feature in accuracies], align='center', alpha=0.5)
    plt.xticks(range(len(accuracies)), [feature for feature in accuracies])
    plt.ylabel('Accuracy')
    plt.title('Accuracy per feature')
    plt.xticks(rotation=90)
    plt.legend()
    plt.show()
    plt.savefig('accuracies.png')

plot_accuracies(accuracies)