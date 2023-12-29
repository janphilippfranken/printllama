import json
import pandas as pd
import numpy as np
import scipy 
import matplotlib.pyplot as plt

lst = json.load(open(f"metrics/{args.data.path}/{args.model.name}/seed{args.model.run.seed}.json", "r"))
df = pd.read_csv(f'data/{args.data.path}')

# PREPROCESS
INDICES = list(df['bug'].notnull())
lst = [item for item, flag in zip(lst, INDICES) if flag]

# Calculating the mean and standard error of the mean (SEM)
mean = np.mean(lst)
sem = scipy.stats.sem(lst)

# Plotting
plt.bar(1, mean, yerr=sem, capsize=5)
plt.xticks([1], ['Mean'])
plt.ylabel('Value')
plt.title('Bar Plot with Mean and SEM')
plt.tight_layout()
plt.show()
plt.savefig('test.png')

breakpoint()