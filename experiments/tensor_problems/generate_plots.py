import hydra
from omegaconf import DictConfig
import argparse
import fire
import json
import logging
import random
import re
import time
import os

#import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np
import seaborn as sns

from matplotlib.patches import FancyBboxPatch


# logging
logging.basicConfig(level=logging.INFO)

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


@hydra.main(version_base=None, config_path="conf", config_name='config')
def main(args: DictConfig) -> None:
    logging.info("Plotting metrics...")

    seed = 1

    models = ['codellama-7b-meta', 'codellama-7b-instruct-meta', 'codellama-13b-instruct-meta', 'codellama-34b-instruct-meta', 'huggingfaceh4-zephyr-7b-beta-hf', 'mistral-7b-instruct-v02-hf']
    
    def many_problems():
        plt.rcParams.update({'font.size': 8})

        overall_control_accuracies = []
        overall_print_accuracies = []
        overall_control_evalcounts = []
        overall_print_evalcounts = []
        # LOAD ATTENTION PROBLEM METRICS
        for model_name in models:
            model_best_control_accuracies, model_best_print_accuracies = [], []
            for problem_name in ['trilproblem', 'maskingproblem', 'row-wisemeanproblem', 'maindiagonalproblem']:
                directory = f'metrics/{problem_name}/{model_name}/seed{seed}/'
                control_accuracies = []
                print_accuracies = []
                control_evalcounts = []
                print_evalcounts = []
                
                control_conditions, print_conditions = list(), list()
                
                for filename in os.listdir(directory):
                    with open(f'{directory}{filename}', 'r') as f:
                        metrics = json.load(f)
                    if 'print' in metrics['ID'] or '_P_' in metrics['ID']:
                        print_accuracies.append(metrics['Overall accuracy'])
                        print_evalcounts.append(metrics['Evaluated # out of 100'])
                        print_conditions.append(filename)
                    else:
                        control_accuracies.append(metrics['Overall accuracy'])
                        control_evalcounts.append(metrics['Evaluated # out of 100'])
                        control_conditions.append(filename)

                    #print(f"Loaded {problem_name}/{filename} metrics for {model_name}.")
                
                best_control = max(control_accuracies)
                best_control_condition = control_conditions[np.array(control_accuracies).argmax()]
                
                best_print = max(print_accuracies)
                best_print_condition = print_conditions[np.array(print_accuracies).argmax()]
                
                print(f"{model_name}'s best " + color.CYAN + "control" + color.END + f" condition for {problem_name} is {best_control_condition}.")
                print(f"{model_name}'s best " + color.RED + "print" + color.END + f" condition for {problem_name} is {best_print_condition}.")
                
                model_best_control_accuracies.append(best_control)
                model_best_print_accuracies.append(best_print)
                
                

                # model_control_accuracies.append(sum(control_accuracies) / len(control_accuracies) if len(control_accuracies) else 0.0)
                # model_print_accuracies.append(sum(print_accuracies) / len(print_accuracies) if len(print_accuracies) else 0.0)
                # model_control_evalcounts.append(sum(control_evalcounts) / len(control_evalcounts) if len(control_evalcounts) else 0.0)
                # model_print_evalcounts.append(sum(print_evalcounts) / len(print_evalcounts) if len(print_evalcounts) else 0.0)
                
            overall_control_accuracies.append(model_best_control_accuracies)
            # overall_control_evalcounts.append(model_control_evalcounts)
            overall_print_accuracies.append(model_best_print_accuracies)
            # overall_print_evalcounts.append(model_print_evalcounts)


        
        #makeplot(means, sems, f'{args.model.name} Overall')
        control_means = [np.mean(lst) for lst in overall_control_accuracies]
        control_sems = [stats.sem(lst) for lst in overall_control_accuracies]
        print_means = [np.mean(lst) for lst in overall_print_accuracies]
        print_sems = [stats.sem(lst) for lst in overall_print_accuracies]
        # Sample data
        categories = models

    # Number of categories
        n_categories = len(categories)

        # X locations for the groups
        ind = np.arange(n_categories)  
        width = 0.35  # the width of the bars

        # Generate bar plots
        fig, ax = plt.subplots()
        rects1 = ax.bar(ind - width/2, control_means, width, yerr=control_sems, label='Control', capsize=5, color='lightskyblue')
        rects2 = ax.bar(ind + width/2, print_means, width, yerr=print_sems, label='Print', capsize=5, color='salmon')
        
        # Add some text for labels, title and custom x-axis tick labels, etc.
        sns.set_theme(style="darkgrid")
        sns.despine(left=True, bottom=False)
        ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5, zorder=-100)
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy by model and condition')
        ax.set_xticks(ind)
        ax.set_xticklabels(categories, fontsize=8, rotation=45, ha='right')
        ax.legend()
        ax.set_facecolor("whitesmoke")
        plt.tight_layout()

        # Function to attach a text label above each bar in *rects*, displaying its height.
        # def autolabel(rects):
        #     for rect in rects:
        #         height = rect.get_height()
        #         ax.annotate('{:.2f}'.format(height),
        #                     xy=(rect.get_x() + rect.get_width() / 2, height),
        #                     xytext=(0, 3),  # 3 points vertical offset
        #                     textcoords="offset points",
        #                     ha='center', va='bottom')

        # # Call the function for each set of bars.
        # autolabel(rects1)
        # autolabel(rects2)


        plt.savefig('TEST no attention.png')
        
   
    many_problems()

    

if __name__ == '__main__':
    try:
        fire.Fire(main())
    except:
        pass


