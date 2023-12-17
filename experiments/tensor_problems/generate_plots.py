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

from matplotlib.patches import FancyBboxPatch


# logging
logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="conf", config_name='config')
def main(args: DictConfig) -> None:
    logging.info("Plotting metrics...")


    directory = f'metrics/{args.data.problem_name}/{args.model.name}/'

    results = {}
    
    # LOAD METRICS
    for filename in os.listdir(directory):
        if filename.startswith('SEED100') and filename.endswith('.json'):
            try:
                with open(f'{directory}{filename}', 'r') as f:
                    metrics = json.load(f)
                results[metrics['ID']] = metrics
                print(f"Loaded {filename} metrics.")
            except:
                continue
            
            
            
            
            
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


    #sns.set_theme(style="darkgrid")
    plt.rcParams['font.family'] = 'Avenir'
    plt.rcParams['font.size'] = 30

    #colors = sns.palettes.color_palette("colorblind", 10)
    #color = colors[0]
    color='r'






    
    CONTROL = ['C_NP_H_NHD_NID', 'C_NP_NH_NHD_NID']
    PRINT = ['1_P_H_HD_NID', '1_P_H_NHD_ID', '1_P_H_NHD_NID', '1_P_NH_NHD_ID', '1_P_NH_NHD_NID', '2_P_H_HD_NID', '2_P_H_NHD_ID', '2_P_H_NHD_NID', '2_P_NH_NHD_ID', '2_P_NH_NHD_NID']
    PRINT_1 = ['1_P_H_HD_NID', '1_P_H_NHD_ID', '1_P_H_NHD_NID', '1_P_NH_NHD_ID', '1_P_NH_NHD_NID']
    PRINT_2 = ['2_P_H_HD_NID', '2_P_H_NHD_ID', '2_P_H_NHD_NID', '2_P_NH_NHD_ID', '2_P_NH_NHD_NID']
    
    
    
    # COMPILE OVERALL ACCURACIES
    CONTROL_overallaccuracies = [results[f'{args.model.name}_{key}']['Overall accuracy'] for key in CONTROL]
    PRINT_overallaccuracies = [results[f'{args.model.name}_{key}']['Overall accuracy'] for key in PRINT]
    PRINT_1_overallaccuracies = [results[f'{args.model.name}_{key}']['Overall accuracy'] for key in PRINT_1]
    PRINT_2_overallaccuracies = [results[f'{args.model.name}_{key}']['Overall accuracy'] for key in PRINT_2]
    
    
    
    # CONTROL vs PRINT plots
    means = [np.mean(lst) for lst in (CONTROL_overallaccuracies, PRINT_overallaccuracies)]
    sems = [stats.sem(lst) for lst in (CONTROL_overallaccuracies, PRINT_overallaccuracies)]    
    
    
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.bar(x=[0, 1], height=means, yerr=sems, capsize=.2, color=color)
    plt.title('Average Accuracy ± SEM for Each Group')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Group Condition')

    plt.xticks(ticks=[0, 1], labels=['No print', 'Print'])
    plt.ylim(-0.05, 0.5)
    plt.yticks(ticks=[0, 0.5, 1.0])

    for patch in ax.patches:
        bb = patch.get_bbox()
        color = patch.get_facecolor()
        p_bbox = get_fancy_bbox(bb, "round,pad=-0.005,rounding_size=0.02", color, mutation_aspect=0.1)
        patch.remove()
        ax.add_patch(p_bbox)


    plt.tight_layout()
    plt.show()
    plt.savefig('testplotgeneration_accuracy.png', dpi=1000)
    breakpoint()
    
    
    # COMPILE # EVALUATABLE OUTPUTS PER 100
    CONTROL_evaluatable = [results[f'{args.model.name}_{key}']['Overall accuracy'] for key in CONTROL]
    PRINT_evaluatable = [results[f'{args.model.name}_{key}']['Overall accuracy'] for key in PRINT]
    PRINT_1_evaluatable = [results[f'{args.model.name}_{key}']['Overall accuracy'] for key in PRINT_1]
    PRINT_2_evaluatable = [results[f'{args.model.name}_{key}']['Overall accuracy'] for key in PRINT_2]
    
    
    # CONTROL vs PRINT_1 vs .... vs PRINT_N plot
    data_accuracy = {
        'CONTROL' : CONTROL_overallaccuracies,
        'PRINT_1' : PRINT_1_overallaccuracies,
        'PRINT_2' : PRINT_2_overallaccuracies
    }
    
    data_evaluatable = {
        'CONTROL' : CONTROL_evaluatable,
        'PRINT_1' : PRINT_1_evaluatable,
        'PRINT_2' : PRINT_2_evaluatable
    }
    
    
    

    

if __name__ == '__main__':
    try:
        fire.Fire(main())
    except:
        pass
