import os
NPROC_PER_NODE = 2
directory = 'conf/data/attentionproblem/'


for filename in os.listdir(directory):
    if filename.endswith('.yaml'):
        print(filename[:-5])
        try:
            os.system(f'torchrun --nproc_per_node {NPROC_PER_NODE} eval_model.py model=codellama-13b-instruct-meta data=attentionproblem/{filename[:-5].strip()}')
        except:
            continue
