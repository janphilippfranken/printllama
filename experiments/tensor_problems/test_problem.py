import os
NPROC_PER_NODE = 1
directory = 'conf/data/trilproblem/'


for filename in os.listdir(directory):
    if filename.endswith('.yaml'):
        print(filename[:-5])
        try:
            os.system(f'torchrun --nproc_per_node {NPROC_PER_NODE} eval_model.py model=gpt4 data=trilproblem/{filename[:-5].strip()}')
        except:
            continue