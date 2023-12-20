import os


#problems = ['attentionproblem', 'trilproblem', 'maskingproblem', 'row-wisemeanproblem']
problems = ['maindiagonalproblem']
models = ['codellama-7b-meta', 'codellama-7b-instruct-meta', 'codellama-13b-instruct-meta', 'codellama-34b-instruct-meta', 'huggingfaceh4-zephyr-7b-beta-hf', 'mistral-7b-instruct-v02-hf']
numgpus = {
    'codellama-7b-meta' : 1,
     'codellama-7b-instruct-meta' : 1,
      'codellama-13b-instruct-meta' : 2,
       'codellama-34b-instruct-meta' : 4,
       'huggingfaceh4-zephyr-7b-beta-hf' : 1, 
       'mistral-7b-instruct-v02-hf' : 1
}

for model in models:
    NPROC_PER_NODE = numgpus[model]
    for problem in problems:
        directory = f'conf/data/{problem}/'
        for filename in os.listdir(directory):
            if filename.endswith('.yaml'):
                print(filename[:-5])
                try:
                    os.system(f'torchrun --nproc_per_node {NPROC_PER_NODE} eval_model.py model={model} data={problem}/{filename[:-5].strip()}')
                except:
                    continue