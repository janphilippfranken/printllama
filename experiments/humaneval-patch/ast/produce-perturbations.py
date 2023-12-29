import hydra
from omegaconf import DictConfig
import argparse
import fire
import pandas as pd
import logging
import random


from utils import Perturber
from utils import extract_candidate_calls, handler, check_executable


# constants
N_TRIES = 100
SEED = 1


# logging
logging.basicConfig(level=logging.INFO)


@hydra.main(version_base=None, config_path="../conf", config_name='config')
def main(args: DictConfig) -> None:
    bugfactory = Perturber()
    random.seed(SEED)
    

    def perturb_humaneval(filepath):
        df = pd.read_csv(filepath)
        df['solution'] = df['prompt'] + df['canonical_solution']

        var_perturbations, expr_perturbations, func_perturbations = list(), list(), list()


        for i, row in df.iterrows():
            print(i)
            solution, test, entry_point = row['solution'], row['test'], row['entry_point']


            for _ in range(N_TRIES):
                try:
                    var_bug = bugfactory.randomly_modify_variable(solution)
                    if not bugfactory.same_tree(solution, var_bug) and check_executable(var_bug, test, entry_point):
                        var_perturbations.append(var_bug)
                        break
                except:
                    pass
            else:
                # The else block of a for loop is executed when the loop completes normally, i.e., without hitting a break statement
                var_perturbations.append(None)


            for _ in range(N_TRIES):
                try:
                    expr_bug = bugfactory.randomly_modify_expression(solution)
                    if not bugfactory.same_tree(solution, expr_bug) and check_executable(expr_bug, test, entry_point):
                        expr_perturbations.append(expr_bug)
                        break
                except:
                    pass
            else:
                # The else block of a for loop is executed when the loop completes normally, i.e., without hitting a break statement
                expr_perturbations.append(None)


            for _ in range(N_TRIES):
                try:
                    func_bug = bugfactory.randomly_modify_functioncall(solution)
                    if not bugfactory.same_tree(solution, func_bug) and check_executable(func_bug, test, entry_point):
                        func_perturbations.append(func_bug)
                        break
                except:
                    pass
            else:
                # The else block of a for loop is executed when the loop completes normally, i.e., without hitting a break statement
                func_perturbations.append(None)


        return list(df['solution']), var_perturbations, expr_perturbations, func_perturbations

    solution, variable_changes, expression_changes, func_changes = perturb_humaneval(f'../data/{args.data.path}')
    print(f'Dataset size: {len(solution)}')
    print(f'Successful variable perturbations: {len([1 for example in variable_changes if example is not None])}')
    print(f'Successful expression perturbations: {len([1 for example in expression_changes if example is not None])}')
    print(f'Successful function perturbations: {len([1 for example in func_changes if example is not None])}')
    
    
    original_df = pd.read_csv(f'../data/{args.data.path}')
    perturb_df = pd.DataFrame({
        'solution' : solution,
        'var changes' : variable_changes,
        'expr changes' : expression_changes,
        'func changes' : func_changes
    })
    
    
    df = pd.concat([original_df, perturb_df], axis=1)
    var_long = df.drop(['expr changes', 'func changes'], axis=1)
    var_long['bugtype'] = 'var'
    expr_long = df.drop(['var changes', 'func changes'], axis=1)
    expr_long['bugtype'] = 'expr'
    func_long = df.drop(['var changes', 'expr changes'], axis=1)
    func_long['bugtype'] = 'func'


    var_long.columns = [*var_long.columns[:-2], 'bug', 'bugtype']
    expr_long.columns = var_long.columns
    func_long.columns = var_long.columns
    
    
    out = pd.concat([var_long, expr_long, func_long])
    out.to_csv('../data/humaneval-patch-122723.csv')
    

if __name__ == '__main__':
    try:
        fire.Fire(main())
    except:
        pass