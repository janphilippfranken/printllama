import random
import os
import csv
import argparse
import pandas as pd
import hydra
from omegaconf import DictConfig
import argparse
import fire
import pickle

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from printllama.helpers import extract_code
from printllama.models.openai.azure import AsyncAzureChatLLM
from printllama.models.openai.gpt4 import GPT4Agent

from utils import REPAIR_PROMPT_FORMAT, PRINT_SYSTEM_MESSAGE, EXPERT_DIR, PATCH_DIR


def chunker(seq, size):
    """
    Returns chunks of a list seq as sublists of length size
    """
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))



def generate_prints(args):
    """
    Generate action, harm, good, preventable cause, external non-prventable cause for both conditions 
    """
    # GET INFERENCE MODEL TYPE (HF, OPENAI, ETC.)
    is_meta = "meta" in args.model.model_type.lower()
    is_hf = "hf" in args.model.model_type.lower()
    is_openai = "openai" in args.model.model_type.lower()
    
    if not (is_openai):
            print(f"Model type {args.model.model_type} not yet supported.")
            return []
        
    if is_openai:
        args.model.model_config.azure_api.api_key = os.getenv("OPENAI_API_KEY")
        llm = AsyncAzureChatLLM(**args.model.model_config.azure_api)
        model = GPT4Agent(llm=llm, **args.model.run.completion_config)

    # Load humaneval-patch dataset
    df = pd.read_csv(PATCH_DIR)
    
    # Load expert prints from manual print dataset
    expert_df = pd.read_csv(EXPERT_DIR)
    
    # get few-shot bugs and corresponding expert prints
    bugs = df[expert_df['bugtype'].str.endswith('print')]['bug'].tolist()
    expert_prints = expert_df[expert_df['bugtype'].str.endswith('print')]['bug'].tolist()
    
    # keep only first 30 rows for test run
    df = df[df['bug'].notnull()].head(30)
    
    
   
    prompts = list()
    # Loop over all problems and generate few-shot prompts
    for i, row in df[df['bug'].notnull()].iterrows():
        few_shot_indices = random.sample(range(0, len(expert_prints)), 3)
        prompt = []
        for j in few_shot_indices:
            prompt.append(bugs[j])
            prompt.append(f"```\n{expert_prints[j]}\n```")
        prompt.append(row['bug'])
        prompts.append(prompt)
    
    # insertions, a list of len(df) lists of size args.run.completion_config.n each
    insertions = list()
    for prompt_chunk in chunker(prompts, args.model.run.batch_size):
        # TODO: Add hf, meta, and vllm functionality here
        if not (is_openai):
            print(f"Model type {args.model.model_type} not yet supported.")
            return []
        
        if is_openai:
            completions = model.batch_prompt(
                    system_message=PRINT_SYSTEM_MESSAGE, 
                    messages=prompt_chunk,
            )
            extracted_completions = [[extract_code(insertion) for insertion in insertion_attempts] for insertion_attempts in completions]
        insertions.extend(extracted_completions)
    
    
    # create new print dataframe
    print_df = df[df['bug'].notnull()].copy(deep=True)
    print_df['bugtype'] = print_df['bugtype'].astype(str) + '-print'
    print_df['bug'] = insertions
    print_df_exploded = print_df.explode('bug')
    
    print_df.to_csv('/sailhome/andukuri/research_projects/printllama/experiments/humaneval-patch/data/humaneval-patch-010924-gpt4prints.csv')
    print_df_exploded.to_csv('/sailhome/andukuri/research_projects/printllama/experiments/humaneval-patch/data/humaneval-patch-010924-gpt4prints-exploded.csv')
    
    
    
    return insertions
    
    
        
        

@hydra.main(version_base=None, config_path="../conf", config_name='config')
def main(args: DictConfig):
    insertions = generate_prints(args)
    
    return 0

if __name__ == "__main__":
    try:
        fire.Fire(main())
    except:
        pass