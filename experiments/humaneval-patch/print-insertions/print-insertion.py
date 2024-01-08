import random
import csv
import argparse
import pandas as pd

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from utils import PROMPT_FORMAT, EXPERT_DIR


def generate_prints(args):
    """
    Generate action, harm, good, preventable cause, external non-prventable cause for both conditions 
    """
    # GET INFERENCE MODEL TYPE (HF, OPENAI, ETC.)
    is_meta = "meta" in args.model.model_type.lower()
    is_hf = "hf" in args.model.model_type.lower()
    is_openai = "openai" in args.model.model_type.lower()


    # Load expert prints from manual print dataset
    expert_df = pd.read_csv(EXPERT_DIR)
    expert_prints = expert_df[expert_df['bugtype'].str.endswith('print')]['bugtype'].tolist()
    
    

   
    # Loop over all problems
    for i, name in enumerate(names[args.start:args.end]):
        profession = professions[i + args.start]

        rand_item = 0#  random.randint(0, args.start - 1) # random example for few shot generation set to 1
        

        for condition in ['CC', 'CoC']:
            prompt_path = f'{PROMPT_DIR}/{condition.lower()}_stage_1_{severity.lower()}.txt'
            system_prompt = open(prompt_path, 'r').read().strip()

            example = get_example(names, professions, condition, rand_item, severity)

            system_message = SystemMessage(content=system_prompt)
            human_message_0 = get_human_msg(names[rand_item], professions[rand_item])
            ai_message_0 = AIMessage(content=example)
            human_message_1 = get_human_msg(name, profession, f"Reminder: You must follow this structure: {system_prompt}")

            # Messages sent to model
            msgs = [system_message, human_message_0, ai_message_0, human_message_1]
            responses = llm.generate([msgs], stop=["System:"], **args.model.run.completion_config)

            for generation in responses.generations[0]:
                if args.verbose:
                    print(f"------ Generated Story ------")
                    print(generation.text)
                    print("------------ Fin --------------")

                vars = get_vars_from_out(generation.text)
                assert(len(vars) == 5) # TODO - change this later
                breakpoint()
                with open(f'{DATA_DIR}/{condition.lower()}_stage_1_{severity.lower()}.csv', 'a') as csvfile:
                    writer = csv.writer(csvfile, delimiter=';')
                    writer.writerow(vars)

                # breakpoint()

def gen_story(args):
    pass  

parser = argparse.ArgumentParser()
parser.add_argument('--start', type=int, default=10, help='start index')
parser.add_argument('--end', type=int, default=12, help='end index')
parser.add_argument('--model', type=str, default='openai/gpt-4-0613', help='model name')
parser.add_argument('--temperature', type=float, default=0, help='temperature')
parser.add_argument('--max_tokens', type=int, default=300, help='max tokens')
# change num completions to 10
parser.add_argument('--num_completions', type=int, default=1, help='number of completions')
parser.add_argument('--num_shots', type=int, default=3, help='number of shots')
parser.add_argument('--num_stories', type=int, default=2, help='number of stories to generate')
parser.add_argument('--verbose', type=bool, default=True, help='verbose')
parser.add_argument('--api', type=str, default='azure', help='which api to use')

if __name__ == "__main__":
    args = parser.parse_args()
    gen_chat(args)