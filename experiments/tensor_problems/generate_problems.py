import hydra
from omegaconf import DictConfig
import fire
import pandas as pd
from tqdm import tqdm
import numpy as np
import logging
from datasets import load_from_disk, Dataset, load_dataset

from scaituning.models.hf_inference_model import HFInferenceModel
from scaituning.prompts.cai.generate_constitution_prompts import (
    PROMPTS, EXAMPLE_FORMATS, build_test_prompt, build_train_prompt
)


EXAMPLE FILE STRUCTURE FOR HOW TO USE HYDRA WITH CONFIGS AND FIRE; CAN LOOK AT THIS TOGETHER

# logging
logging.basicConfig(level=logging.INFO)


# tokenizer constants
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

# system prompts
SYSTEM_TRAIN = "Respond to the best of your ability."
SYSTEM_TEST = "Respond to the best of your ability."


@hydra.main(version_base=None, config_path="config", config_name="generate_constitution")
def main(args: DictConfig) -> None:
    logging.info("Running generate_constitution.py...")
    
    # MODEL
    if not args.generation.verbose:
        model = HFInferenceModel(**args.model.hf_config)
    else: 
        model = None
    
    # DATASET
    for n_shuffle in range(args.data.dataset.n_shuffles):
        if not args.generation.original_dataset:
            dataset = load_from_disk(f"{args.data.dataset.cache_dir}_shuffled_{n_shuffle}") # load the shuffled dataset 
        else: # for debugging load original dataset
            dataset = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", cache_dir="/scr/jphilipp/scai/datasets/hh-rlhf")["train"]
        # shuffle dataset item positions
        dataset = dataset.shuffle() 
       

        # CONSTITUTIONS TO STORE
        constitutions = {
            "constitutions": { # the constitions written by modle 
                k: [] for k in range(args.generation.constitution_batch_size)
            }, 
            "scores": { # how good the constitions are at predicting random labels 
                k: [] for k in range(args.generation.constitution_batch_size)
            },
        }


        # INITIALIZE CONSTITUTIONS AND EVALUATED CONVERSATIONS
        current_constitutions = [args.generation.init_constitution] * args.generation.constitution_batch_size
        current_scores = [0] * args.generation.constitution_batch_size 
        evaluated_conversations = {} # for computing score of constitution
       

        # MAIN LOOP
        for revision_idx in tqdm(range(args.generation.n_revisions)):
            
            # GENERATION
            formatted_train_prompts = [] 
            for constitution_idx in range(args.generation.constitution_batch_size):
                train_prompt = build_train_prompt(
                    constitution=current_constitutions[constitution_idx],
                    train_prompt=PROMPTS[args.prompts.train_prompt],
                    chosen=dataset["chosen"][revision_idx],
                    rejected=dataset["rejected"][revision_idx],
                    example_formats=EXAMPLE_FORMATS,
                    revision_idx=revision_idx,
                )
                # add conversation for evaluation
                evaluated_conversations[revision_idx] = {
                    "chosen": dataset["chosen"][revision_idx],
                    "rejected": dataset["rejected"][revision_idx],
                }
                # format train prompt
                formatted_prompt = f"<s>{B_INST} {B_SYS}{SYSTEM_TRAIN}{E_SYS}{train_prompt} {E_INST}"
                formatted_train_prompts.append(formatted_prompt)
            # generate responses
            if model:
                response = model.batch_prompt(formatted_train_prompts, **args.model.train_config)
                revised_constitutions = [r[0].split(E_INST)[1] for r in response]
            else:
                revised_constitutions = current_constitutions
                logging.info(f"Revised constitution: {revised_constitutions}")
            breakpoint()
            
            # EVALUATION
            n_eval_conversations = min(args.generation.n_evals_per_revision, revision_idx + 1) # number of conversations to evaluate
            rand_eval_conversations = np.random.choice(
                list(evaluated_conversations.keys()), size=n_eval_conversations, replace=False
                ) # sampel conversations to evaluate
            eval_conversations = [evaluated_conversations[k] for k in rand_eval_conversations]
            # batched eval prompts
            batch_formatted_test_prompts = []
            batch_corrected_answers = []
            for revised_constitution in revised_constitutions:
                test_prompts = []
                correct_answers = []
                for eval_conversation in eval_conversations:
                    test_prompt, correct_answer = build_test_prompt(
                        test_prompt=PROMPTS[args.prompts.test_prompt],
                        chosen=eval_conversation["chosen"],
                        rejected=eval_conversation["rejected"],
                        constitution=revised_constitution,
                    )
                    test_prompts.append(test_prompt)
                    correct_answers.append(correct_answer)
                # format test prompts
                formatted_test_prompts = [
                    f"<s>{B_INST} {B_SYS}{SYSTEM_TEST}{E_SYS}{test_prompt} {E_INST}" for test_prompt in test_prompts
                    ]
                batch_formatted_test_prompts += formatted_test_prompts # flattening list for batched call
                batch_corrected_answers.append(correct_answers) # dont have to flatten this one 
            # generate responses
            breakpoint()
            if model:
                response = model.batch_prompt(batch_formatted_test_prompts, **args.model.test_config) # batch_formatted_test_prompts is a list of n_eval_conversations * n_constitutions prompts
                batched_predicted_answers = [r[0] for r in response] # 0 because other returns are log probs, need to update this
                # batch predicted answers into n_constitutions batches of len n_eval_conversations
                batched_predicted_answers = [
                    predicted_answers[i:i + n_eval_conversations] 
                        for i in range(
                            0, len(predicted_answers), 
                            n_eval_conversations
                        )
                    ]
            else:
                batched_predicted_answers = correct_answers
                logging.info(f"Predicted answers: {batched_predicted_answers}")

            breakpoint()
            
            # UPDATING CONSTITUTIONS
            revised_scores = []
            for batch_idx, revised_constitution in enumerate(revised_constitutions):
                predicted_answers = batched_predicted_answers[batch_idx]
                correct_answers = batch_corrected_answers[batch_idx]
                # compute score
                score = sum([
                    1 if correct_answer.lower() in predicted_answer.lower() else 0 for predicted_answer, correct_answer in zip(predicted_answers, correct_answers)
                    ]) / len(predicted_answers)
                revised_scores.append(score)
                # update current constitution and score
                if score > current_scores[batch_idx]:
                    current_constitutions[batch_idx] = revised_constitution
                    current_scores[batch_idx] = score
                    constitutions["constitutions"][batch_idx].append(revised_constitution)
                    constitutions["scores"][batch_idx].append(score)
                else:
                    constitutions["constitutions"][batch_idx].append(current_constitutions[batch_idx])
                    constitutions["scores"][batch_idx].append(current_scores[batch_idx])
        breakpoint()
        # WRITE TO DISK
        constitution_ds = Dataset.from_pandas(pd.DataFrame(constitutions))
        constitution_ds.save_to_disk(f"constitutions_shuffled_{n_shuffle}")


if __name__ == '__main__':
    fire.Fire(main())