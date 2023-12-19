
from typing import Optional, List, Dict

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class HFInferenceModel():
    """
    Wrapper for running inference with HF Model.
    """
    def __init__(
        self, 
        model_id: str, # unique identifier for accessing conversation buffer
        pretrained_model_name_or_path: str = "mistralai/Mistral-7B-Instruct-v0.1",
        load_in_8bit: str = True,
        device_map: str = "auto",
        torch_dtype: str = "float16",
        model_cache_dir: str = "/scr/jphilipp/scai/pretrained_models/Mistral-7B-Instruct-v0.1",
        tokenizer_cache_dir: str = "/scr/jphilipp/scai/pretrained_models/Mistral-7B-Instruct-v0.1",
        ):
        """Initializes HF Inference Model"""
        self.model_id = model_id
        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path, 
            cache_dir=tokenizer_cache_dir,
            token=os.getenv("HF_TOKEN"),
        )
        # check which model we are using
        is_mistral = "mistral" in pretrained_model_name_or_path.lower()
        is_llama_2 = "llama-2" in pretrained_model_name_or_path.lower() or "llama" in pretrained_model_name_or_path.lower() 
        is_zephyr = "zephyr" in pretrained_model_name_or_path.lower()
        is_starcoder = "starcoder" in pretrained_model_name_or_path.lower()
        if is_mistral:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        elif is_llama_2:
            self.tokenizer.pad_token = "[PAD]"
            self.tokenizer.padding_side = "left"
        elif is_zephyr:
            self.tokenizer.pad_token = "[PAD]"
            self.tokenizer.padding_side = "left"
        elif is_starcoder:
            pass
        else:
            raise ValueError(f"Model not implemented: {pretrained_model_name_or_path}")
        # load model
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            load_in_8bit=load_in_8bit,
            torch_dtype=torch.float16 if "16" in torch_dtype else torch.float32,
            device_map=device_map,
            cache_dir=model_cache_dir,
            token=os.getenv("HF_TOKEN"),
        )

    @property
    def model_type(self):
        return "HFInferenceModel"

    def get_log_probs(
        self, 
        prompts: List[str],
        answer: str,
    ) -> float:
        """
        Returns log probability of answer.
        """
        token_id_answer_batch = self.tokenizer(
            [prompt + answer + "</s>" for prompt in prompts],
            add_special_tokens=True, return_tensors="pt",
        )
        sequence_logprob_answer_batch = self.model(
            input_ids=token_id_answer_batch.input_ids,
            attention_mask=token_id_answer_batch.attention_mask,
        )
        logits = sequence_logprob_answer_batch.logits[:, :-1]
        labels = token_id_answer_batch.input_ids[:, 1:]
    
        cross_entropy = torch.nn.CrossEntropyLoss(reduction="none")

        log_probs = -cross_entropy(
            logits.view(-1, logits.shape[-1]), 
            labels.view(-1)
        )
        log_probs = log_probs.view(logits.shape[0], -1)
        log_probs = log_probs.sum(dim=-1)        
        return log_probs
            
    def batch_prompt(self, 
        prompts: List[str],
        max_new_tokens: Optional[int] = 500,
        do_sample: Optional[bool] = True,
        top_p: Optional[float] = 0.9,
        temperature: Optional[float] = 0.1,
        log_probs: Optional[bool] = False,
        log_probs_answer: Optional[bool] = True,
        log_probs_mcq: Optional[bool] = False,
        answer_a: Optional[str] = "A",
        answer_b: Optional[str] = "B",  
    ):
        """
        Batched inferences (either generate or log_probs)
        """
        inputs = self.tokenizer(prompts, 
                        return_tensors="pt", 
                        padding=True).to(self.model.device)
        if not log_probs:
            output = self.model.generate(
                inputs["input_ids"], 
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
            )
            output = self.tokenizer.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            return output
        else: 
            if log_probs_answer:
                breakpoint()
                # TODO: Get this to work
                # get log probs of full sequence + answer 
                log_probs_a = self.get_log_probs(prompts, answer_a)
                log_probs_b = self.get_log_probs(prompts, answer_b)
                results = {
                    "log_probs_answer_a": log_probs_a,
                    "log_probs_answer_b": log_probs_b,
                }
                
                return results
            elif log_probs_mcq:
                # ask model to generate answer
                output = self.model.generate(
                    inputs["input_ids"], 
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    top_p=top_p,
                    temperature=temperature,
                    output_scores=True,  
                    return_dict_in_generate=True,
                )
                token_id_answer_a = self.tokenizer.encode(answer_a, add_special_tokens=False)[0]
                token_id_answer_b = self.tokenizer.encode(answer_b, add_special_tokens=False)[0]
                results = {
                    "log_probs_answer_a": [],
                    "log_probs_answer_b": [],
                }
                breakpoint()
                for i in range(len(prompts)):
                    log_probs = torch.nn.functional.log_softmax(output.scores[0][i], dim=-1)
                    log_prob_answer_a = log_probs[token_id_answer_a].item()
                    log_prob_answer_b = log_probs[token_id_answer_b].item()
                    results["log_probs_answer_a"].append(log_prob_answer_a)
                    results["log_probs_answer_b"].append(log_prob_answer_b)
                return results
            