from typing import List, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# from torch.nn import DataParallel


class CodeLlama():
    """
    Wrapper for HF CodeLlama model.
    """
    def __init__(
        self, 
        pretrained_model_name_or_path: str = "codellama/CodeLlama-7b-hf",
        load_in_8bit: str = True,
        device_map: str = "auto",
        torch_dtype: str = "float16",
        model_cache_dir: str = "/scr/jphilipp/printllama-hgx/pretrained_hf_models/codellama_7b_hf",
        tokenizer_cache_dir: str = "/scr/jphilipp/printllama-hgx/pretrained_hf_models/codellama_7b_hf",
        max_new_tokens: int = 2000,
        ):
        """
        Initializes CodeLLama.
        """
        torch_dtype = torch.float16 if "16" in torch_dtype else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path, 
            cache_dir=tokenizer_cache_dir)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            load_in_8bit=load_in_8bit,
            torch_dtype=torch_dtype,
            device_map=device_map,
            cache_dir=model_cache_dir,
        )


        self.max_new_tokens = max_new_tokens

    @property
    def llm_type(self):
        return "HFCodeLlama"
    
    def __call__(self, 
        prompt: str,
    ):
        """
        Make a call.
        """
        prompt = """def remove_non_ascii(s: str) -> str:
<FILL_ME>
    return result"""
        print(prompt)
        model_input = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)["input_ids"]
        with torch.no_grad():
            generated_ids = self.model.generate(model_input, max_new_tokens=self.max_new_tokens)
            response = self.tokenizer.batch_decode(generated_ids[:, model_input.shape[1]:], skip_special_tokens = True)[0]
            breakpoint()
            # response = self.tokenizer.decode(self.model.generate(model_input, max_new_tokens=self.max_new_tokens)[0], skip_special_tokens=True)
        return response