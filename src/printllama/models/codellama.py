from typing import List, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from typing import Optional
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
        model_cache_dir: str = "/scr/jphilipp/printllama-hgx/pretrained_hf_models/codellama_7b_instruct_hf",
        tokenizer_cache_dir: str = "/scr/jphilipp/printllama-hgx/pretrained_hf_models/codellama_7b_instruct_hf",
        use_flash_attention_2: bool = True,
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
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.model.device)
        output = self.model.generate(
            inputs["input_ids"],
            max_new_tokens=self.max_new_tokens,
            top_p=0.95,
            do_sample=True,
            temperature=0.7,
        )
        return self.tokenizer.decode(output[0])