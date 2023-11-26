from transformers import AutoTokenizer
import transformers
import torch
from transformers import AutoModelForCausalLM
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

base_model = "meta-llama/Llama-2-13b-chat-hf",
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
    cache_dir="/scr/jphilipp/scai/pretrained_models/Llama-2-13b-chat-hf",
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-13b-chat-hf", cache_dir="/scr/jphilipp/scai/pretrained_models/Llama-2-13b-chat-hf"),

 
system = """You are a smart psychologist."""

user =  """You are given interactions between two agents: 
Agent A: I have to split 10 dollars between you and me. I will give you 1 dollar and keep 9 dollars for myself. 
Agent B: I will accept the offer.
             
Predict what the values agent A might have. Return your response as a list of values, starting with the most likely value. Return 5 values."""

prompt = f"<s>{B_INST} {B_SYS}{system}{E_SYS} {user} {E_INST}"

breakpoint()
        
inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")

model.eval()
with torch.no_grad():
    result = tokenizer.decode(model.generate(inputs["input_ids"], max_new_tokens=200,
    do_sample=True,
    top_p=0.9,
    temperature=0.1,)[0])

print(result)
breakpoint()
