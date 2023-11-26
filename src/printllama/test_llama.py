from transformers import AutoTokenizer
import transformers
import torch
import os
from transformers import AutoModelForCausalLM
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

base_model = "meta-llama/Llama-2-13b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=base_model,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
    cache_dir="/scr/jphilipp/scai/pretrained_models/Llama-2-7b-chat-hf",
    token=os.getenv("HF_TOKEN"),
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", cache_dir="/scr/jphilipp/scai/pretrained_models/Llama-2-7b-chat-hf", token=os.getenv("HF_TOKEN"))

 
system = """You are selfish."""

user =  """You have to split 10 grams of medicine between yourself and another player. The other player will accept any offer.
             
Make a proposal for how much the other player gets vs how much you get; and be selfish."""

prompt = f"<s>{B_INST} {B_SYS}{system}{E_SYS} {user} {E_INST}"

prompt2 = f"<s>{B_INST}  You love coffeeeeeeee {system}{E_SYS} {user} {E_INST}"

breakpoint()

texts = [prompt] * 100 + [prompt2]

tokenizer.pad_token = tokenizer.eos_token

inputs = tokenizer(texts, padding=True, return_tensors='pt').to("cuda")



breakpoint()

model.eval()
with torch.no_grad():
    result = tokenizer.batch_decode(model.generate(inputs["input_ids"], max_new_tokens=500, do_sample=True, top_p=0.9, temperature=0.1), skip_special_tokens=True)

print(result)
breakpoint()
for res in result.split("\n"):
    print(res)

breakpoint()
