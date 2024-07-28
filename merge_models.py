import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Path to your base model
base_model_path = "meta-llama/Meta-Llama-3-8B-Instruct"

# Path to your LoRA adapter
adapter_path = "the1ullneversee/llama-7B-Rest"

# Path where you want to save the merged model
merged_model_path = "restful-llama"

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Load the LoRA config
config = PeftConfig.from_pretrained(adapter_path)

# Load the LoRA model
model = PeftModel.from_pretrained(base_model, adapter_path)

# Merge the LoRA model with the base model
model = model.merge_and_unload()

# Save the merged model
model.save_pretrained(merged_model_path)

# Save the tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.save_pretrained(merged_model_path)

print(f"Merged model saved to {merged_model_path}")
