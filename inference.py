import transformers
import torch



from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "MaziyarPanahi/Meta-Llama-3-70B-Instruct-GGUF"
filename = "Meta-Llama-3-70B-Instruct.Q4_K_M.gguf"

# tokenizer = AutoTokenizer.from_pretrained(model_id, gguf_file=filename)
model = AutoModelForCausalLM.from_pretrained(filename)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

# terminators = [
#     tokenizer.eos_token_id,
#     tokenizer.convert_tokens_to_ids("<|eot_id|>")
# ]

chat_prompt = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Tell me a joke!"},
]

# Generate chat completions
output = model(chat_prompt, max_tokens=50, stop=["</s>"], echo=True)
print(output)