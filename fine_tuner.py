import os, sys
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    GenerationConfig
)
#from peft import PeftModel, LoraConfig, prepare_model_for_kbit_training, get_peft_model

model_id='NousResearch/Llama-2-7b-hf'
max_length = 512
device_map = "auto"
batch_size = 128
micro_batch_size = 32
gradient_accumulation_steps = batch_size // micro_batch_size

# nf4" use a symmetric quantization scheme with 4 bits precision
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
#     load_in_8bit_fp32_cpu_offload=True,
# )

# load model from huggingface
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     quantization_config=bnb_config,
#     use_cache=False,
#     device_map=device_map
# )

from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
  device_map='auto',
  load_in_8bit=True,
  pretrained_model_name_or_path='NousResearch/Llama-2-7b-hf',
  max_memory=f'{int(torch.cuda.mem_get_info()[0]/1024**3)-2}GB')

# load tokenizer from huggingface
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# LoRA config
model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)
