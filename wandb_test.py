import os
from trl import SFTTrainer
from transformers import AutoModelForCausalLM
from datasets import load_dataset
import torch
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import PeftModel, LoraConfig, prepare_model_for_kbit_training, get_peft_model
from fine_tuner import generate_prompt
import wandb

os.environ["WANDB_PROJECT"] = "alpaca_ft"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints
os.environ["WANDB_API_KEY"] = "a89022a323f21514f65b632439d68dc16451d2d2"
# start a new wandb run to track this script

wandb.init(
    # set the wandb project where this run will be logged
    project="my-awesome-project",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.02,
    "architecture": "LLM",
    "dataset": "Custom",
    "epochs": 1,
    }
)
model_id='NousResearch/Llama-2-7b-hf'


data_files = {"train": "train_dataset.jsonl", "test": "test_dataset.jsonl"}

main_dataset = load_dataset('json', data_files=data_files)
eval_dataset, train_dataset = main_dataset['test'], main_dataset['train']

training_args = TrainingArguments(
    report_to="wandb", # enables logging to W&B 😎
    per_device_train_batch_size=16,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    num_train_epochs=3,
    gradient_accumulation_steps=2, # simulate larger batch sizes
)

# load model from huggingface
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    use_cache=False,
    device_map='auto'
)

trainer = SFTTrainer(
    model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    packing=True, # pack samples together for efficient training
    max_seq_length=1024, # maximum packed length 
    args=training_args,
    formatting_func=generate_prompt, # format samples with a model schema
)
trainer.train()
