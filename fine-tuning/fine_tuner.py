import json
import os
from datetime import datetime
from io import StringIO

import optuna
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import TrainingArguments
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported

import wandb


def hyperparameter_tuning(
    base_model,
    train_dataset,
    test_dataset,
    tokenizer,
    collator,
    max_seq_length,
    n_trials=20,
    run_hyperparameter_optimization=False,
):
    def objective(trial):
        # Define the hyperparameters to tune (removed LoRA-specific ones)
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
        per_device_train_batch_size = trial.suggest_categorical(
            "per_device_train_batch_size", [1, 2, 4]
        )
        gradient_accumulation_steps = trial.suggest_int(
            "gradient_accumulation_steps", 1, 8
        )
        num_train_epochs = trial.suggest_int("num_train_epochs", 1, 5)
        weight_decay = trial.suggest_loguniform("weight_decay", 1e-4, 1e-1)
        warmup_ratio = trial.suggest_uniform("warmup_ratio", 0.0, 0.2)
        r = trial.suggest_int("r", 8, 32)
        lora_alpha = trial.suggest_int("lora_alpha", 8, 32)

        model = FastLanguageModel.get_peft_model(
            base_model,
            r=r,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            max_seq_length=max_seq_length,
            use_rslora=False,
            loftq_config=None,
            lora_alpha=lora_alpha,
            lora_dropout=0,
        )

        # Create TrainingArguments (adjust as needed for full fine-tuning)
        args = TrainingArguments(
            output_dir=f"outputs/{project_name}/trial_{trial.number}",
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            bf16=True,  # Enable bfloat16 precision
            fp16=False,  # Disable fp16
        )

        # Initialize wandb run
        wandb.init(
            project=project_name,
            config={
                "learning_rate": learning_rate,
                "epochs": num_train_epochs,
                "architecture": "LLM",
                "dataset": "Custom - Conversational Instructions",
                "per_device_train_batch_size": per_device_train_batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "weight_decay": weight_decay,
                "warmup_ratio": warmup_ratio,
                "r": r,
                "lora_alpha": lora_alpha,
            },
            reinit=True,
        )

        # Create SFTTrainer
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            dataset_text_field="messages",
            max_seq_length=max_seq_length,
            tokenizer=tokenizer,
            data_collator=collator,
            args=args,
        )

        # Train the model
        train_result = trainer.train()

        # Log the results to wandb
        wandb.log({"train_loss": train_result.training_loss})

        # Evaluate the model
        eval_result = trainer.evaluate()

        wandb.finish()

        return eval_result["eval_loss"]

    if run_hyperparameter_optimization:
        # Create an Optuna study object
        study = optuna.create_study(
            direction="minimize",
            study_name=project_name,
            storage="sqlite:///./fine-tuning/db.sqlite3",
        )
        # # Optimize the objective function
        study.optimize(objective, n_trials=n_trials)
    else:
        # Load an existing study
        study = optuna.load_study(
            study_name="1722762862_llm-rest-fine-tune_unsloth_llama-3-8b-Instruct",
            storage="sqlite:///./fine-tuning/db.sqlite3",
        )

    best_params = study.best_params

    # Get the best parameters and train the final model
    best_model = FastLanguageModel.get_peft_model(
        base_model,
        r=best_params["r"],
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=best_params["lora_alpha"],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=max_seq_length,
        use_rslora=False,
        loftq_config=None,
    )

    best_args = TrainingArguments(
        output_dir=f"outputs/{project_name}/best_model",
        num_train_epochs=best_params["num_train_epochs"],
        per_device_train_batch_size=best_params["per_device_train_batch_size"],
        gradient_accumulation_steps=best_params["gradient_accumulation_steps"],
        learning_rate=best_params["learning_rate"],
        weight_decay=best_params["weight_decay"],
        warmup_ratio=best_params["warmup_ratio"],
        optim="adamw_8bit",
        logging_steps=1,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        max_grad_norm=0.3,
        max_steps=-1,
        group_by_length=True,
        lr_scheduler_type="constant",
        length_column_name="length",
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
    )

    # # Initialize wandb for the final run
    # wandb.init(
    #     project=project_name,
    #     config={
    #         "learning_rate": best_params["learning_rate"],
    #         "epochs": best_params["num_train_epochs"],
    #         "architecture": "LLM",
    #         "dataset": "Custom - Conversational Instructions",
    #         "per_device_train_batch_size": best_params["per_device_train_batch_size"],
    #         "gradient_accumulation_steps": best_params["gradient_accumulation_steps"],
    #         "weight_decay": best_params["weight_decay"],
    #         "warmup_ratio": best_params["warmup_ratio"],
    #         "r": best_params["r"],
    #         "lora_alpha": best_params["lora_alpha"],
    #     },
    # )

    # Create the final trainer
    final_trainer = SFTTrainer(
        model=best_model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        dataset_text_field="messages",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        data_collator=collator,
        args=best_args,
    )

    # Train the final model
    final_trainer.train()

    # Save the model
    peft_model_id = "/workspace/peft_rest_llama_7b"
    final_trainer.model.save_pretrained(peft_model_id)
    tokenizer.save_pretrained(peft_model_id)

    # wandb.finish()

    return final_trainer, best_params


model_id = "unsloth/llama-3-8b-Instruct"
max_seq_length = 8096  # Supports RoPE Scaling interally, so choose any!
os.environ["WANDB_PROJECT"] = "alpaca_ft"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints
# # # start a new wandb run to track this script
base_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_id,
    max_seq_length=max_seq_length,
    dtype=torch.bfloat16,  # Explicitly set dtype to bfloat16
    load_in_4bit=False,
)

# Use a different token as pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = "<|padding|>"
    tokenizer.pad_token_id = tokenizer.add_special_tokens({"pad_token": "<|padding|>"})

# Ensure eos_token and pad_token are different
if tokenizer.eos_token_id == tokenizer.pad_token_id:
    tokenizer.eos_token = "</s>"
    tokenizer.eos_token_id = tokenizer.add_special_tokens({"eos_token": "</s>"})

# Update the model's embedding layer if new tokens were added
base_model.resize_token_embeddings(len(tokenizer))


def map_dataset(data_line) -> None:
    conversation = json.loads(data_line)
    messages = []
    for msg in conversation:
        if isinstance(msg.get("content"), dict):
            msg["content"] = (
                "{"
                + ", ".join(f"'{k}': '{v}'" for k, v in msg.get("content").items())
                + "}"
            )
        messages.append(msg)
    return messages


with open("./fine-tuning/training_data_set.jsonl", "r") as file:
    data = file.read()
    df = pd.read_json(StringIO(data), lines=True)

# Split the data into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

"""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful AI assistant for travel tips and recommendations<|eot_id|><|start_header_id|>user<|end_header_id|>
What can you help me with?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

# Apply the mapping function to both datasets
train_df["messages"] = train_df.apply(lambda x: map_dataset(x.messages), axis=1)
test_df["messages"] = test_df.apply(lambda x: map_dataset(x.messages), axis=1)

train_df["messages"] = train_df["messages"].apply(
    lambda x: tokenizer.apply_chat_template(
        x, tokenize=False, add_generation_prompt=True
    )
)
test_df["messages"] = test_df["messages"].apply(
    lambda x: tokenizer.apply_chat_template(
        x, tokenize=False, add_generation_prompt=True
    )
)

# Convert the DataFrames to Datasets
train_dataset = Dataset.from_list(
    train_df["messages"].apply(lambda x: tokenizer(x, return_length=True)).to_list()
)
test_dataset = Dataset.from_list(
    test_df["messages"].apply(lambda x: tokenizer(x, return_length=True)).to_list()
)
instruction_template = "<|start_header_id|>user<|end_header_id|>"
response_template = "<|start_header_id|>assistant<|end_header_id|>"
collator = DataCollatorForCompletionOnlyLM(
    instruction_template=instruction_template,
    response_template=response_template,
    tokenizer=tokenizer,
)

project_name = (
    f"{int(datetime.now().timestamp())}_llm-rest-fine-tune_{model_id}".replace("/", "_")
)

final_trainer, best_params = hyperparameter_tuning(
    base_model,
    train_dataset,
    test_dataset,
    tokenizer,
    collator,
    max_seq_length,
    n_trials=20,
    run_hyperparameter_optimization=False,
)

print("Best hyperparameters:", best_params)
