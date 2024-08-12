from unsloth import FastLanguageModel 
from unsloth import is_bfloat16_supported
import torch
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments, LlamaTokenizer, TextStreamer, AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import HfApi
import os 
from peft import PeftConfig, PeftModel
import wandb
from datetime import datetime
import json
from sklearn.model_selection import train_test_split
import pandas as pd
import optuna
from io import StringIO

max_seq_length = 4096 # Supports RoPE Scaling interally, so choose any!
os.environ["WANDB_PROJECT"] = "alpaca_ft"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints
os.environ["WANDB_API_KEY"] = "6aca40bdea1861aa55358838e745f71b96114a0e"
# # # start a new wandb run to track this script

 
model_id = "unsloth/llama-3-8b-Instruct-bnb-4bit"
model_id = "unsloth/llama-3-8b-Instruct"

# base_model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name = model_id,
#     max_seq_length = max_seq_length,
#     dtype=torch.bfloat16,  # Explicitly set dtype to bfloat16
#     load_in_4bit=False,
# )

# # Use a different token as pad_token
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = '<|padding|>'
#     tokenizer.pad_token_id = tokenizer.add_special_tokens({'pad_token': '<|padding|>'})

# # Ensure eos_token and pad_token are different
# if tokenizer.eos_token_id == tokenizer.pad_token_id:
#     tokenizer.eos_token = '</s>'
#     tokenizer.eos_token_id = tokenizer.add_special_tokens({'eos_token': '</s>'})

# # Update the model's embedding layer if new tokens were added
# base_model.resize_token_embeddings(len(tokenizer))


# def map_dataset(data_line) -> None:
#     conversation = json.loads(data_line)
#     messages = []
#     for msg in conversation:
#         if isinstance(msg.get('content'), dict):
#             msg['content'] = "{" + ", ".join(f"'{k}': '{v}'" for k, v in msg.get('content').items()) + "}"
#         messages.append(msg)
#     return messages



# with open("/workspace/training_data_set.jsonl", "r") as file:
#     data = file.read()
#     df = pd.read_json(StringIO(data), lines=True)

# # Split the data into training and testing sets
# train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# """
# <|begin_of_text|><|start_header_id|>system<|end_header_id|>
# You are a helpful AI assistant for travel tips and recommendations<|eot_id|><|start_header_id|>user<|end_header_id|>
# What can you help me with?<|eot_id|><|start_header_id|>assistant<|end_header_id|>
# """

# # Apply the mapping function to both datasets
# train_df['messages'] = train_df.apply(lambda x: map_dataset(x.messages), axis=1)
# test_df['messages'] = test_df.apply(lambda x: map_dataset(x.messages), axis=1)

# train_df['messages'] = train_df['messages'].apply(lambda x: tokenizer.apply_chat_template(x, tokenize=False, add_generation_prompt=True))
# test_df['messages'] = test_df['messages'].apply(lambda x: tokenizer.apply_chat_template(x, tokenize=False, add_generation_prompt=True))

# # Convert the DataFrames to Datasets
# train_dataset = Dataset.from_list(train_df['messages'].apply(lambda x: tokenizer(x, return_length=True)).to_list())
# test_dataset = Dataset.from_list(test_df['messages'].apply(lambda x: tokenizer(x, return_length=True)).to_list())
# instruction_template = '<|start_header_id|>user<|end_header_id|>'
# response_template = '<|start_header_id|>assistant<|end_header_id|>'
# collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template, response_template=response_template, tokenizer=tokenizer)

# project_name = f'{int(datetime.now().timestamp())}_llm-rest-fine-tune_{model_id}'.replace("/", "_")

def hyperparameter_tuning(base_model, train_dataset, test_dataset, tokenizer, collator, max_seq_length, n_trials=20):
    def objective(trial):
        # Define the hyperparameters to tune (removed LoRA-specific ones)
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
        per_device_train_batch_size = trial.suggest_categorical('per_device_train_batch_size', [1, 2, 4])
        gradient_accumulation_steps = trial.suggest_int('gradient_accumulation_steps', 1, 8)
        num_train_epochs = trial.suggest_int('num_train_epochs', 1, 5)
        weight_decay = trial.suggest_loguniform('weight_decay', 1e-4, 1e-1)
        warmup_ratio = trial.suggest_uniform('warmup_ratio', 0.0, 0.2) 
        r = trial.suggest_int('r', 8, 32)
        lora_alpha = trial.suggest_int('lora_alpha', 8, 32)

        model = FastLanguageModel.get_peft_model(
            base_model,
            r=r,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",],
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
            reinit=True
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
            args=args
        )
        
        # Train the model
        train_result = trainer.train()
        
        # Log the results to wandb
        wandb.log({"train_loss": train_result.training_loss})
        
        # Evaluate the model
        eval_result = trainer.evaluate()
        
        wandb.finish()
        
        return eval_result['eval_loss']
    
    # Create an Optuna study object
    # study = optuna.create_study(direction='minimize', study_name=project_name, storage="sqlite:///db.sqlite3")
    study = optuna.load_study(study_name='1722762862_llm-rest-fine-tune_unsloth_llama-3-8b-Instruct', storage="sqlite:///db.sqlite3")
    
    # # Optimize the objective function
    # study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params

    # Get the best parameters and train the final model
    best_model = FastLanguageModel.get_peft_model(
        base_model,
        r=best_params['r'],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha=best_params['lora_alpha'],
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
        num_train_epochs=best_params['num_train_epochs'],
        per_device_train_batch_size=best_params['per_device_train_batch_size'],
        gradient_accumulation_steps=best_params['gradient_accumulation_steps'],
        learning_rate=best_params['learning_rate'],
        weight_decay=best_params['weight_decay'],
        warmup_ratio=best_params['warmup_ratio'],
        optim="adamw_8bit",
        logging_steps=1,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        max_grad_norm=0.3,
        max_steps=-1,
        group_by_length=True,
        lr_scheduler_type="constant",
        length_column_name='length',
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
    )
    
    # Initialize wandb for the final run
    wandb.init(
        project=project_name,
        config={
            "learning_rate": best_params['learning_rate'],
            "epochs": best_params['num_train_epochs'],
            "architecture": "LLM",
            "dataset": "Custom - Conversational Instructions",
            "per_device_train_batch_size": best_params['per_device_train_batch_size'],
            "gradient_accumulation_steps": best_params['gradient_accumulation_steps'],
            "weight_decay": best_params['weight_decay'],
            "warmup_ratio": best_params['warmup_ratio'],
            "r": best_params['r'],
            "lora_alpha": best_params['lora_alpha'],
        }
    )
    
    # Create the final trainer
    final_trainer = SFTTrainer(
        model=best_model,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        dataset_text_field="messages",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        data_collator=collator,
        args=best_args
    )
    
    # Train the final model
    final_trainer.train()
    
    # Save the model
    peft_model_id = "/workspace/peft_rest_llama_7b"
    final_trainer.model.save_pretrained(peft_model_id)
    tokenizer.save_pretrained(peft_model_id)
    
    wandb.finish()
    
    return final_trainer, best_params

# _, best_params = hyperparameter_tuning(base_model, train_dataset, test_dataset, tokenizer, collator, max_seq_length)
# print("Best hyperparameters:", best_params)

ALLOWED_QUANTS = \
{
    "q8_0"    : "Fast conversion. High resource use, but generally acceptable.",
    "q4_k_m"  : "Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q4_K",
    "q5_k_m"  : "Recommended. Uses Q6_K for half of the attention.wv and feed_forward.w2 tensors, else Q5_K",
    "q2_k"    : "Uses Q4_K for the attention.vw and feed_forward.w2 tensors, Q2_K for the other tensors.",
    "q3_k_l"  : "Uses Q5_K for the attention.wv, attention.wo, and feed_forward.w2 tensors, else Q3_K",
    "q3_k_m"  : "Uses Q4_K for the attention.wv, attention.wo, and feed_forward.w2 tensors, else Q3_K",
    "q3_k_s"  : "Uses Q3_K for all tensors",
    "q4_0"    : "Original quant method, 4-bit.",
    "q4_1"    : "Higher accuracy than q4_0 but not as high as q5_0. However has quicker inference than q5 models.",
    "q4_k_s"  : "Uses Q4_K for all tensors",
    "q4_k"    : "alias for q4_k_m",
    "q5_k"    : "alias for q5_k_m",
    "q5_0"    : "Higher accuracy, higher resource usage and slower inference.",
    "q5_1"    : "Even higher accuracy, resource usage and slower inference.",
    "q5_k_s"  : "Uses Q5_K for all tensors",
    "q6_k"    : "Uses Q8_K for all tensors",
}

# model_id = "peft_rest_llama_7b"
# trained_model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name = model_id,
#     max_seq_length = 4000,
#     dtype=None,
#     load_in_4bit=False,
# )

# FastLanguageModel.for_inference(trained_model)
#trained_model.push_to_hub_merged("https://huggingface.co/the1ullneversee/llama-7B-Rest", tokenizer, save_method="merged_16bit", token = "")
# for quant in ALLOWED_QUANTS:
#     trained_model.push_to_hub_gguf("https://huggingface.co/the1ullneversee/llama-7B-Rest", tokenizer, quantization_method=quant)


# model_id = "unsloth/llama-3-8b-Instruct-bnb-4bit"
# # git_cmd = https://huggingface.co/unsloth/llama-3-8b-Instruct-bnb-4bit

# Path to your base model
# base_model_path = "unsloth/llama-3-8b-Instruct"

# # Path to your LoRA adapter
# adapter_path = "peft_rest_llama_7b"

# # Path where you want to save the merged model
# merged_model_path = "restful-llama"

# # Load the base model
# base_model = AutoModelForCausalLM.from_pretrained(
#     base_model_path,
#     torch_dtype=torch.float16,
#     device_map="auto",
#     token="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"
# )

# # Load the LoRA config
# config = PeftConfig.from_pretrained(adapter_path)

# # Load the LoRA model
# model = PeftModel.from_pretrained(base_model, adapter_path)

# # Merge the LoRA model with the base model
# model = model.merge_and_unload()

# # Save the merged model
# model.save_pretrained(merged_model_path)

# # Save the tokenizer
# 
# tokenizer.save_pretrained(merged_model_path)

# print(f"Merged model saved to {merged_model_path}")

model_repo_name = "the1ullneversee/RestfulLlama-8B-Instruct"  # Format of Input  <Profile Name > / <Model Repo Name> 


# trained_model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name = "restful_llama_gguf_8/restful_llama_8_Q8.gguf",
#     max_seq_length = 4000,
#     dtype=None,
#     load_in_4bit=False,
# )
tokenizer = AutoTokenizer.from_pretrained("restful-llama")
api = HfApi()
#Create Repo in Hugging Face
api.create_repo(repo_id=model_repo_name, token='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')

#Upload Model folder from Local to HuggingFace 
api.upload_folder(
    folder_path='restful_llama_gguf_8',
    repo_id=model_repo_name,
    token='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'
)
# Publish Model Tokenizer on Hugging Face
tokenizer.push_to_hub(model_repo_name, token='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')
