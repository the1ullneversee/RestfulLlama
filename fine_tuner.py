import torch
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import PeftModel, LoraConfig, prepare_model_for_kbit_training, get_peft_model

model_id='NousResearch/Llama-2-7b-hf'
max_length = 512
device_map = "auto"
batch_size = 128
micro_batch_size = 32
gradient_accumulation_steps = batch_size // micro_batch_size

from transformers import AutoModelForCausalLM
compute_dtype = getattr(torch, "float16")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

# load model from huggingface
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quant_config,
    use_cache=False,
    device_map=device_map
)
model.config.use_cache = False
model.config.pretraining_tp = 1


from datasets import load_dataset
data_files = {"train": "../train_dataset.jsonl", "test": "../test_dataset.jsonl"}

main_dataset = load_dataset('json', data_files=data_files)
test, train = main_dataset['test'], main_dataset['train']
# load tokenizer from huggingface
tokenizer = AutoTokenizer.from_pretrained("TheBloke/Llama-2-13B-chat-GPTQ", token="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")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    target_modules=["q_proj", "k_proj","v_proj","o_proj"], # the name of the layers to add LoRA
    task_type="CAUSAL_LM",
    modules_to_save=None, # layers to unfreeze and train from the original pre-trained model
)

def get_instructions(user_content, system_content):
    instructions = [
        { "role": "system","content": f"{system_content} "},
    ]

    instructions.append({"role": "user", "content": f"{user_content}"})

    return instructions

def build_llama2_prompt(instructions):
    stop_token = "</s>"
    start_token = "<s>"
    startPrompt = f"{start_token}[INST] "
    endPrompt = " [/INST]"
    conversation = []
    for index, instruction in enumerate(instructions):
        if instruction["role"] == "system" and index == 0:
            conversation.append(f"<<SYS>>\n{instruction['content']}\n<</SYS>>\n\n")
        elif instruction["role"] == "user":
            conversation.append(instruction["content"].strip())
        else:
            conversation.append(f"{endPrompt} {instruction['content'].strip()} {stop_token}{startPrompt}")

    return startPrompt + "".join(conversation) + endPrompt

from trl import SFTTrainer
import json

def generate_prompt(dataset: list[dict]):
    processed = []
    for i in range(len(dataset["question"])):
        question = dataset["question"][i]
        answer = dataset["answer"][i]
        question_answer = {}
        question_answer['question'] = question
        question_answer['answer'] = answer
        question = ""
        system_content_a = '''
        You are a to be given questions about REST API Endpoints. Your answers are to be given in code or text.
        Give no explanation, and only give the answer in the following format, no fluff text:
        url: <url> headers: <headers> params: <params> body: <body> python code: <code> code needs to be in Python code using the requests library. Do not include imports, just code.
        '''
        system_content_b = f'''
            The input will be a list of strings representing resource paths for a RESTFul API. Give you answers as operations a consumer of the API could perform.
            Write the operations in natural language, and separate them with a comma, no fluff text.
        '''
        system_content_c = f'''
            The input will be a list of strings representing resource paths for a RESTFul API.
            Give your response based on which paths require an entity id, and if you can get that entity elsewhere, tell the user how.
            Write the operations in natural language, and separate them with a comma, no fluff text.
        '''
        question = question_answer['question']
        system_content = ""
        if 'entity_id' in question:
            system_content = system_content_c
        else:
            system_content = system_content_a
        system_content_a = '''
        You are a to be given questions about REST API Endpoints. Your answers are to be given in code or text.
        Give no explanation, and only give the answer in the following format, no fluff text:
        url: <url> headers: <headers> params: <params> body: <body> python code: <code> code needs to be in Python code using the requests library. Do not include imports, just code.
        '''
        instructions = get_instructions(question, system_content)
        prompt = build_llama2_prompt(instructions)
        whole_prompt = f"{prompt} " + f"Answer: {answer}" 
        processed.append(whole_prompt)
    return processed

batch_size = 8
gradient_accumulation_steps = 2
num_train_epochs = 3

training_params = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size//3,
    gradient_accumulation_steps=gradient_accumulation_steps,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=20,
    warmup_ratio=0.03,
    group_by_length=False,
    lr_scheduler_type="constant",
    report_to="tensorboard"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train,
    peft_config=peft_params,
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_params,
    formatting_func=generate_prompt,
    packing=False,
)

trainer.train()
