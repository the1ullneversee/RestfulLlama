import json
import os
import shutil
import sys

import torch
from llama_cpp.llama_cpp import _load_shared_library
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# model_url = "microsoft/Phi-3-medium-128k-instruct"
model_url = "models/Llama-3-8B-Instruct-Gradient-1048k-Q8_0.gguf"
import llama_cpp

llm = None
llm = llama_cpp.Llama(
    model_path=model_url,
    # n_threads=12,  # CPU cores
    # n_batch=512,  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    # n_gpu_layers=24,  # Change this value based on your model and your GPU VRAM pool.
    # n_ctx=2048,  # Context window
    # verbose=True,
)

# llm = AutoModelForCausalLM.from_pretrained(
#     model_url,
#     model_kwargs={"torch_dtype": torch.bfloat16},
#     device_map="auto",
#     attn_implementation="flash_attention_2",
# )

inference_params = {
    "do_sample": True,
    "top_p": 0.6,
    "temperature": 0.9,
    "top_k": 50,
    "max_new_tokens": 512,
    "repetition_penalty": 1.03,
    "stop": ["</s>"],
    "return_full_text": False,
}


def get_instructions(user_content, system_content):
    """
    Note: We are creating a fresh user content everytime by initializing instructions for every user_content.
    This is to avoid past user_content when you are inferencing multiple times with new ask everytime.
    """

    instructions = [
        {"role": "system", "content": f"{system_content} "},
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
            conversation.append(
                f"{endPrompt} {instruction['content'].strip()} {stop_token}{startPrompt}"
            )

    return startPrompt + "".join(conversation) + endPrompt


def chat_completion(prompt, system_content):
    instructions = get_instructions(prompt, system_content)
    prompt = build_llama2_prompt(instructions)
    payload = {
        "inputs": prompt,
        "parameters": inference_params,
        "stream": False,  ## <-- to have response stream.
    }
    response = llm(
        prompt=prompt,
        temperature=0.5,
        top_p=0.95,
        repeat_penalty=1.2,
        top_k=50,
        stop=["USER:"],  # Dynamic stopping when such token is detected.
        echo=True,  # return the prompt
    )

    return response["choices"][0]["text"]


# generate me multiple prompts based on the prompt above:
def get_answer(path, response):
    # create a text prompt
    question = f"Given the following endpoint {path}, write me the HTTP request to get the success response"
    # sub_prompt = f"""[INST] <<SYS>>
    # You are to be given questions about REST API Endpoints. Your answers are to be given in code or text.
    # Give no explanation and provide no Note or opening Sure, and only give the answer in the following format, no fluff text:
    # url: <url> headers: <headers> params: <params> body: <body> python code: <code> code needs to be in Python code using the requests library. Do not include imports, just code.
    # <</SYS>>
    # {question}[/INST]"""
    system_content = """
    You are a to be given questions about REST API Endpoints. Your answers are to be given in code or text.
    Give no explanation, and only give the answer in the following format, no fluff text:
    url: <url> headers: <headers> params: <params> body: <body> python code: <code> code needs to be in Python code using the requests library. Do not include imports, just code.
    """
    answer = chat_completion(prompt=question, system_content=system_content)
    return answer, question


def clean_answer(answer):
    try:
        # strip out newlines
        answer = answer.replace("\n", " ")
        # the answer always contains some explanation before the url: tag, so we remove it
    except Exception as exc:
        print(exc)
    return answer


# def question_answer_generator_from_parsed_api(api: ParsedApi, main_path):
#     questions, answers = [], []
#     for path_response in api.paths_with_response():
#         path, response = path_response[0], path_response[1]
#         answer, question = get_answer(path, response)
#         answer = clean_answer(answer)
#         questions.append(question)
#         answers.append(answer)

#     paths = [p_r[0] for p_r in api.paths_with_response()]
#     question = (
#         "Give the following RESTful Endpoint paths. What operations could I perform on them? \n"
#         + json.dumps(paths)
#         + "\n"
#     )
#     system_content = f"""
#         The input will be a list of strings representing resource paths for a RESTFul API. Give you answers as operations a consumer of the API could perform.
#         Write the operations in natural language, and separate them with a comma, no fluff text.
#     """
#     answer = chat_completion(prompt=question, system_content=system_content)
#     answer = clean_answer(answer)

#     question = (
#         "Give the following RESTful Endpoint paths. What Paths require an entity_id to access and can you get that entity from a different path? If you can which path do you need to call? \n"
#         + json.dumps(paths)
#         + "\n"
#     )
#     system_content = f"""
#         The input will be a list of strings representing resource paths for a RESTFul API.
#         Give your response based on which paths require an entity id, and if you can get that entity elsewhere, tell the user how.
#         Write the operations in natural language, and separate them with a comma, no fluff text.
#     """
#     answer = chat_completion(prompt=question, system_content=system_content)
#     answer = clean_answer(answer)
#     questions.append(question)
#     answers.append(answer)

#     dump_to_file(questions, answers, main_path)


readable = 0
broken_files = []
read = False
definitions = {}
parsed_apis = []

system_content = """You are a helpful, smart, kind, and efficient AI assistant. 
                    You always fulfill the user's requests to the best of your ability.
                    You are to be given questions about REST API Endpoints.
                    Where synthetic data is required, you are to generate it to best fit the parameter type.
                    """
question = """For each endpoint path in the following context, write a question that a user might ask that could be answered by that endpoint.
              If the request needs parameters, include data that would be required to make the request.
              <CONTEXT>{context}</CONTEXT>"""

hf_tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    revision="refs/pr/6",
    token="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",
)
# pipe = pipeline(
#     "text2text-generation",
#     model=llm,
#     tokenizer=hf_tokenizer,
#     max_length=1000,
#     token="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",
# )
with open("output.json", "r") as f:
    line = f.readline()
    while line != "":
        json_dict = json.loads(line)
        contexts = list(json_dict.values()).pop()
        path_context = contexts["path_context"]
        full_context = contexts["full_schema_context"]
        chat_messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": question.format(context=full_context)},
        ]
        # Generate completions
        completion = llm.create_chat_completion(messages=chat_messages)
        print(completion)
        line = f.readline()


# main_path = './swagger-files/'


# for file_name in files:
#     try:
#         file = file_name
#         if file in processed_files or file in broken_path:
#           print("processed file or its broken " + file)
#           continue
#         shutil.move(main_path+"output/"+file, processing_path+file)
#         parser = SwaggerParser(swagger_path=processing_path+file)
#         print(f"{parser.specification['info']['description']} has {len(parser.paths)} paths")
#         paths = list(parser.paths.keys())
#         for key, value in parser.specification['definitions'].items():
#             definitions[key] = value
#         parsed_api = parse_api_resources(paths)
#         parsed_apis.append(parsed_api)
#         question_answer_generator_from_parsed_api(parsed_api, main_path)
#         # move file to processed folder.
#         shutil.move(processing_path+file, processed_path+file)
#         readable += 1
#         read = True
#         print(f"--- processed {readable} out of {file_count} ---")

#     except Exception as exc:
#         print(exc)
#         broken_files.append(file)
#         shutil.move(processing_path+file, broken_path+file)
# print(readable)
# print(len(broken_files))
