import hashlib
import json
from collections import defaultdict

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('NousResearch/Hermes-2-Theta-Llama-3-70B', trust_remote_code=True)

model_path = "Meta-Llama-3-70B-Instruct.Q4_K_M.gguf"
model_path = "models/Hermes-2-Pro-Llama-3-8B-Q8_0.gguf"
import llama_cpp

llm = llama_cpp.Llama(
    model_path=model_path,
    n_threads=32,  # CPU cores
    n_batch=512,  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    n_gpu_layers=24,  # Change this value based on your model and your GPU VRAM pool.
    n_ctx=8000,  # Context window
    verbose=True,
)

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

import json


def dump_empty(output_file, paths_hash):
    with open(output_file, 'a') as file:
        file.write(json.dumps({"hash": paths_hash, "questions": []}) + '\n')

def dump_data(output_file, paths_hash, multi_stage_questions, single_stage_questions):
    with open(output_file, 'a') as file:
        file.write(json.dumps({"hash": paths_hash, "questions": multi_stage_questions, "single_stage_questions": single_stage_questions}) + '\n')

def format_questions_response(input_str, output_file, paths_hash):
    # Split the input string into lines
    lines = input_str.split('\n')
    questions = []
    for line in lines:
        # Check if the line is a question (does not contain "No match")
        if not line:
            continue 
        if 'Question' not in line:
            continue
        questions.append(line)
    return questions


def extract_resource_name(path):
    components = path.split("/")
    # Iterate in reverse to find the first non-parameter component
    for component in reversed(components):
        if not (component.startswith("{") and component.endswith("}")):
            return component  # This is the resource name
    return None  # In case no non-parameter component is found

def prompt_llama(messages):
    #chat_template = build_llama3_prompt(prompt, system_content)
    
    # Generate a response from the model
    answer = ""
    try:
        answer = llm.create_chat_completion(
            messages=messages
        )
    except Exception as exc:
        print(exc)
        return answer
    return answer["choices"][0]["message"]

def get_index_to_start(output_file):
    index = 0
    with open(output_file, 'r') as file:
        lines = file.readlines()
        index = len(lines)
    return index


def create_path_hash(paths: list):
    concat = ''.join(paths)

    hash_object = hashlib.sha256(concat.encode())
    hash_hex = hash_object.hexdigest()
    return hash_hex

def multi_stage_questioning(first_context, second_context):
    question_placeholder = """<CONTEXT>{context}</CONTEXT>"""
    system_content = """You will be presented lists of a JSON representation of a RESTFul API Path including the method.
                    You should tell me if to call this API path I need to provide any information specified by the JSON schema parameters or request body. Your response should be in JSON {"param": value, "type": value}
                    If nothing is required just put an empty JSON
                """
    messages = [
            {"role": "system", "content": system_content},
            {
                "role": "user",
                "content": question_placeholder.format(context=first_context)
            }
    ]
    answer = prompt_llama(messages=messages)
    if answer == "" or answer == "":
        return None

    print(answer['content'])
    messages.append({"role": "assistant", "content": answer['content']})
    print("=== END ===")
    # clear the terminal
    new_question =f"""Now by looking at the same list of endpoint paths but just their response bodies,
    pay attention to the response body of each path, and try and see if you can match any of the information given in the response body to a required parameter in the previous lists.
    Your response should be in the format
    {second_context}
    """
    
    messages.append({"role": "user", "content": new_question})
    answer = prompt_llama(messages=messages)
    if not answer or answer == "":
        return None
    llm_response = answer["content"]
    messages.append({"role": "assistant", "content": llm_response})

    question = f""""For each of the endpoints that have a match, write a synthetic question that a user might ask a chatbot that could be answered by that endpoint, and the question requires data that is retrieved from the matching endpoint.
        Please respond in the following format:
        INDEX: [line index] Question: [insert synthetic question]

        For example:
        INDEX: 0 Question: What is my user ID?
        Please provide one response per endpoint, following this format."
                """
    messages.append({"role": "user", "content": question})
    answer = prompt_llama(messages=messages)
    if not answer or answer == "":
        return None
    
    llm_response = answer["content"]
    print(llm_response)
    return llm_response


def single_stage_questioning(first_context):
    system_context = """You will be presented a list of JSON representations of a RESTFul API Path including the method.
                    For each path, you should generate a synthetic question a user might ask that would be answerable by the API endpoint.
                    If the path requires any parameters or request body, you should include that in your question as fake data.
                    Please respond in the following format:
                    INDEX: [line index] Question: [insert synthetic question]

                    For example:
                    INDEX: 0 Question: What is my user ID?
                    Please provide one response per endpoint, following this format.
                    """
    question_placeholder = """<CONTEXT>{context}</CONTEXT>"""
    messages = [
            {"role": "system", "content": system_context},
            {
                "role": "user",
                "content": question_placeholder.format(context=first_context)
            }
    ]
    answer = prompt_llama(messages=messages)
    if not answer or answer == "":
        return None
    llm_response = answer["content"]
    print(llm_response)
    return llm_response



input_file = "./data/0_output.json"
f = open(input_file, "r")
lines = f.readlines()
f.close()

output_file = '0_questions.jsonl'
index = get_index_to_start(output_file=output_file)
print(f"Starting from {index}")   

try:
    paths = []
    for line in lines[index:]:
        data = json.loads(line)
        full_schema_context = json.loads(data["full_schema_context"])
        path_context = json.loads(data["path_context"])
        path_parameters = {}
        response_parameters = defaultdict(list)
        request_params = defaultdict(list)
        index = 0
        first_context = ""
        second_context = ""
        for path, values in full_schema_context.items():
            method = values.get("method")

            resource = extract_resource_name(path)
            if not values.get("parameters") and not values.get("request_body"):
                continue
            full_line_item = f"{index}. [{{method: {values.get('method')}], path: {path}, summary: {values.get('summary')}, parameters: {values.get('parameters')}, request_body: {values.get('request_body') or []}, response_body: {values.get('response_body') or []}}}"
            short_line_item = f"{index}. | {{\method\": \"{values.get('method')}\", \"path\": \"{path}\", \"response_body\": {values.get('response_body') or []}}}\n"
            
            second_context += short_line_item
            first_context += full_line_item
            index += 1
            paths.append(path)
        
        paths_hash = create_path_hash(paths=paths)
        multi_part_llm_response = multi_stage_questioning(first_context, second_context)
        single_part_llm_response = single_stage_questioning(first_context)
        multi_stage_questions = format_questions_response(input_str=multi_part_llm_response, output_file=output_file, paths_hash=paths_hash)
        single_stage_questions = format_questions_response(input_str=single_part_llm_response, output_file=output_file, paths_hash=paths_hash)

        dump_data(output_file=output_file, paths_hash=paths_hash, multi_stage_questions=multi_stage_questions, single_stage_questions=single_stage_questions)
except Exception:
    dump_data(output_file=output_file, paths_hash=paths_hash, multi_stage_questions=[], single_stage_questions=[])