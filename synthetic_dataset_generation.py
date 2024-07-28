import hashlib
import json
from collections import defaultdict

# from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained('NousResearch/Hermes-2-Theta-Llama-3-70B', trust_remote_code=True)

# model_path = "Meta-Llama-3-70B-Instruct.Q4_K_M.gguf"
# model_path = "models/Hermes-2-Pro-Llama-3-8B-Q8_0.gguf"
# import llama_cpp

llm = None
# llm = llama_cpp.Llama(
#     model_path=model_path,
#     n_threads=32,  # CPU cores
#     n_batch=512,  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
#     n_gpu_layers=24,  # Change this value based on your model and your GPU VRAM pool.
#     n_ctx=8000,  # Context window
#     verbose=True,
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

def prompt_llama(messages: list):
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

def get_questions(output_file):
    questions = []
    index = 0
    with open(output_file, 'r') as file:
        lines = file.readlines()
        questions = [json.loads(line) for line in lines]
    for question in questions:
        if 'conversation' not in question:
            break
        index += 1
    return questions, index


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

import re

def execute_context_call(full_schema_context, context_call):
    """Executes the context call and returns the schema for the requested endpoint."""
    # Extract the endpoint path from the context call
    pattern = r"get_context\('(.*?)'\)"
    match = re.search(pattern, context_call)
    if not match:
        return "No match found."
    endpoint_path = match.group(1)

    # Get the schema for the requested endpoint
    schema = full_schema_context.get(endpoint_path)
    if not schema:
        return "No schema found for the endpoint."
    return schema

def extract_context(llm_response):
    """Extracts the context from the LLM response."""
    # Regex pattern to match the function call
    pattern = r"get_context\('(.*?)'\)"

    # Search for the pattern in the input string
    match = re.search(pattern, llm_response)
    function_call = None
    # If a match is found, extract the function call
    if match:
        function_call = match.group(0)  # match.group(0) returns the entire match
        print(function_call)
    else:
        print("No function call found.")
    return function_call

def extract_code_placeholder_questions(llm_response):
    """
    After reviewing the provided Python code, I have identified two placeholders that need to be replaced with actual values:
    Question 1: What is the actual job ID that should be used in place of "{your_job_id_here}"?
    Question 2: What is the actual API base URL that should be used in place of "https://your-api-base-url.com"?
    These questions need to be answered by the end-user to replace the placeholders and make the code functional.
    """
    # Regex pattern to match the placeholders in the code
    pattern = r'"{(.*?)}"'

    # Search for the pattern in the input string
    matches = re.findall(pattern, llm_response)
    questions = []
    # If matches are found, extract the placeholders
    if matches:
        for i, match in enumerate(matches, start=1):
            question = f"Question {i}: What is the actual value that should be used in place of \"{match}\"?"
            questions.append(question)
    else:
        print("No placeholders found in the code.")
    return questions


CODE_RESPONSE = """
import requests
jobid = "{your_job_id_here}"  # replace with your actual job ID
url = f"https://your-api-base-url.com/v1.0/provisioning/ondemandstatus/{jobid}"
response = requests.get(url)
if response.status_code == 200:
    data = response.json()
    print(f"Provisioning job status: {data['body']}")
    print(f"Status code: {data['statusCode']}")
    print(f"Status code value: {data['statusCodeValue']}")
else:
    print(f"Error: {response.status_code}")
"""

CODE_CONTEXT = """You are a language model reviewing a piece of Python code provided by the user. Your task is to:
1. Examine the code for any placeholders or missing information that may be required for the code to function correctly.
2. Generate a list of questions in a formatted way, so that these questions can be easily extracted and asked to the end-user to gather the necessary information needed to replace the placeholders.
3. Expect a follow-up response from the user answering those questions.
4. Amend the Python code to include the values provided in the follow-up response.
Each question should be formatted as follows:
```
Question x: [Your question here]
```"""

CODE_QUESTION_ANSWER_RESPONSE = """
Questions:

Question 1: What is your actual job ID that should be used in the API request?
Question 2: What is your API base URL that should be used in the API request?
Simulated answers:
Answer 1: 1234567890
Answer 2: https://api.example.com
Amended code:

import requests
jobid = "1234567890"  # replaced with the actual job ID
url = f"https://api.example.com/v1.0/provisioning/ondemandstatus/{jobid}"
response = requests.get(url)
if response.status_code == 200:
    data = response.json()
    print(f"Provisioning job status: {data['body']}")
    print(f"Status code: {data['statusCode']}")
    print(f"Status code value: {data['statusCodeValue']}")
else:
    print(f"Error: {response.status_code}")
"""

def extract_simulated_answers(llm_response):
    """
    Extracts the simulated answers from the LLM response.
    """
    # Regex pattern to match the simulated answers
    pattern = r"Answer \d+: (.*)"

    # Search for the pattern in the input string
    matches = re.findall(pattern, llm_response)
    answers = []
    # If matches are found, extract the simulated answers
    if matches:
        for match in matches:
            answers.append(match)
    else:
        print("No simulated answers found.")
    return answers

def extract_amended_code(llm_response):
    """
    Extracts the amended code from the LLM response.
    """
    # Regex pattern to match the amended code
    # find the first occurrence of "Amended code:"
    pattern = r"Amended code:(.*)"
    index = llm_response.find("Amended code:")
    amended_code = llm_response[index+len("Amended code:"):]
    return amended_code

def simulate_code_reasoning(answer: str):
    return_messages = []
    messages = [
            {"role": "system", "content": CODE_CONTEXT},
            {"role": "user", "content": answer}
        ]
    questions = extract_code_placeholder_questions(answer)
    print(questions)
    messages.append({"role": "user", "content": str(questions)})
    return_messages.append({"role": "user", "content": str(questions)})

    simulated_answers = extract_simulated_answers(CODE_QUESTION_ANSWER_RESPONSE)
    messages.append({"role": "assistant", "content": str(simulated_answers)})
    return_messages.append({"role": "assistant", "content": str(simulated_answers)})

    amended_code = extract_amended_code(CODE_QUESTION_ANSWER_RESPONSE)
    messages.append({"role": "assistant", "content": amended_code})

    return_messages.append({"role": "assistant", "content": amended_code})
    return return_messages

def conversation_simulation(llm, multi_stage_questions, single_stage_questions, short_path_context: str, full_path_context: dict):

    system_context = f"""You are a helpful assistant that can generate Python code to interact with an application's API. You have access to the application's API endpoints and their corresponding schemas.
        When a user asks a question, your task is to ask for additional context about the API endpoints as needed, reason about the schema, and generate Python code to achieve the user's goal.
        You can ask for context about an API endpoint by saying get_context('path'), where path is the endpoint path.
        The application's API endpoints are: {short_path_context}
        Please respond with a get_context request to clarify the API endpoint needed to answer the user's question."""
    
    # system_context = f"""You are a helpful assistant that can generate Python code to interact with an application's API. You have access to the application's API endpoints and their corresponding schemas.
    # When a user asks a question, your task is to ask for additional context about the API endpoints as needed, reason about the schema, and generate Python code to achieve the user's goal.
    # You can ask for context about an API endpoint by saying get_context('path'), where path is the endpoint path. You will receive a response with the schema and any relevant information about the endpoint.
    # Your goal is to generate Python code that is correct, concise, and easy to understand. You may ask follow-up questions to clarify the user's request or request additional context about the API endpoints.
    # The application's API endpoints are: {short_path_context}.
    # Go ahead and respond to the user's question."""
    system_context = system_context.replace('\\n', '').replace('\\\'', '')
    simulated_questions = []
    simulated_questions.append(generate_multi_stage_questions(multi_stage_questions, full_path_context, system_context, simulated_questions))
    return simulated_questions

def generate_multi_stage_questions(multi_stage_questions, full_path_context, system_context, simulated_questions):
    for question in multi_stage_questions:
        user_question = question.split("Question: ")[1]
        full_messages = []
        messages = [
            {"role": "system", "content": system_context},
            {"role": "user", "content": user_question}
        ]
        answer = """To answer your question, I need more information about the API endpoint that retrieves the status of a provisioning job.
                    Let me get some context about the /v1.0/provisioning/ondemandstatus/{jobid} endpoint.
                    get_context('/v1.0/provisioning/ondemandstatus/{jobid}')"""
        context_call = extract_context(answer)
        if not context_call:
            continue
        messages.append({"role": "context_layer", "content": context_call})
        schema = execute_context_call(full_schema_context=full_path_context, context_call=context_call)
        if schema == "No match found.":
            continue
        messages.append({"role": "assistant", "content": schema})
        #answer = prompt_llama(messages=messages)
        answer = CODE_RESPONSE
        full_messages.append(messages)

        messages.extend(simulate_code_reasoning(CODE_RESPONSE))
        simulated_questions.append(full_messages)


input_file = "./data/0_output.json"
f = open(input_file, "r")
lines = f.readlines()
f.close()

output_file = '0_questions.jsonl'
previous_questions, start_index = get_questions(output_file=output_file) 

line_index = start_index
try:
    paths = []
    multi_stage_questions = []
    single_stage_questions = []
    for line in lines[start_index:]:
        data = json.loads(line)
        full_schema_context = json.loads(data["full_schema_context"])
        path_context = json.loads(data["path_context"])
        path_parameters = {}
        response_parameters = defaultdict(list)
        request_params = defaultdict(list)
        index = 0
        first_context = ""
        second_context = ""
        third_context = ""
        for path, values in full_schema_context.items():
            method = values.get("method")

            resource = extract_resource_name(path)
            if not values.get("parameters") and not values.get("request_body"):
                continue
            full_line_item = f"{index}. [{{method: {values.get('method')}], path: {path}, summary: {values.get('summary')}, parameters: {values.get('parameters')}, request_body: {values.get('request_body') or []}, response_body: {values.get('response_body') or []}}}"
            short_line_item = f"{index}. | {{\method\": \"{values.get('method')}\", \"path\": \"{path}\", \"response_body\": {values.get('response_body') or []}}}\n"
            tiny_line_item = f"{index}. | {{\"path\": \"{path}\"}}\n"
            
            second_context += short_line_item
            first_context += full_line_item
            third_context += tiny_line_item
            index += 1
            paths.append(path)
        
        paths_hash = create_path_hash(paths=paths)

        if line_index > len(previous_questions):
            multi_part_llm_response = multi_stage_questioning(first_context, second_context)
            single_part_llm_response = single_stage_questioning(first_context)
            multi_stage_questions = format_questions_response(input_str=multi_part_llm_response, output_file=output_file, paths_hash=paths_hash)
            single_stage_questions = format_questions_response(input_str=single_part_llm_response, output_file=output_file, paths_hash=paths_hash)
        else:
            multi_stage_questions = previous_questions[line_index].get("questions")
            single_stage_questions = previous_questions[line_index].get("single_stage_questions", [])
        conversation_simulation(llm=llm, multi_stage_questions=multi_stage_questions, single_stage_questions=single_stage_questions, short_path_context=third_context, full_path_context=full_schema_context)

        dump_data(output_file=output_file, paths_hash=paths_hash, multi_stage_questions=multi_stage_questions, single_stage_questions=single_stage_questions)
        line_index += 1
except Exception:
    dump_data(output_file=output_file, paths_hash=paths_hash, multi_stage_questions=[], single_stage_questions=[])
    line_index += 1
