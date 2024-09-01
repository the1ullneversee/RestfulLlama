import asyncio
import hashlib
import json
import os
import re

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

from schema_processor import execute_context_call

# region CONSTANTS
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
3. Generate a list of answers to the questions from Stage 2, as if they were answered by the user.
4. Amend the Python code to include the values provided in the follow-up response.
Each question should be formatted as follows:
```
Question x: [Your question here]

Your response should be as follows:
Simulated Questions:
Q1
Q2
...
Simulated answers:
A1
A2
...
Amended code:
....
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

# endregion


def dump_empty(output_file, paths_hash):
    with open(output_file, "a") as file:
        file.write(json.dumps({"hash": paths_hash, "questions": []}) + "\n")


def dump_data(output_file, paths_hash, multi_stage_questions, single_stage_questions, simulated_conversations):
    with open(output_file, "a") as file:
        for convo in simulated_conversations:
            file.write(
                json.dumps(
                    {
                    "messages": convo
                    } 
                )+ "\n"
            )


def format_questions_response(input_str, output_file, paths_hash):
    # Split the input string into lines
    questions = []
    try:
        lines = input_str.split("\n")
        
        for line in lines:
            # Check if the line is a question (does not contain "No match")
            if not line:
                continue
            if "Question" not in line:
                continue
            questions.append(line)
        return questions
    except Exception as exc:
        print(exc)


def extract_resource_name(path):
    components = path.split("/")
    # Iterate in reverse to find the first non-parameter component
    for component in reversed(components):
        if not (component.startswith("{") and component.endswith("}")):
            return component  # This is the resource name
    return None  # In case no non-parameter component is found


def prompt_llama(messages: list, llm):
    # Generate a response from the model
    answer = ""
    try:
        answer = llm.create_chat_completion(
            messages=messages,
            temperature=0.0,
            top_p=1.0,
            repeat_penalty=1.0,
            top_k=1,
        )
    except Exception as exc:
        print(exc)
        return answer
    return answer["choices"][0]["message"]


def get_questions(output_file):
    questions = []
    index = 0
    with open(output_file, "r") as file:
        lines = file.readlines()
        questions = [json.loads(line) for line in lines]
    for question in questions:
        if "conversation" not in question:
            break
        index += 1
    return questions, index


def create_path_hash(paths: list):
    concat = "".join(paths)

    hash_object = hashlib.sha256(concat.encode())
    hash_hex = hash_object.hexdigest()
    return hash_hex


def multi_stage_questioning(first_context, second_context, llm):
    question_placeholder = """<CONTEXT>{context}</CONTEXT>"""
    system_content = """You will be presented lists of a JSON representation of a RESTFul API Path including the method.
                    You should tell me if to call this API path I need to provide any information specified by the JSON schema parameters or request body. Your response should be in JSON {"param": value, "type": value}
                    If nothing is required just put an empty JSON
                """
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": question_placeholder.format(context=first_context)},
    ]
    answer = prompt_llama(messages=messages, llm=llm)
    if answer == "" or answer == "":
        return None

    print(answer["content"])
    messages.append({"role": "assistant", "content": answer["content"]})
    print("=== END ===")
    # clear the terminal
    new_question = f"""Now by looking at the same list of endpoint paths but just their response bodies,
    pay attention to the response body of each path, and try and see if you can match any of the information given in the response body to a required parameter in the previous lists.
    Your response should be in the format
    {second_context}
    """

    messages.append({"role": "user", "content": new_question})
    answer = prompt_llama(messages=messages, llm=llm)
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
    answer = prompt_llama(messages=messages, llm=llm)
    if not answer or answer == "":
        return None

    llm_response = answer["content"]
    print(llm_response)
    return llm_response


def single_stage_questioning(first_context, llm):
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
        {"role": "user", "content": question_placeholder.format(context=first_context)},
    ]
    answer = prompt_llama(messages=messages, llm=llm)
    if not answer or answer == "":
        return None
    llm_response = answer["content"]
    print(llm_response)
    return llm_response


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
    llm_response = llm_response.replace('\n', ' ')
    pattern = r'"{(.*?)}"'

    # Search for the pattern in the input string
    matches = re.findall(pattern, llm_response)
    questions = []
    # If matches are found, extract the placeholders
    if matches:
        for i, match in enumerate(matches, start=1):
            question = f'Question {i}: What is the actual value that should be used in place of "{match}"?'
            questions.append(question)
    else:
        print("No placeholders found in the code.")
    return questions


def extract_simulated_answers(llm_response):
    """
    Extracts the simulated answers from the LLM response.
    """
    # Regex pattern to match the simulated answers
    # first strip out line breaks and encodings
    pattern = r"Answer \d+: (.*)"
    other_pattern = r"A\d+: (.*)"

    # Search for the pattern in the input string
    matches = re.findall(pattern, llm_response)
    other_matches = re.findall(pattern=other_pattern, string=llm_response)
    answers = []
    # If matches are found, extract the simulated answers
    if matches:
        for match in matches:
            answers.append(match)
    elif other_matches:
        for match in other_matches:
            answers.append(match)
    else:
        print("No simulated answers found.")
        return llm_response
    return answers


def extract_amended_code(llm_response):
    """
    Extracts the amended code from the LLM response.
    """
    # Regex pattern to match the amended code
    # find the first occurrence of "Amended code:"
    try:
        pattern = r"Amended code:(.*)"
        index = llm_response.find("Amended code:")
        amended_code = llm_response[index + len("Amended code:") :]
        return amended_code
    except:
        return None


def extract_simulated_data_parts(response):
    start = "Simulated Questions:"
    end = "Simulated answers:"
    str_index = response.find(start)
    end_index = response.find(end)

    questions = response[str_index + len(start) : end_index]
    response = response[end_index:]

    start = "Simulated answers:"
    end = "Amended code:"
    str_index = response.find(start)
    end_index = response.find(end)

    answers = response[str_index + len(start) : end_index]
    response = response[end_index:]

    return questions, answers, response


def simulate_code_reasoning(answer: str, llm):
    return_messages = []
    messages = [
        {"role": "system", "content": CODE_CONTEXT},
        {"role": "user", "content": answer},
    ]
    answer = prompt_llama(messages=messages, llm=llm)
    questions = extract_code_placeholder_questions(answer.get('content'))
    print(questions)
    if not questions:
        return return_messages
    messages.append({"role": "user", "content": str(questions)})
    return_messages.append({"role": "user", "content": str(questions)})
    code_question_answer_response = prompt_llama(messages=messages, llm=llm)
    if code_question_answer_response == "":
        return []
    code_question_answer_response = code_question_answer_response["content"]

    simulated_answers = ""
    code_question_answer_response = code_question_answer_response.replace(
        "\n", " "
    ).replace("`", " ")

    _, simulated_answers, response = extract_simulated_data_parts(
        code_question_answer_response
    )
    simulated_answers = extract_simulated_answers(simulated_answers)

    messages.append({"role": "assistant", "content": str(simulated_answers)})
    return_messages.append({"role": "assistant", "content": str(simulated_answers)})

    amended_code = extract_amended_code(response)
    if not amended_code:
        return []
    messages.append({"role": "assistant", "content": amended_code})
    return_messages.append({"role": "assistant", "content": amended_code})

    return return_messages


def conversation_simulation(
    llm,
    multi_stage_questions,
    single_stage_questions,
    short_path_context: str,
    full_path_context: dict,
):

    system_context = f"""You are a helpful assistant that can generate Python code to interact with an application's API. You have access to the application's API endpoints and their corresponding schemas.
        When a user asks a question, your task is to ask for additional context about the API endpoints as needed, reason about the schema, and generate Python code to achieve the user's goal.
        You can ask for context about an API endpoint by saying get_context(path='path', method='method'), where path is the endpoint path, and method is the HTTP method (e.g., 'GET', 'POST').
        The application's API endpoints are: {short_path_context}
        Please respond with a get_context request to clarify the API endpoint needed to answer the user's question."""
    system_context = system_context.replace("\\n", "").replace("\\'", "")
    simulated_questions = []
    simulated_questions.extend(
        generate_multi_stage_conversations(
            multi_stage_questions=multi_stage_questions,
            full_path_context=full_path_context,
            system_context=system_context,
            llm=llm,
        )
    )
    simulated_questions.extend(
        generate_single_stage_conversation(
            single_stage_questions=single_stage_questions,
            full_path_context=full_path_context,
            system_context=system_context,
            llm=llm,
        )
    )

    return simulated_questions


def generate_multi_stage_conversations(
    multi_stage_questions, full_path_context, system_context, llm
):
    try:
        multi_question_responses = []
        for question in multi_stage_questions or []:
            if not question:
                continue
            user_question = question.split("Question: ")[1]
            full_messages = []
            messages = [
                {"role": "system", "content": system_context},
                {"role": "user", "content": user_question},
            ]
            answer = prompt_llama(messages=messages, llm=llm)
            if not answer:
                print("Decide")
                continue
            answer = answer["content"]
            messages.append({"role": "assistant", "content": answer})
            schema = execute_context_call(
                full_schema_context=full_path_context, context_call=answer
            )
            if not schema:
                continue
            messages.append({"role": "assistant", "content": schema})
            answer = prompt_llama(messages=messages, llm=llm)
            if answer == "":
                continue
            answer = answer["content"]
            full_messages.append(messages)

            messages.extend(simulate_code_reasoning(answer=answer, llm=llm))
            multi_question_responses.extend(full_messages)
        return multi_question_responses
    except Exception as exc:
        print(exc)


def generate_single_stage_conversation(
    single_stage_questions, full_path_context, system_context, llm
):
    try:
        single_stage_responses = []
        for question in single_stage_questions or []:
            user_question = question.split("Question: ")[1]
            full_messages = []
            messages = [
                {"role": "system", "content": system_context},
                {"role": "user", "content": user_question},
            ]
            answer = prompt_llama(messages=messages, llm=llm)
            if not answer:
                continue
            answer = answer["content"]
            messages.append({"role": "assistant", "content": answer})
            schema = execute_context_call(
                full_schema_context=full_path_context, context_call=answer
            )
            if schema == "No match found.":
                continue
            messages.append({"role": "assistant", "content": schema})
            answer = prompt_llama(messages=messages, llm=llm)
            if answer == "":
                continue
            answer = answer["content"]
            full_messages.append(messages)
            single_stage_responses.extend(full_messages)
        return single_stage_responses
    except Exception as exc:
        print(exc)


def download_gguf_model(
    local_dir: str, repo_name: str, model_name: str, file_name: str
):
    try:
        # Create the local directory if it doesn't exist
        os.makedirs(local_dir, exist_ok=True)
        # Download the file
        model_path = hf_hub_download(
            repo_id=repo_name + "/" + model_name,
            filename=file_name,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )
        print(f"Model downloaded successfully to: {model_path}")
        return model_path

    except Exception as e:
        print(f"An error occurred while downloading the model: {e}")
        return None


async def load_language_model(
    local_dir: str, repo_name: str, model_name: str, file_name: str
) -> None:
    def _load_model():
        model_path = download_gguf_model(
            local_dir=local_dir,
            repo_name=repo_name,
            model_name=model_name,
            file_name=file_name,
        )
        return Llama(
            model_path=model_path,
            # model_path="./models/restful_llama_8_Q8.gguf",
            n_threads=8,  # CPU cores
            n_batch=512,  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
            n_gpu_layers=-1,  # Change this value based on your model and your GPU VRAM pool.
            n_ctx=8096,  # Context window
            verbose=False,
        )

    # Run the CPU-bound model loading in a thread pool
    llm = await asyncio.to_thread(_load_model)
    return llm
