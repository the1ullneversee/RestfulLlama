import argparse
from collections import defaultdict
import os
import sys
import json
import time

from prompt_helpers import build_llama2_prompt, get_instructions
# from transformers import AutoTokenizer
# import transformers
# import torch

from dataclasses import dataclass

@dataclass
class Content:
    method: str
    summary: str
    parameters: list
    request_body: dict
    response_body: dict

class MethodInfo:
    """
    Class to represent a method in the API documentation
    Each method will have a summary, parameters, request body and response body
    """
    method: str
    summary: str
    parameters: list
    request_body: dict | None = None
    response_body: dict | None = None


class PathInfo:
    """
    Class to represent a path in the API documentation
    Each Method in the path will have a summary, parameters, request body and response body
    """

    path_name: str
    #methods: dict[str, Content] = {}

    def __init__(self):
        self.methods = defaultdict(MethodInfo)

class APIInfo:

    #paths: dict[str, PathInfo] = {}

    def __init__(self):
        self.paths = defaultdict(PathInfo, {})

    def __str__(self):
        return f"Paths: {self.paths}"


model = "meta-llama/Llama-2-7b-chat-hf"
# generate a entry to the script, use parse arguments to get a number between 1-3

def generate_dataset(folder_name, folder_index, model_id):
    # for each file in the folder, we open the filer and try read the contents
    # if the file is not a json file, we skip it
    tokenizer, pipeline = None, None
    if not os.path.exists(folder_name+'/processed/'):
        os.mkdir(folder_name+'/processed/')
    for filename in os.listdir(folder_name):
        if filename.endswith('.json'):
            print('Processing file: ' + filename)
            with open(os.path.join(folder_name, filename)) as f:
                try:
                    data = json.load(f)
                    contents = parse_file(data)
                    if contents:
                        # tokenizer, pipeline = setup_llm()
                        llm = setup_llm_llama_cpp(model_id)
                        generate_output(contents, llm, folder_index, model_id)
                except Exception as exc:
                    print('Error: Invalid JSON file' + exc)
            shutil.move(folder_name+"/"+filename, folder_name+'/processed/'+filename)

def navigate_json_ref(json_obj, ref):
    keys = ref.split('/')[1:]  # split the ref into keys, ignoring the first empty string
    result = json_obj
    for key in keys:
        if isinstance(result, dict):
            result = result.get(key, None)
        else:
            return None
    return result

def extract_definition(data, schema):
    one_of = schema.get('oneOf', None)
    items = schema.get('items', None)
    any_of = schema.get('anyOf', None)
    if one_of is not None:
        for item in one_of:
            content_ref = item.get('$ref', None)
            definition = navigate_json_ref(data, content_ref)
            if definition is None:
                print("Error: Invalid JSON file, missing definition")
    elif items is not None:
        if isinstance(items, list):
            for item in items:
                extract_definition(data, item)
        else:
            content_ref = items.get('$ref', None)
            items_any_of = items.get('anyOf', None)
            items_one_of = items.get('oneOf', None)
            singular_type = items.get('type', None)
            if items_any_of is not None:
                return extract_definition(data, items)
            if items_one_of is not None:
                return extract_definition(data, items)
            if singular_type is not None:
                return singular_type
            definition = navigate_json_ref(data, content_ref)
            if definition is None:
                print("Error: Invalid JSON file, missing definition")
    elif any_of is not None:
        for item in any_of:
            content_ref = item.get('$ref', None)
            base_type = item.get('type', None)
            if base_type is not None:
                definition = base_type
            else:
                definition = navigate_json_ref(data, content_ref)
                if definition is None:
                    print("Error: Invalid JSON file, missing definition")
    else:
        content_ref = schema.get('$ref', None)
        if content_ref is not None:
            definition = navigate_json_ref(data, content_ref)
            if definition is None:
                print("Error: Invalid JSON file, missing definition")
            def_properties = definition.get('properties', {})
            # data_properties = def_properties.get('data', None)
            # if data_properties is not None:
            #     return extract_definition(data, data_properties)
            if def_properties.get('type', None) == 'array':
                definition += extract_definition(data, definition.get('items', None))
        else:
            definition = schema.get('type', None)
            if definition is None:
                print("Error: Invalid JSON file, missing definition")

    return definition

def parse_file(data):
    # parse the file and return a list of the contents
    api_info = APIInfo()
    try:
        contents = []
        print(data.get('swagger', None))
        for path in data['paths']:
            api_info.paths[path] = PathInfo()
            for method in data['paths'][path]:
                parameters = []
                if 'parameters' not in data['paths'][path][method]:
                    continue
                for parameter in data['paths'][path][method]['parameters']:
                    type = parameter.get('type', None)
                    schema_type = parameter.get('schema', None)
                    if type is None and schema_type is not None:
                        s_type = schema_type.get('type', None)
                        s_ref_type = schema_type.get('$ref', None)
                        if s_type is not None:
                            type = s_type
                        elif s_ref_type is not None:
                            type = s_ref_type
                        else:
                            print("Error: Invalid JSON file, missing type")
                    parameters.append("Name: {} Description: {} Type: {}".format(parameter['name'], parameter.get('description', ""), type if type is not None else "Unknown"))
                contents.append({"path": path, "method": method, "summary": data['paths'][path][method].get('summary', ""), "parameters": parameters})
        contents.append({"Type Definitions" : f" {data['definitions']}"})
        return contents
    except Exception as exc:
        print(f"Error: Invalid JSON file, missing key {exc}")
        return None

def generate_output(contents, llm, folder_index, model_id):
    # generate the output file
    # for each path in the contents, we generate a prompt
    # we then pass the prompt through the LLM
    # for each method of the prompt, we generate a response
    outputs = []
    outputs.extend(generate_per_path_qanda(contents, llm, model_id))
    #outputs.extend(generate_entity_qanda(contents, tokenizer, pipeline))
    dump_to_file(outputs, './', folder_index)

    # write the outputs to a file

def dump_to_file(outputs, main_path, folder_index):
    folder = main_path
    file_name = f'{folder_index}_raw_rest_fine_tune.jsonl'
    if not os.path.exists(folder):
        os.mkdir(folder)
    if not os.path.exists(folder+file_name):
        with open(folder+file_name, "w") as f:
            f.write("")
    with open(folder+file_name, "a") as f:
        for pair in outputs:
            f.write(json.dumps({"question": pair["question"], "answer": pair["answer"]})+"\n")

def generate_per_path_qanda(contents, llm, model_id):
    # for each path in the contents, we generate a prompt
    # we then pass the prompt through the LLM
    # for each method of the prompt, we generate a response
    system_content = '''
            You are about to be shown detail of a Swagger Representation of a REST Endpoint Path, including the path, method, summary and parameters.
            Your answers are to be give in text for all answers except the python code answer which must be in python code.
            Give no explanation, and only give the answer in the following format, no fluff text:
            url: <url> headers: <headers> params: <params> body: <body> python code: <code> code needs to be in Python code using the requests library. Do not include imports, just code.
            '''
    outputs = []
    for line in contents:
        if 'Type Definitions' in line:
            continue
        endpoint = f"Path: {line['path']} Method: {line['method']} Summary: {line['summary']} Parameters: {line['parameters']}"
        question = f"What is the url, headers, params, body and python code to call the following REST endpoint? {endpoint}"
        # instructions = get_instructions(question, system_content)
        # prompt = build_llama2_prompt(instructions)
        answer, raw_output, t = inference(llm=llm, system_content=system_content, question=question)
        tok_s = raw_output["usage"]["completion_tokens"] / t
        out = {"question": question, 
                            "answer": answer, 
                            "model_name":model_id, 
                            "tokens_sec": tok_s,
                            "model_file":raw_output["model"]
                            }

        outputs.append(out)
    return outputs

def generate_whole_api_qanda(contents, tokenizer, pipeline):
    # for each path in the contents, we generate a prompt
    # we then pass the prompt through the LLM
    # for each method of the prompt, we generate a response
    pass

def generate_entity_qanda(contents, llm, model_id):
    # for each path in the contents, we generate a prompt
    # we then pass the prompt through the LLM
    # for each method of the prompt, we generate a response
    system_content = f'''
        The input will be a list of strings representing resource paths for a RESTFul API.
        Give your response based on which paths require an entity id, and if you can get that entity elsewhere, tell the user how.
        Write the operations in natural language, and separate them with a comma, no fluff text.
    '''
    outputs = []
    question = "Give the following RESTful Endpoint paths. What Paths require an entity_id to access and can you get that entity from a different path? If you can which path do you need to call? \n" + json.dumps(contents) + "\n"
    # instructions = get_instructions(question, system_content)
    # prompt = build_llama2_prompt(instructions)
    answer, raw_output, t = inference(llm=llm, system_content=system_content, question=question)
    tok_s = raw_output["usage"]["completion_tokens"] / t
    out = {"question": question, 
                        "answer": answer, 
                        "model_name":model_id, 
                        "tokens_sec": tok_s,
                        "model_file":raw_output["model"]
                        }

    outputs.append(out)
    return outputs

def inference(llm, system_content, question):
    t0 = time.perf_counter()
    response = llm.create_chat_completion(
            messages = [
                {"role": "system", "content": f"{system_content}"},
                {
                    "role": "user",
                    "content": question
                }
            ]
        )
    content = response["choices"][0]['message']['content']
    # extract everything from [CODE] to [/CODE]
    code = content[content.find('[CODE]')+len('[CODE]'):content.find('[/CODE]')]
    return code, response, time.perf_counter() - t0

def pass_through_llm(llm, system_content, question):
    # pass the file through the LLM
    response = llm.create_chat_completion(
            messages = [
                {"role": "system", "content": f"{system_content}"},
                {
                    "role": "user",
                    "content": question
                }
            ]
        )
    content = response["choices"][0]['message']['content']
    # extract everything from [CODE] to [/CODE]
    code = content[content.find('[CODE]')+len('[CODE]'):content.find('[/CODE]')]
    return code
    
def setup_llm():
    tokenizer = AutoTokenizer.from_pretrained(model, token="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", load_in_4bit=True)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token="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",
    )
    return tokenizer, pipeline

def reduce_json_info(json_response):
    if not json_response:
        return []
    if not isinstance(json_response, dict):
        return [json_response]
    properties = json_response.get('properties', {})
    reduced_info = []
    for prop, details in properties.items():
        title = details.get('title', '')
        if title.lower() not in json_response.get('required', []):
            continue
        type_info = details.get('type', '')
        if not type_info:
            type_info = [item.get('type', '') for item in details.get('anyOf', [])]
            type_info = [t for t in type_info if t]  # remove empty strings
        reduced_info.append({'title': title, 'type': type_info})
    return reduced_info

def _generate_path_to_response(api_info: APIInfo):
    """
    Generates a textual output for each path and what its response is.
    """
    path_context = ""
    # with open(file_name + ".txt", 'w') as f:
    for path in api_info.paths:
        for method in api_info.paths[path].methods:
            response = {}
            if api_info.paths[path].methods[method].response_body:
                response = api_info.paths[path].methods[method].response_body
            reduce_response = reduce_json_info(response)
            #print(f"Path: {path} Method: {method} Response: {reduce_response}")
            path_context += f"<Path>: {path} <Method>: {method} <Summary>: {api_info.paths[path].methods[method].summary} <Parameters> {api_info.paths[path].methods[method].parameters} <Response>: {reduce_response}\n"
    return path_context

def _generate_path_to_query(api_info: APIInfo, file_name: str):
    """ Generates a textual output for each path and what its query parameters are. """
    schema_context = ""
    # with open(file_name + ".txt", 'w') as f:
    for path in api_info.paths:
        for method in api_info.paths[path].methods:
            parameters = api_info.paths[path].methods[method].parameters
            request_body = {}
            if api_info.paths[path].methods[method].request_body:
                request_body = api_info.paths[path].methods[method].request_body
            #print(f"Path: {path} Method: {method} Parameters: {parameters}")
            schema_context = f"<Path>: {path} <Method>: {method} <Summary>: {api_info.paths[path].methods[method].summary}"
    return schema_context

def _generate_full_path_context(api_info: APIInfo):
    """ Generates a textual output for each path and what its query parameters are. """
    schema_context = ""
    for path in api_info.paths:
        for method in api_info.paths[path].methods:
            parameters = api_info.paths[path].methods[method].parameters
            request_body = {}
            if api_info.paths[path].methods[method].request_body:
                request_body = api_info.paths[path].methods[method].request_body
            schema_context += f"<Path>: {path} <Method>: {method} <Summary>: {api_info.paths[path].methods[method].summary} <Parameters>: {parameters} <Request Body>: {request_body}\n"
    return schema_context

def _dump_to_file(output_file: str, key: str, values: dict):
    with open(output_file, 'a') as f:
        # write as a json object
        f.write(json.dumps({key: values}) + '\n')

def _move_file(folder: str, file: str, destination: str):
    os.rename(folder + file, f"{destination}/{file}")

if __name__ == '__main__':
    # open directory and read the files
    parser = argparse.ArgumentParser(description='Run dataset generation on a selected folder')
    parser.add_argument('--folder_index', type=str, help='The folder number to process')
    args = parser.parse_args()
    input_folder = f'./input_{args.folder_index}/'
    input_folder = "./old_parsed/"
    files = os.listdir(input_folder)
    for file in files:
        f = open(input_folder+file)
        try:
            data = json.load(f)
            api_information = parse_file(data)
            if not api_information:
                f.close()
                _move_file(folder=input_folder, file=file, destination="./broken_parsed")
                continue
            path_context = _generate_path_to_response(api_information)
            full_schema_context = _generate_full_path_context(api_information)
            key = file.split(".")[0]
            values = {"path_context": path_context, "full_schema_context": full_schema_context}
            _dump_to_file("output.json", key, values)
            f.close()
            _move_file(folder=input_folder, file=file, destination="./parsed")
        except Exception as e:
            print('Error: Invalid JSON file')
            f.close()
            _move_file(folder=input_folder, file=file, destination="./broken_parsed")
    


