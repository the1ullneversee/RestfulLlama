import argparse
import os
import sys
import json

from prompt_helpers import build_llama2_prompt, get_instructions
# from transformers import AutoTokenizer
# import transformers
# import torch

model = "meta-llama/Llama-2-7b-chat-hf"
# generate a entry to the script, use parse arguments to get a number between 1-3

def generate_dataset(folder_name):
    # for each file in the folder, we open the filer and try read the contents
    # if the file is not a json file, we skip it
    tokenizer, pipeline = None, None
    for filename in os.listdir(folder_name):
        if filename.endswith('.json'):
            print('Processing file: ' + filename)
            with open(os.path.join(folder_name, filename)) as f:
                try:
                    data = json.load(f)
                    contents = parse_file(data)
                    if contents is None:
                        continue
                    generate_output(contents, tokenizer, pipeline)
                except json.decoder.JSONDecodeError:
                    print('Error: Invalid JSON file')
                    continue

def parse_file(data):
    # parse the file and return a list of the contents
    try:
        contents = []
        for path in data['paths']:
            for method in data['paths'][path]:
                parameters = []
                for parameter in data['paths'][path][method]['parameters']:
                    type = parameter.get('type', None)
                    schema_type = parameter.get('schema', None)
                    if type is None and schema_type is not None:
                        type = schema_type.get('$ref', None)
                        if type is None:
                            print("Error: Invalid JSON file, missing type")
                    parameters.append("Name: {} Description: {} Type: {}".format(parameter['name'], parameter['description'], type if type is not None else "Unknown"))
                contents.append({"path": path, "method": method, "summary": data['paths'][path][method]['summary'], "parameters": parameters})
        contents.append({"Type Definitions" : f" {data['definitions']}"})
        return contents
    except KeyError as e:
        print(f"Error: Invalid JSON file, missing key {e}")
        return None

def generate_output(contents, tokenizer, pipeline):
    # generate the output file
    # for each path in the contents, we generate a prompt
    # we then pass the prompt through the LLM
    # for each method of the prompt, we generate a response
    outputs = {}
    outputs += generate_per_path_qanda(contents, tokenizer, pipeline)
    outputs += generate_whole_api_qanda(contents, tokenizer, pipeline)
    outputs += generate_entity_qanda(contents, tokenizer, pipeline)

    # write the outputs to a file

def dump_to_file(outputs, main_path):
    folder = main_path
    if not os.path.exists(folder):
        os.mkdir(folder)
    if not os.path.exists(folder+"raw_rest_fine_tune.jsonl"):
        with open(folder+"raw_rest_fine_tune.jsonl", "w") as f:
            f.write("")
    with open(folder+"raw_rest_fine_tune.jsonl", "a") as f:
        for question, answer in outputs:
            f.write(json.dumps({"question": question, "answer": answer})+"\n")



def generate_per_path_qanda(contents, tokenizer, pipeline):
    # for each path in the contents, we generate a prompt
    # we then pass the prompt through the LLM
    # for each method of the prompt, we generate a response
    system_content = '''
            You are about to be shown detail of a Swagger Representation of a REST Endpoint Path, including the path, method, summary and parameters.
            Your answers are to be given in code or text.
            Give no explanation, and only give the answer in the following format, no fluff text:
            url: <url> headers: <headers> params: <params> body: <body> python code: <code> code needs to be in Python code using the requests library. Do not include imports, just code.
            '''
    outputs = []
    for line in contents:
        endpoint = f"Path: {line['path']} Method: {line['method']} Summary: {line['summary']} Parameters: {line['parameters']}"
        question = f"What is the url, headers, params, body and python code to call the following REST endpoint? {endpoint}"
        instructions = get_instructions(question, system_content)
        prompt = build_llama2_prompt(instructions)
        output = pass_through_llm(tokenizer, pipeline, prompt)
        output = f"Sure thing, here is the answer:"
        outputs.append({"question": question, "answer": output})
    dump_to_file(outputs, './')
    pass

def generate_whole_api_qanda(contents, tokenizer, pipeline):
    # for each path in the contents, we generate a prompt
    # we then pass the prompt through the LLM
    # for each method of the prompt, we generate a response
    pass

def generate_entity_qanda(contents, tokenizer, pipeline):
    # for each path in the contents, we generate a prompt
    # we then pass the prompt through the LLM
    # for each method of the prompt, we generate a response
    pass

def pass_through_llm(tokenizer, pipeline, prompt):
    # pass the file through the LLM
    sequences = pipeline(
        prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=200,
    )
    output = ""
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")
        output += seq['generated_text']
    return output
    

def setup_llm():
    tokenizer = AutoTokenizer.from_pretrained(model)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    return tokenizer, pipeline


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run dataset generation on a selected folder')
    parser.add_argument('--folder_index', type=str, help='The folder number to process')
    args = parser.parse_args()

    folder_index = int(args.folder_index)
    if folder_index < 1 or folder_index > 3:
        print('Invalid folder index, must be between 1 and 3')
        sys.exit(1)
    
    folder_name = 'input_' + str(folder_index)
    generate_dataset(folder_name)
    print('Processing folder: ' + folder_name)
