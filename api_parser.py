import argparse
from collections import defaultdict
import os
import sys
import json
import time

from prompt_helpers import build_llama2_prompt, get_instructions
from dataclasses import dataclass

import re

def to_camel_case(s):
    if len(s) == 0:
        return s
    s = re.sub(r'(_|-)+', ' ', s).title().replace(' ', '')
    return s[0].lower() + s[1:]

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

def navigate_json_ref(json_obj, ref):
    if ref is None:
        return None
    keys = ref.split('/')[1:]  # split the ref into keys, ignoring the first empty string
    result = json_obj
    for key in keys:
        if isinstance(result, dict):
            result = result.get(key, None)
        else:
            return None
    return result

def _extract_any_of(any_of: list, data: dict):
    # type can be any_of the types in the list
    # so the definition is a list of types
    definition = []
    for item in any_of:
        content_ref = item.get('$ref', None)
        if content_ref is not None:
            definition.append(navigate_json_ref(data, content_ref))
        else:
            if item.get('type', None) is not None:
                definition.append(item.get('type', None))
            elif item.get('const', None) is not None:
                definition.append(item.get('const', None))
    return definition

def _handle_all_of(all_of: list, data: dict):
    definition = []
    for item in all_of:
        content_ref = item.get('$ref', None)
        if content_ref is not None:
            definition.append(navigate_json_ref(data, content_ref))
        if item.get('properties', None):
            definition += _extract_properties(data, item.get('properties', {}), item.get('required', []))
        else:
            if item.get('type', None) is not None:
                definition.append(item.get('type', None))
            elif item.get('const', None) is not None:
                definition.append(item.get('const', None))
    return definition

def extract_definition(data, schema):
    one_of = schema.get('oneOf', None)
    items = schema.get('items', None)
    any_of = schema.get('anyOf', None)
    all_of = schema.get('allOf', None)
    if one_of is not None:
        for item in one_of:
            content_ref = item.get('$ref', None)
            definition = navigate_json_ref(data, content_ref)
            if definition is None:
                print("Error: Invalid JSON file, missing definition")
    elif all_of is not None:
        definition = _handle_all_of(all_of, data)
    elif items is not None and items:
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
        # type can be any_of the types in the list
        # so the definition is a list of types
        definition = _extract_any_of(any_of, data)
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

def _extract_parameters(data: dict, parameters: list):
    extracted_parameters = []
    for parameter in parameters:
        schema = parameter.get('schema', {})
        content_ref = schema.get('$ref', None)
        parameter_ref = parameter.get('$ref', None)
        if content_ref is not None or parameter_ref is not None:
            definition = navigate_json_ref(data, content_ref if content_ref is not None else parameter_ref)
            if definition is None:
                print("Error: Invalid JSON file, missing definition when extracting parameters")
            extracted_parameters.append(definition)
        elif parameter.get('type', None):
            extracted_parameters.append(f"{parameter.get('name', '')}: {parameter.get('type', '')}")
        elif schema and schema.get('properties', None):
            extracted_parameters += _extract_properties(data, schema.get('properties', {}), schema.get('required', []))
        else:
            type_definition = extract_definition(data, schema)
            extracted_parameters.append(f"{parameter.get('name', '')}: {type_definition}")
    return extracted_parameters

def _extract_request_body(data: dict, request_body: dict):
    schema = request_body.get('content', {}).get('application/json', {}).get('schema', {})
    if not schema:
        return []
    content_ref = schema.get('$ref', None)
    if content_ref is not None:
        definition = navigate_json_ref(data, content_ref)
        if definition is None:
            print("Error: Invalid JSON file, missing definition when extracting request body")
        properties = _extract_properties(data, definition.get('properties', {}), definition.get('required', []))
        return properties
    else:
        return extract_definition(data, schema)


def _extract_properties(data: dict, properties: dict, required: list = []):
    extracted_properties = []
    for prop, details in properties.items():
        type_info = details.get('type', '')
        if not type_info and details.get('anyOf', None):
            type_info = _extract_any_of(details.get('anyOf', []), data)
        if details.get('$ref', None):
            definition = navigate_json_ref(data, details.get('$ref', ''))
            if definition is None:
                print("Error: Invalid JSON file, missing definition, when extracting properties")
            type_info = _extract_properties(data, definition.get('properties', {}), required)
        extracted_properties.append({'title': prop, 'type': type_info, 'required': prop in required})
    return extracted_properties

def _handle_resonse_list(data: dict, responses: list):
    extracted_responses = []
    for response in responses:
        schema = responses[response].get('schema', {})
        if not schema:
            continue
        content_ref = schema.get('$ref', None)
        if content_ref is not None:
            definition = navigate_json_ref(data, content_ref)
            if definition is None:
                print("Error: Invalid JSON file, missing definition, when extracting response body")
            if definition.get('properties', None):
                extracted_responses += _extract_properties(data, definition.get('properties', {}), definition.get('required', []))
            else:
                extracted_responses.append(definition)
        else:
            if schema.get('title', None):
                extracted_responses.append({'title': schema.get('title', ''), 'type': schema.get('type', '')})
            else:
                extracted_responses.append(extract_definition(data, schema))
    return extracted_responses

def _extract_response_body(data: dict, response_body: dict):
    response_data = response_body.get('200', {})
    if response_data.get('content', None):
        response_data = response_data.get('content')
    extracted_responses = []

    if isinstance(response_data, list):
        return _handle_resonse_list(data, response_data)
    
    schema = response_data.get('schema', {})
    content_ref = schema.get('$ref', None)

    if not schema and content_ref is None:
        return []
    if content_ref is not None:
        definition = navigate_json_ref(data, content_ref)
        if definition is None:
            print("Error: Invalid JSON file, missing definition, when extracting response body")
        if definition.get('properties', None):
            extracted_responses += _extract_properties(data, definition.get('properties', {}), definition.get('required', []))
        else:
            extracted_responses.append(definition)
    else:
        if schema.get('title', None):
            extracted_responses.append({'title': schema.get('title', ''), 'type': schema.get('type', '')})
        else:
            extracted_responses.append(extract_definition(data, schema))
    return extracted_responses

def parse_file(data: dict):
    # parse the file and return a list of the contents
    api_info = APIInfo()
    try:
        paths = data.get('paths', {})
        for path, path_details in paths.items():
            # take the parameters from the path
            # take the request body
            # take the response body
            # take the summary and path name
            path_info = PathInfo()
            path_info.path_name = path
            for method, method_details in path_details.items():
                params = _extract_parameters(data, method_details.get('parameters', []))
                responses = _extract_response_body(data, method_details.get('responses', {}))
                request_body = _extract_request_body(data, method_details.get('requestBody', {}))
                summary = method_details.get('summary', '')
                content = Content(method=method, summary=summary, parameters=params, request_body=request_body, response_body=responses)
                path_info.methods[method] = content
            api_info.paths[path] = path_info
        return api_info
    except Exception as exc:
        print(f"Error: Invalid JSON file, missing key {exc}")
        return None

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
            path_context += f"<Path> {path} </Path> <Method> {method} </Method> <Summary> {api_info.paths[path].methods[method].summary} </Summary> <Parameters> {api_info.paths[path].methods[method].parameters} </Parameters> <Response> {reduce_response} </Response>\n"
    return path_context

def _generate_full_path_context(api_info: APIInfo):
    """ Generates a textual output for each path and what its query parameters are. """
    schema_context = ""
    for path in api_info.paths:
        for method in api_info.paths[path].methods:
            parameters = api_info.paths[path].methods[method].parameters
            request_body = {}
            if api_info.paths[path].methods[method].request_body:
                request_body = api_info.paths[path].methods[method].request_body
            schema_context += f"<Path> {path} </Path> <Method> {method} </Method> <Summary> {api_info.paths[path].methods[method].summary} </Summary> <Parameters> {parameters} </Parameters> <Request> {request_body} </Request>\n"
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
    input_folder = "./data/broken_files/"
    files = os.listdir(input_folder)
    for file in files:
        f = open(input_folder+file)
        try:
            data = json.load(f)
            api_information = parse_file(data)
            if not api_information:
                f.close()
                _move_file(folder=input_folder, file=file, destination="./data/parsed_and_broken")
                continue
            path_context = _generate_path_to_response(api_information)
            full_schema_context = _generate_full_path_context(api_information)
            if path_context == "" or full_schema_context == "":
                f.close()
                _move_file(folder=input_folder, file=file, destination="./data/parsed_and_broken")
                continue
            key = file
            values = {"path_context": path_context, "full_schema_context": full_schema_context}
            _dump_to_file("./data/further_output.json", key, values)
            f.close()
            _move_file(folder=input_folder, file=file, destination="./data/data_set_parsed")
            api_information = None
        except Exception as e:
            print('Error: Invalid JSON file')
            f.close()
            _move_file(folder=input_folder, file=file, destination="./data/parsed_and_broken")
    


