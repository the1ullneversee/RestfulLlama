import re

input_file = "./data/0_output.json"

import json
from collections import defaultdict


def _clean_value(value):
    value = value.strip()
    value = value.replace(" ", "")
    value = value.replace("'", "")
    value = value.replace("[", "")
    value = value.replace("]", "")
    value = value.replace("}", "")
    value = value.replace("{", "")
    return value


def _extract_to_delimeter(line, from_delimeter, to_delimeter):
    start_idx = line.find(from_delimeter)
    line_from_start = line[start_idx:]
    end_idx = line_from_start.find(to_delimeter) + start_idx
    value = _clean_value(line[start_idx + len(from_delimeter) : end_idx])
    return value, line[end_idx + 1 :]


f = open(input_file, "r")
lines = f.readlines()
f.close()
line = lines[-1]


def extract_method(l_path_context):
    start_idx = l_path_context.find("<Method>")
    end_idx = l_path_context.find("</Method>")
    method = l_path_context[start_idx + len("<Method>") : end_idx]
    return method, l_path_context[end_idx + len("</Method>") :]


def extract_parameters(path_parameters, l_path_context, r_key):
    start_idx = l_path_context.find("<Parameters>")
    end_idx = l_path_context.find("</Parameters>")
    parameters = l_path_context[start_idx + len("<Parameters>") : end_idx]
    parameters = (
        parameters.replace("[", "")
        .replace("]", "")
        .replace("'", "")
        .replace(" ", "")
        .split(",")
    )
    for i in range(len(parameters)):
        parameters[i] = parameters[i].split(":")[0]
    path_parameters[r_key] = parameters
    return l_path_context[end_idx + len("</Parameters>") :]


def extract_response_parameters(response_parameters, l_path_context, r_key):
    start_idx = l_path_context.find("<Response>")
    end_idx = l_path_context.find("</Response>")
    response = l_path_context[start_idx + len("<Response>") : end_idx]
    response = (
        response.replace("[", "").replace("]", "").replace("'", "").replace(" ", "")
    )
    while response.find("title:") != -1:
        r_param, response = _extract_to_delimeter(
            response, from_delimeter="title:", to_delimeter=","
        )
        response_parameters[r_key].append(r_param)

    return l_path_context[end_idx + len("</Response>") :]


def extract_request_parameters(l_path_context):
    request_params = []
    start_idx = l_path_context.find("<Request>")
    end_idx = l_path_context.find("</Request>")
    request = l_path_context[start_idx + len("<Request>") : end_idx]
    request = (
        request.replace("[", "").replace("]", "").replace("'", "").replace(" ", "")
    )
    while request.find("title:") != -1:
        r_param, request = _extract_to_delimeter(
            request, from_delimeter="title:", to_delimeter=","
        )
        r_required, request = _extract_to_delimeter(
            request, from_delimeter="required:", to_delimeter=","
        )
        request_params.append((r_param, r_required))
    return request_params, l_path_context[end_idx + len("</Request>") :]


while True:
    data = json.loads(line)
    path_parameters = {}
    response_parameters = defaultdict(list)
    request_params = defaultdict(list)
    for key, value in data.items():
        main_path_context = value["full_schema_context"]
        l_path_context = main_path_context
        # chunk by paths
        start_idx = 0
        while True:
            start_idx = l_path_context.find("<Path>", start_idx)
            if start_idx == -1:
                break
            end_idx = l_path_context.find("</Path>", start_idx)
            path, l_path_context = (
                l_path_context[start_idx + len("<Path>") : end_idx],
                l_path_context[end_idx + len("</Path>") :],
            )

            # now get the method
            method, l_path_context = extract_method(l_path_context)

            r_key = (path + method).strip()
            path_parameters[r_key] = []
            response_parameters[r_key] = []

            l_path_context = extract_parameters(path_parameters, l_path_context, r_key)

            l_request_params, l_path_context = extract_request_parameters(
                l_path_context
            )
            request_params[r_key] = l_request_params

            l_path_context = extract_response_parameters(
                response_parameters, l_path_context, r_key
            )

    for key, value in request_params.items():
        # check if any of the parameters are required
        for i in range(len(value)):
            if value[i][1] == "True":
                print(f"Path {key} has a required parameter '{value[i][0]}'")
                # find the endpoint that get's this parameter
                for k, v in response_parameters.items():
                    if value[0][0] in v:
                        print(f"Endpoint: {k}")
                        break
                if bool(re.search(r"\{.+?\}", key)):
                    print("Path has a parameter in the path itself")
    print(main_path_context)
    break
    # do something with data
