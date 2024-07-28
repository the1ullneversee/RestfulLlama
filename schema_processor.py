import structlog
from collections import defaultdict
from dataclasses import dataclass
import re

logger = structlog.get_logger()

def to_camel_case(s):
    if len(s) == 0:
        return s
    s = re.sub(r"(_|-)+", " ", s).title().replace(" ", "")
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
    # methods: dict[str, Content] = {}

    def __init__(self):
        self.methods = defaultdict(MethodInfo)


class APIInfo:

    # paths: dict[str, PathInfo] = {}

    def __init__(self):
        self.paths = defaultdict(PathInfo, {})

    def __str__(self):
        return f"Paths: {self.paths}"


def navigate_json_ref(json_obj, ref):
    if ref is None:
        return None
    keys = ref.split("/")[
        1:
    ]  # split the ref into keys, ignoring the first empty string
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
        content_ref = item.get("$ref", None)
        if content_ref is not None:
            definition.append(navigate_json_ref(data, content_ref))
        else:
            if item.get("type", None) is not None:
                definition.append(item.get("type", None))
            elif item.get("const", None) is not None:
                definition.append(item.get("const", None))
    return definition


def _handle_all_of(all_of: list, data: dict, global_definitions: dict = {}):
    definition = []
    for item in all_of:
        content_ref = item.get("$ref", None)
        if content_ref is not None:
            definition.append(navigate_json_ref(data, content_ref))
        if item.get("properties", None):
            definition += _extract_properties(
                data, item.get("properties", {}), item.get("required", [])
            )
        else:
            if item.get("type", None) is not None:
                definition.append(item.get("type", None))
            elif item.get("const", None) is not None:
                definition.append(item.get("const", None))
    return definition


def _handle_items(
    items: dict,
    data: dict,
    caller_properties=None,
    global_definitions: dict = {},
    definition_name: str = "",
):
    definition = []
    if isinstance(items, list):
        for item in items:
            definition += extract_definition(data, item)
    else:
        content_ref = items.get("$ref", None)
        items_any_of = items.get("anyOf", None)
        items_one_of = items.get("oneOf", None)
        singular_type = items.get("type", None)
        if items_any_of is not None:
            return extract_definition(data, items)
        if items_one_of is not None:
            return extract_definition(data, items)
        if singular_type is not None:
            return singular_type

        if global_definitions:
            if content_ref is not None and content_ref in global_definitions:
                return global_definitions.get(content_ref)

        if content_ref:
            definition_ref = content_ref.split("/")[-1]
            if definition_ref == definition_name:
                logger.debug("Circular reference detected")
                return f"<DEFINE LATER> {definition_name}"
        if content_ref:
            def_ref = content_ref.split("/")[-1]
            if def_ref in global_definitions:
                return global_definitions.get(def_ref)
        definition = navigate_json_ref(data, content_ref)
        if definition is None:
            logger.debug("Error: Invalid JSON file, missing definition")

        if definition.get("properties", None):
            definition = _extract_properties(
                data,
                definition.get("properties", {}),
                definition.get("required", []),
                caller_properties=caller_properties,
                global_definitions=global_definitions,
                definition_name=definition_name,
            )
    return definition


def extract_definition(data, schema, global_definitions=None, caller_properties=None):
    one_of = schema.get("oneOf", None)
    items = schema.get("items", None)
    any_of = schema.get("anyOf", None)
    all_of = schema.get("allOf", None)
    if one_of is not None:
        for item in one_of:
            content_ref = item.get("$ref", None)
            definition = navigate_json_ref(data, content_ref)
            if definition is None:
                logger.debug("Error: Invalid JSON file, missing definition, in extract definition")
    elif all_of is not None:
        definition = _handle_all_of(all_of, data)
    elif items is not None and items:
        definition = _handle_items(items, data, global_definitions=global_definitions)
    elif any_of is not None:
        # type can be any_of the types in the list
        # so the definition is a list of types
        definition = _extract_any_of(any_of, data)
    else:
        content_ref = schema.get("$ref", None)
        if content_ref is not None:
            definition = navigate_json_ref(data, content_ref)
            if definition is None:
                logger.debug("Error: Invalid JSON file, missing definition, in extract definition")
            def_properties = definition.get("properties", {})
            # data_properties = def_properties.get('data', None)
            # if data_properties is not None:
            #     return extract_definition(data, data_properties)
            if def_properties.get("type", None) == "array":
                definition += extract_definition(data, definition.get("items", None))
        else:
            definition = schema.get("type", None)
            if definition is None:
                logger.debug("Error: Invalid JSON file, missing definition, in extract definition")

    return definition


def _extract_parameters(data: dict, parameters: list, global_definitions=None):
    extracted_parameters = []
    for parameter in parameters:
        schema = parameter.get("schema", {})
        content_ref = schema.get("$ref", None)
        parameter_ref = parameter.get("$ref", None)
        if content_ref is not None or parameter_ref is not None:
            definition = navigate_json_ref(
                data, content_ref if content_ref is not None else parameter_ref
            )
            if definition is None:
                logger.debug("Error: Invalid JSON file, missing definition when extracting parameters")
            extracted_parameters.append(definition)
        elif parameter.get("type", None):
            extracted_parameters.append(
                f"{parameter.get('name', '')}: {parameter.get('type', '')}, required: {parameter.get('required', False)}"
            )
        elif schema and schema.get("properties", None):
            extracted_parameters += _extract_properties(
                data=data,
                properties=schema.get("properties", {}),
                required=schema.get("required", []),
                global_definitions=global_definitions,
            )
        else:
            type_definition = extract_definition(data, schema, global_definitions)
            extracted_parameters.append(
                f"{parameter.get('name', '')}: {type_definition}, required: {parameter.get('required', False)}"
            )
    return extracted_parameters


def _extract_request_body(data: dict, request_body: dict, global_definitions=None):
    schema = (
        request_body.get("content", {}).get("application/json", {}).get("schema", {})
    )
    if not schema:
        return []
    content_ref = schema.get("$ref", None)
    if content_ref is not None:
        definition = navigate_json_ref(data, content_ref)
        if definition is None:
            logger.debug("Error: Invalid JSON file, missing definition when extracting request body")
        properties = _extract_properties(
            data=data,
            properties=definition.get("properties", {}),
            required=definition.get("required", []),
            global_definitions=global_definitions,
        )
        return properties
    else:
        return extract_definition(data, schema, global_definitions=global_definitions)


def _look_for_ref(
    data: dict, caller_properties: dict, global_definitions: dict, caller: str
):
    keys = ["items", "anyOf", "oneOf", "allOf", "$ref"]
    caller_prop_names = list(caller_properties[caller].keys())
    if any(key in caller_prop_names for key in keys):
        key = next(key for key in keys if key in caller_prop_names)
    else:
        key = None
    match key:
        case "items":
            ref = caller_properties[caller].get("items", {}).get("$ref", None)
        case "anyOf":
            ref = caller_properties[caller].get("anyOf", [{}])[0]

        case "oneOf":
            ref = caller_properties[caller].get("oneOf", [{}])[0].get("$ref", None)
        case "allOf":
            ref = caller_properties[caller].get("allOf", [{}])[0].get("$ref", None)
        case "$ref":
            ref = caller_properties[caller].get("$ref", None)
        case _:
            ref = None
            logger.debug("Error: Invalid caller")

    if isinstance(ref, dict):
        return extract_definition(
            data=data, schema=ref, global_definitions=global_definitions
        )
    if isinstance(ref, str):
        ref = ref.split("/")[-1]
        if ref in global_definitions:
            return global_definitions.get(ref)
    return None


def extract_definition_for_global(data: dict, schema: dict, global_definitions=None):
    extracted_properties = []
    try:
        schema_properties = schema.get("properties")
        required_properties = schema.get("required", [])
        special_keys = ["items", "anyOf", "oneOf", "allOf", "$ref"]
        for prop, details in schema_properties.items():
            type_info = details.get("type", "")
            type_value = ""
            enum = ""
            if any(key in details for key in special_keys):
                if "$ref" in details:
                    content_ref = details.get("$ref", None).split("/")[-1]
                    if content_ref not in global_definitions:
                        type_value = f"<DEFINE LATER> {content_ref}"
                    else:
                        type_value = global_definitions.get(content_ref)
                elif "items" in details:
                    for item_key, value in details.get("items", []).items():
                        if "$ref" == item_key:
                            content_ref = value.split("/")[-1]
                            if content_ref not in global_definitions:
                                type_value = f"<DEFINE LATER> {content_ref}"
                            else:
                                type_value = global_definitions.get(content_ref)
                        else:
                            if item_key == "type":
                                type_value += value
                            elif item_key == "enum":
                                enum = value
                elif "anyOf" in details:
                    values_could_be = []
                    for item in details.get("anyOf", []):
                        if "$ref" in item:
                            content_ref = item.get("$ref", "").split("/")[-1]
                            if content_ref not in global_definitions:
                                values_could_be.append(f"<DEFINE LATER> {content_ref}")
                                # type_value = f"<DEFINE LATER> {content_ref}"
                            else:
                                definition = global_definitions.get(content_ref)
                                if isinstance(definition, list):
                                    values_could_be.append(str(definition))
                                else:
                                    values_could_be.append(definition)
                                ##type_value = global_definitions.get(content_ref)
                        else:
                            if "type" in item:
                                values_could_be.append(item.get("type", ""))
                            if "enum" in item:
                                enum = item.get("enum", "")
                    type_value = " or ".join(values_could_be)
            else:
                if isinstance(type_info, str) and type_info != "":
                    type_value = type_info
                    if "enum" in details:
                        enum = details.get("enum")
                else:
                    type_value = details
            extracted_properties.append(
                {
                    "title": prop,
                    "type": type_value,
                    "enum": enum,
                    "required": prop in required_properties,
                }
            )
        return extracted_properties

    except Exception as exc:
        logger.error(f"Error: Something went wrong when extracting properties {exc}")
        raise exc
    return extracted_properties


def _extract_properties(
    data: dict,
    properties: dict,
    required: list = [],
    caller_properties: dict = {},
    global_definitions: dict = {},
    definition_name: str = "",
):
    extracted_properties = []
    try:
        for prop, details in properties.items():
            type_info = details.get("type", "")
            if not type_info and details.get("anyOf", None):
                type_info = _extract_any_of(details.get("anyOf", []), data)
            if details.get("$ref", None):
                content_ref = details.get("$ref", None)
                if content_ref:
                    def_ref = content_ref.split("/")[-1]
                    if def_ref in global_definitions:
                        type_info = global_definitions.get(def_ref)

                # definition = navigate_json_ref(data, details.get("$ref", ""))
                # if definition is None:
                #     print(
                #         "Error: Invalid JSON file, missing definition, when extracting properties"
                #     )
                #     if definition.get("properties", None):
                #         type_info = _extract_properties(
                #             data,
                #             definition.get("properties", {}),
                #             definition.get("required", []),
                #             caller_properties=properties,
                #             global_definitions=global_definitions,
                #             definition_name=definition_name,
                #         )
            if details.get("items", None):
                caller = prop
                if caller_properties and caller in caller_properties:
                    type_info = _look_for_ref(
                        data=data,
                        caller_properties=caller_properties,
                        global_definitions=global_definitions,
                        caller=caller,
                    )
                else:
                    type_info = _handle_items(
                        details.get("items", {}),
                        data,
                        caller_properties=properties,
                        global_definitions=global_definitions,
                        definition_name=definition_name,
                    )
            extracted_properties.append(
                {"title": prop, "type": type_info, "required": prop in required}
            )
    except Exception as exc:
        logger.error(f"Error: Something went wrong when extracting properties {exc}")
        raise exc
    return extracted_properties


def ___extract_properties_old(
    data: dict,
    properties: dict,
    required: list = [],
    caller_properties: dict = {},
    global_definitions: dict = {},
    definition_name: str = "",
):
    extracted_properties = []
    try:
        for prop, details in properties.items():
            type_info = details.get("type", "")
            if not type_info and details.get("anyOf", None):
                type_info = _extract_any_of(details.get("anyOf", []), data)
            if details.get("$ref", None):
                definition = navigate_json_ref(data, details.get("$ref", ""))
                if definition is None:
                    logger.debug(
                        "Error: Invalid JSON file, missing definition, when extracting properties"
                    )
                if definition == prop:
                    logger.debug("Error: Circular reference detected")
                    if definition in global_definitions:
                        type_info = global_definitions.get(definition)
                    continue
                if definition.get("properties", None):
                    type_info = _extract_properties(
                        data,
                        definition.get("properties", {}),
                        definition.get("required", []),
                        caller_properties=properties,
                        global_definitions=global_definitions,
                        definition_name=definition_name,
                    )
                    if "$ref" in details:
                        target_ref = details.get("$ref").split("/")[-1]
                        global_definitions[target_ref] = type_info
            if details.get("items", None):
                caller = prop
                if caller_properties and caller in caller_properties:
                    type_info = _look_for_ref(
                        data=data,
                        caller_properties=caller_properties,
                        global_definitions=global_definitions,
                        caller=caller,
                    )
                else:
                    type_info = _handle_items(
                        details.get("items", {}),
                        data,
                        caller_properties=properties,
                        global_definitions=global_definitions,
                        definition_name=definition_name,
                    )
                    if "$ref" in details:
                        target_ref = details.get("$ref").split("/")[-1]
                        global_definitions[target_ref] = type_info
            extracted_properties.append(
                {"title": prop, "type": type_info, "required": prop in required}
            )
    except Exception as exc:
        logger.error(f"Error: Something went wrong when extracting properties {exc}")
        raise exc
    return extracted_properties


def _flatten_definition(data: dict, parent_definition: dict):
    properties = parent_definition.get("properties", {})
    required = parent_definition.get("required", [])
    extracted_properties = []
    circular_references = []
    for prop, details in properties.items():
        logger.debug(f"Prop: {prop} Details: {details}")
        for prop, details in properties.items():
            type_info = details.get("type", "")
            if not type_info and details.get("anyOf", None):
                type_info = _extract_any_of(details.get("anyOf", []), data)
            if details.get("$ref", None):
                definition = navigate_json_ref(data, details.get("$ref", ""))
                if definition is None:
                    logger.debug(
                        "Error: Invalid JSON file, missing definition, when extracting properties"
                    )
                if definition == parent_definition.get("title", None):
                    logger.debug("Error: Circular reference detected")
                    circular_references.append(definition)
                    continue
                if definition.get("properties", None):
                    type_info = _extract_properties(
                        data,
                        definition.get("properties", {}),
                        definition.get("required", []),
                        caller_properties=properties,
                    )
            if details.get("items", None):
                caller = prop
                type_info = _handle_items(caller, details.get("items", {}), data)
            extracted_properties.append(
                {"title": prop, "type": type_info, "required": prop in required}
            )
    return {}


def _extract_response_body(data: dict, response_body: dict, global_definitions=None):
    response_data = response_body.get("200", {})
    if response_data.get("content", None):
        response_data = response_data.get("content")

    if response_data.get("application/json", None):
        response_data = response_data.get("application/json")
    extracted_responses = []

    schema = response_data.get("schema", {})
    content_ref = schema.get("$ref", None)

    if not schema and content_ref is None:
        return []
    if content_ref is not None:
        definition = navigate_json_ref(data, content_ref)
        if definition is None:
            logger.debug(
                "Error: Invalid JSON file, missing definition, when extracting response body"
            )
        if definition.get("properties", None):
            extracted_responses += _extract_properties(
                data=data,
                properties=definition.get("properties", {}),
                required=definition.get("required", []),
                global_definitions=global_definitions,
            )
        else:
            extracted_responses.append(definition)
    else:
        if schema.get("title", None):
            extracted_responses.append(
                {"title": schema.get("title", ""), "type": schema.get("type", "")}
            )
        else:
            extracted_responses.append(
                extract_definition(data, schema, global_definitions)
            )
    return extracted_responses


def _get_definitions(data: dict):

    if "components" in data:
        return data.get("components", {}).get("schemas", {})
    if "definitions" in data:
        return data.get("definitions", {})
    logger.debug("Error: Invalid JSON file, missing definitions")


def process_global_definitions(data: dict):
    components = _get_definitions(data)
    global_definitions = {}
    try:
        for key, schema in components.items():
            schema_properties = schema.get("properties")
            if not schema_properties:
                global_definitions[key] = extract_definition(
                    data, schema, global_definitions=global_definitions
                )
            else:
                global_definitions[key] = extract_definition_for_global(
                    data=data, schema=schema, global_definitions=global_definitions
                )
        # loop through each definition
        for key, value in global_definitions.items():
            if isinstance(value, list):
                for item in value:
                    if "<DEFINE LATER>" in item.get("type"):
                        # get the definition
                        definition = item.get("type").split(" ")[-1]
                        if definition in global_definitions:
                            defi = get_definition(global_definitions, definition)
                            item["type"] = defi
    except Exception as exc:
        logger.error(
            f"Error: Something went wrong when processing global definitions {exc}"
        )
        raise exc
    return global_definitions


def get_definition(global_definitions: dict, caller_definition_name: str):
    definition = global_definitions.get(caller_definition_name)
    self_definition = []
    if isinstance(definition, list):
        for item in definition:
            if "<DEFINE LATER>" in item.get("type"):
                definition_name = item.get("type").split(" ")[-1]
                if (
                    definition_name in global_definitions
                    and definition_name != caller_definition_name
                ):
                    item["type"] = get_definition(global_definitions, definition_name)
                elif definition_name == caller_definition_name:
                    logger.debug("Circular reference detected")
                    self_definition.append(item)
    for item in self_definition:
        for ix in range(len(definition)):
            if definition[ix] == item:
                definition[ix] = definition
    return definition


def parse_file(data: dict):
    # parse the file and return a list of the contents
    api_info = APIInfo()
    try:
        paths = data.get("paths", {})
        global_definitions = process_global_definitions(data)
        for path, path_details in paths.items():
            # take the parameters from the path
            # take the request body
            # take the response body
            # take the summary and path name
            path_info = PathInfo()
            path_info.path_name = path
            for method, method_details in path_details.items():
                logger.debug(f"Path: {path}, Method: {method} Method Details: {method_details}")
                params = _extract_parameters(
                    data,
                    method_details.get("parameters", []),
                    global_definitions=global_definitions,
                )
                responses = _extract_response_body(
                    data, method_details.get("responses", {}), global_definitions
                )
                request_body = _extract_request_body(
                    data, method_details.get("requestBody", {}), global_definitions
                )
                summary = method_details.get("summary", "")
                content = Content(
                    method=method,
                    summary=summary,
                    parameters=params,
                    request_body=request_body,
                    response_body=responses,
                )
                path_info.methods[method] = content
            api_info.paths[path] = path_info
        return api_info
    except Exception as exc:
        logger.error(f"Error: Something went wrong when parsing the file {exc}")
        raise exc


def reduce_json_info(json_response):
    if not json_response:
        return []
    if not isinstance(json_response, dict):
        return [json_response]
    properties = json_response.get("properties", {})
    reduced_info = []
    for prop, details in properties.items():
        title = details.get("title", "")
        if title.lower() not in json_response.get("required", []):
            continue
        type_info = details.get("type", "")
        if not type_info:
            type_info = [item.get("type", "") for item in details.get("anyOf", [])]
            type_info = [t for t in type_info if t]  # remove empty strings
        reduced_info.append({"title": title, "type": type_info})
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
            path_context += f"<Path> {path} </Path> <Method> {method} </Method> <Summary> {api_info.paths[path].methods[method].summary} </Summary>\n"
    return path_context


def _generate_full_path_context(api_info: APIInfo):
    """Generates a textual output for each path and what its query parameters are."""
    schema_context = defaultdict(dict)
    for path in api_info.paths:
        for method in api_info.paths[path].methods:
            parameters = api_info.paths[path].methods[method].parameters
            request_body = {}
            response = {}
            if api_info.paths[path].methods[method].request_body:
                request_body = api_info.paths[path].methods[method].request_body
            if api_info.paths[path].methods[method].response_body:
                response = api_info.paths[path].methods[method].response_body
            reduce_response = reduce_json_info(response)
            schema_context[path][method] = f"<Path> {path} </Path> <Method> {method} </Method> <Summary> {api_info.paths[path].methods[method].summary} </Summary> <Parameters> {parameters} </Parameters> <Request> {request_body} </Request> <Response> {reduce_response} </Response>\n"
            #schema_context[path] = f"<Path> {path} </Path> <Method> {method} </Method> <Summary> {api_info.paths[path].methods[method].summary} </Summary> <Parameters> {parameters} </Parameters> <Request> {request_body} </Request> <Response> {reduce_response} </Response>\n"
           #schema_context += f"<Path> {path} </Path> <Method> {method} </Method> <Summary> {api_info.paths[path].methods[method].summary} </Summary> <Parameters> {parameters} </Parameters> <Request> {request_body} </Request> <Response> {reduce_response} </Response>\n"
    return schema_context

def execute_context_call(full_schema_context, context_call):
    """Executes the context call and returns the schema for the requested endpoint."""

    path_index= context_call.find("path=")
    path = context_call[path_index+6:context_call.find(",", path_index)-1]
    sub_string = context_call[path_index+6:]
    method_index = sub_string.find("method=")
    delimiter = sub_string.find(")")
    if not delimiter:
        delimiter = sub_string.find(",", method_index)
    method = sub_string[method_index+8:delimiter-1]

    # Get the schema for the requested endpoint
    schema = full_schema_context.get(path, {}).get(method)
    if not schema:
        return None
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