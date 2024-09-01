import copy
import json
import os
import sys

from colorama import Fore
import structlog
from openai import OpenAI

from data_generation.schema_processor import (
    _generate_full_path_context,
    _generate_path_to_response,
    execute_context_call,
    parse_file,
)
from shared.helper_funcs import stream_output
from model_converser import load_language_model

logger = structlog.get_logger()

use_llama = True
repo_name = "the1ullneversee"
model_name = "RestfulLlama-8B-Instruct"
file_name = "restful_llama_8_Q8.gguf"

client = OpenAI(base_url="http://172.29.0.1:1234/v1", api_key="lm-studio")


def save_response(history, question_metadata, context_type, model_name):
    file_name = "{model}_{context_type}_history.json".format(
        model=model_name, context_type=context_type
    )
    with open(file_name, "a") as f:
        question_metadata["history"] = history
        f.write(json.dumps(question_metadata, indent=2) + "\n")
    return


def prompt_openai(history, model_name):
    new_message = {"role": "assistant", "content": ""}
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=history,
            temperature=0.7,
            stream=True,
        )
        for chunk in completion:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
                new_message["content"] += chunk.choices[0].delta.content
    except Exception as e:
        print("Error: OpenAI API call")
    return new_message


def prompt_llm_cpp(llm, history):
    answer = llm.create_chat_completion(messages=history)
    return answer.get("choices")[0].get("message")


def prompt_llm(history, use_llama, model_name, llm):
    if use_llama:
        return prompt_llm_cpp(llm, history)
    else:
        return prompt_openai(history, model_name)


whole_context = """You are a helpful assistant that can generate Python code to interact with an application's API. You have access to the application's API endpoints and their corresponding schemas.
                When a user asks a question, your task is to ask for additional context about the API endpoints as needed, reason about the schema context, and how it can answer the question or part of it.
                You can ask for context about an API endpoint by making a call to get_context, like so: get_context(path='/path/', method='method').
                Look at the schema context to understand what PARAMETERS are required to call each endpoint.
                The application's API endpoints are: {path_context}
                Please respond with a get_context request to clarify the API endpoint needed to answer the user's question.
                Remember to use the schema context to generate the Python code. And to ask for context about an API endpoint, use the get_context function call.
                """

partial_schema_context = """You are a helpful assistant that can generate Python code to interact with an application's API. You have access to the application's API endpoints and their corresponding schemas.
                When a user asks a question, your task is to ask for additional context about the API endpoints as needed, reason about the schema context, and how it can answer the question or part of it.
                You can ask for context about an API endpoint by making a call to get_context function call, that takes two arguments: path and method. path=/path/, method='method'.
                Look at the schema context to understand what PARAMETERS are required to call each endpoint.
                The application's API endpoints are: {path_context}
                Please respond with a get_context request to clarify the API endpoint needed to answer the user's question.
                Remember to use the schema context to generate the Python code"""

limited_schema_context = """You are a helpful assistant that can generate Python code to interact with an application's API. You have access to the application's API endpoints and their corresponding schemas.
                When a user asks a question, your task is to ask for additional context about the API endpoints as needed, reason about the schema context, and how it can answer the question or part of it.
                You can ask for context about an API endpoint by making a call to get_context.
                Look at the schema context to understand what PARAMETERS are required to call each endpoint.
                The application's API endpoints are: {path_context}
                Please respond with a get_context request to clarify the API endpoint needed to answer the user's question.
                Remember to use the schema context to generate the Python code"""

all_system_contexts = {
    "whole_context": whole_context,
    "partial_schema_context": partial_schema_context,
    "limited_schema_context": limited_schema_context,
}


def generate_schema_info(
    logger,
    whole_context,
    partial_schema_context,
    limited_schema_context,
    all_system_contexts,
    input_file,
):
    with open(input_file) as f:
        logger.info("Reading Evaluation File")
        try:
            data = json.load(f)
            api_information = parse_file(data)
            if not api_information:
                sys.exit(0)
            # api_information= process_paths(api_information=api_information)
            path_context = _generate_path_to_response(api_information)
            full_schema_context = _generate_full_path_context(api_information)

            all_system_contexts["whole_context"] = whole_context.format(
                path_context=path_context
            )
            all_system_contexts["partial_schema_context"] = (
                partial_schema_context.format(path_context=path_context)
            )
            all_system_contexts["limited_schema_context"] = (
                limited_schema_context.format(path_context=path_context)
            )
        except Exception as e:
            logger.error("Error: Invalid JSON file")
    return api_information, full_schema_context


def construct_evaluation_metadata(api_information):
    questions_to_metadata = {}
    questions_to_metadata = [
        {
            "questions": ["Can you get me a list of events?"],
            "target_context_calls": ["get_context(path='/event/', method='get')"],
            "parameters": [
                api_information.paths.get("/event/").methods.get("get").parameters
            ],
        },
        {
            "questions": ["Get a list of products that are archived"],
            "target_context_calls": ["get_context(path='/product/', method='get')"],
            "parameters": [
                api_information.paths.get("/product/").methods.get("get").parameters
            ],
        },
        {
            "questions": [
                "Can you get me all performances for an event that start after tomorrow?"
            ],
            "target_context_calls": [
                "get_context(path='/event/{event_id}/performance/', method='get')"
            ],
            "parameters": [
                api_information.paths.get("/event/{event_id}/performance/")
                .methods.get("get")
                .parameters
            ],
        },
        {
            "questions": [
                "I have a discount with ID 5, and I want to get the visbility for the discount.",
                "Now I want to update the visibility to start from tomorrow and end in a months time.",
            ],
            "target_context_calls": [
                "get_context(path='/discount/{discount_id}/visibility/', method='get')",
                "get_context(path='/discount/{discount_id}/visibility/', method='put')",
            ],
            "parameters": [
                api_information.paths.get("/discount/{discount_id}/visibility/")
                .methods.get("get")
                .parameters,
                api_information.paths.get("/discount/{discount_id}/visibility/")
                .methods.get("put")
                .parameters,
            ],
        },
        {
            "questions": [
                "Can you change all performances that start after tomorrow to start at 10:00 AM for Event 1?"
            ],
            "target_context_calls": [
                "get_context(path='/event/{event_id}/performance/bulk/', method='put')"
            ],
            "parameters": [
                api_information.paths.get("/event/{event_id}/performance/bulk/")
                .methods.get("put")
                .parameters
            ],
        },
        {
            "questions": [
                "I would like to see the price tables for event 1, with 10 results per page",
                "Now select one and put in draft mode",
            ],
            "target_context_calls": [
                "get_context(path='/event/{event_id}/price-table/', method='get')",
                "get_context(path='/event/{event_id}/price-table/{price_table_id}/draft/', method='post')",
            ],
            "parameters": [
                api_information.paths.get("/event/{event_id}/price-table/")
                .methods.get("get")
                .parameters,
                api_information.paths.get(
                    "/event/{event_id}/price-table/{price_table_id}/draft/"
                )
                .methods.get("post")
                .parameters,
            ],
        },
        {
            "questions": [
                "Can you update the product with ID 1 to have a title of 'New Product' and description of 'This is a great product'?"
            ],
            "target_context_calls": [
                "get_context(path='/product/{product_id}/', method='put')"
            ],
            "parameters": [
                api_information.paths.get("/product/{product_id}/")
                .methods.get("put")
                .parameters
            ],
        },
        {
            "questions": [
                "Get me the pricing for all products that are not archived with 50 prices per page"
            ],
            "target_context_calls": [
                "get_context(path='/product/{product_id}/price/', method='get')"
            ],
            "parameters": [
                api_information.paths.get("/product/{product_id}/price/")
                .methods.get("get")
                .parameters
            ],
        },
        {
            "questions": [
                "I have a product with ID 1, and performance with ID 12. Get me the product-performance-inventory.",
                "Now update the inventory with a value of 50",
            ],
            "target_context_calls": [
                "get_context(path='/product-performance-inventory/', method='get')",
                "get_context(path='/product-performance-inventory/', method='put')",
            ],
            "parameters": [
                api_information.paths.get("/product-performance-inventory/")
                .methods.get("get")
                .parameters,
                api_information.paths.get("/product-performance-inventory/")
                .methods.get("put")
                .parameters,
            ],
        },
        {
            "questions": [
                "I want to create a new Event called 'Spectacular Show' with a description of 'A show like no other'",
                "Now setup 10 performances for this event, starting from tomorrow at 10:00 AM, and ending the day after at 10:00 PM",
            ],
            "target_context_calls": [
                "get_context(path='/event/', method='post')",
                "get_context(path='/event/{event_id}/performance/', method='post')",
            ],
            "parameters": [
                api_information.paths.get("/event/").methods.get("post").parameters,
                api_information.paths.get("/event/{event_id}/performance/")
                .methods.get("post")
                .parameters,
            ],
        },
    ]

    return questions_to_metadata


async def eval_runner():
    console_clear = "\033[H\033[J"
    print(console_clear)
    stream_output("Starting Evaluation...")
    local_dir = "./models/"
    llm = await load_language_model(
        local_dir=local_dir,
        repo_name=repo_name,
        model_name=model_name,
        file_name=file_name,
    )
    # open directory and read the files
    input_file = "./evaluation/evaluation_data/API_docs.json"
    if not os.path.exists(input_file):
        stream_output("Critical: Evaluation file not found", Fore.RED)
        return
    api_information, full_schema_context = generate_schema_info(
        logger,
        whole_context,
        partial_schema_context,
        limited_schema_context,
        all_system_contexts,
        input_file,
    )

    questions_to_metadata = construct_evaluation_metadata(api_information)

    for question_metadata in questions_to_metadata:
        for context_type, system_context in all_system_contexts.items():
            history = [
                {"role": "system", "content": system_context},
            ]
            convo = copy.copy(question_metadata["questions"])
            question = convo.pop(0)
            logger.info("Question: {}, Context Type: {}".format(question, context_type))
            history.append({"role": "user", "content": question})
            answer = prompt_llm(history, use_llama, model_name, llm)
            context_stage_count = 0
            while True:
                new_message = answer
                history.append(new_message)
                if "get_context" in new_message["content"]:
                    context_stage_count += 1
                    schema_context = execute_context_call(
                        full_schema_context=full_schema_context,
                        context_call=new_message["content"],
                    )
                    if not schema_context:
                        message = {
                            "role": "assistant",
                            "content": "Couldn't find that path and method in the schema context. Are you sure it's that path and method? Could be a bulk operation or patch? Do not put params in the URL path during context calls.",
                        }
                    else:
                        context_stage = False
                        message = {
                            "role": "assistant",
                            "content": f"Here is the schema context, now use it to answer the question: {schema_context}",
                        }
                    history.append(message)
                else:
                    if len(convo) == 1:
                        history.append({"role": "user", "content": convo.pop(0)})
                    else:
                        save_response(
                            history, question_metadata, context_type, model_name
                        )
                        break

                gray_color = "\033[90m"
                reset_color = "\033[0m"
                print(f"{gray_color}\n{'-'*20} History dump {'-'*20}\n")
                print(json.dumps(history, indent=2))
                print(f"\n{'-'*55}\n{reset_color}")

                answer = prompt_llm(history, use_llama, model_name, llm)
                if answer.get("content") == "" or context_stage_count > 5:
                    save_response(history, question_metadata, context_type, model_name)
                    break