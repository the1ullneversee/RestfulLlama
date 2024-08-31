import asyncio
import json
import logging
import os
import random
import sys
import time
import uuid

import colorama
import requests
from colorama import Fore
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

logging.basicConfig(level=logging.CRITICAL)
logger = logging.getLogger(__name__)

from data_generation.schema_processor import (
    APIInfo,
    _generate_full_path_context,
    _generate_path_to_response,
    execute_context_call,
    parse_file,
)

logging.basicConfig(level=logging.CRITICAL)

repo_name = "the1ullneversee"
model_name = "RestfulLlama-8B-Instruct"
file_name = "restful_llama_8_Q8.gguf"
colorama.init()

max_seq_length = 2048  # Supports RoPE Scaling interally, so choose any!
llm_ctx_window = 8096
# 10% less than the max context window
llm_ctx_window_warning = llm_ctx_window - (llm_ctx_window * 0.1)


def download_gguf_model(local_dir="./models"):
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


async def load_language_model() -> None:
    def _load_model():
        model_path = download_gguf_model()
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


def stream_output(
    output_text: str, colour: str | None = None, output_cadence: float = 0.02
) -> None:
    if colour:
        sys.stdout.write(colour)
    else:
        sys.stdout.write(
            random.choice([Fore.CYAN, Fore.GREEN, Fore.MAGENTA, Fore.YELLOW])
        )
    for char in output_text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(output_cadence)
    print()


def download_docs(url: str) -> str:
    if not os.path.exists("./temp"):
        os.makedirs("./temp")
    try:
        response = requests.get(url)
        # Raise an exception for bad status codes
        response.raise_for_status()
        # Save the JSON data to a file
        tmp_file = "./temp/" + str(uuid.uuid4()) + ".json"
        with open(tmp_file, "w", encoding="utf-8") as f:
            f.write(response.text)
        stream_output(f"API Docs downloaded successfully to {tmp_file}", Fore.CYAN)
        return tmp_file
    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading API Docs: {e}")


further_text_blocks = [
    "RestfulLlama is a fine-tuned Large Language Model specifically designed to interacting with RESTFul based APIs!",
    "RestfulLlama is based on the LLama 3 7B Instruct Model.",
    "RestfulLlama was trained on a dataset of synthetic conversations between users and RESTFul APIs.",
    "Llama 7 70B Instruct was used as the base model for synthetic dataset generation.",
]


def prompt_llm(llm: Llama, content: str, history: list[dict]) -> dict:
    history.append({"role": "system", "content": content})
    answer = llm.create_chat_completion(messages=history)
    return answer.get("choices")[0].get("message")


def count_tokens(llm: Llama, prompt: str) -> int:
    # Tokenize the prompt
    tokens = llm.tokenize(prompt.encode("utf-8"))
    # Return the number of tokens
    return len(tokens)


def _api_context_summary(llm: Llama, api_context: APIInfo) -> None:

    stream_output(output_text="Analyzing the API docs...", colour=Fore.CYAN)
    system_message = "Here is some information about the API you are interacting with, please provide a very concise summary of the API. Product your output in MARKDOWN or HTML format."
    context = _generate_path_to_response(api_context)
    system_message += f"\n\n{context}"
    token_count = count_tokens(llm, system_message)
    if token_count > llm_ctx_window:
        stream_output(
            output_text="Error: The context window is full, the model will not be able to process the input. Please try a smaller API docs file.",
            colour=Fore.RED,
        )
        sys.exit(1)
    elif token_count >= llm_ctx_window_warning:
        stream_output(
            output_text="Warning: The context window will be almost full even with advanced compression of your API docs, performance will be degraded.",
            colour=Fore.RED,
        )
    history = [{"role": "system", "content": system_message}]
    answer: dict = prompt_llm(llm, system_message, history)
    answer.get("content").replace(
        "Here is a concise summary of the API in MARKDOWN format:", ""
    )
    history.append(answer)
    stream_output(answer.get("content"), Fore.CYAN, output_cadence=0.01)


def _load_docs(api_docs_file: str) -> dict:
    with open(api_docs_file, "r") as f:
        api_docs = json.load(f)
    return api_docs


def _conversation_loop(llm: Llama, api_docs_file: str) -> None:
    try:
        api_docs = _load_docs(api_docs_file)
        api_context = parse_file(api_docs)
        _api_context_summary(llm, api_context)
        path_context = _generate_path_to_response(api_context)
        full_schema_context = _generate_full_path_context(api_context)
        whole_context = f"""You are a helpful assistant that can generate Python code to interact with an application's API. You have access to the application's API endpoints and their corresponding schemas.
                When a user asks a question, your task is to ask for additional context about the API endpoints as needed, reason about the schema context, and how it can answer the question or part of it.
                You can ask for context about an API endpoint by making a call to get_context, like so: get_context(path='/path/', method='method').
                Look at the schema context to understand what PARAMETERS are required to call each endpoint.
                The application's API endpoints are: {path_context}
                Please respond with a get_context request to clarify the API endpoint needed to answer the user's question.
                Remember to use the schema context to generate the Python code. And to ask for context about an API endpoint, use the get_context function call.
                When generating code, please put the code in a <CODE> block so that it can be detected.
        """
        history = [{"role": "system", "content": whole_context}]
        stream_output(
            "If you are unsure on what you can do, why not ask the model? :)", Fore.CYAN
        )
        while True:
            stream_output(
                "Ready for input, press enter to exit, or ask away :)", Fore.YELLOW
            )
            question = input()
            if not question:
                break
            history.append({"role": "user", "content": question})
            answer = prompt_llm(llm, question, history)
            new_message = answer
            history.append(new_message)
            if "get_context" in new_message["content"]:
                schema_context = execute_context_call(
                    full_schema_context=full_schema_context,
                    context_call=new_message["content"],
                )
                if not schema_context:
                    message = {
                        "role": "assistant",
                        "content": "Couldn't find that path and method in the schema context. Are you sure it's that path and method? Could be a bulk operation or patch? Do not put params in the URL path during context calls.",
                    }
                    history.append(message)
                    response = prompt_llm(llm, message["content"], history)
                    history.append(response)
                else:
                    message = {
                        "role": "assistant",
                        "content": f"Here is the schema context, now use it to answer the question: {schema_context}",
                    }
                    history.append(message)
                    response = prompt_llm(llm, message["content"], history)
                    history.append(response)
                    stream_output(response["content"], Fore.CYAN)
            else:
                stream_output(new_message["content"], Fore.CYAN)
    except Exception as e:
        stream_output(f"Error: {e}")


async def inference() -> None:
    console_clear = "\033[H\033[J"
    print(console_clear)
    model_task = asyncio.create_task(load_language_model())

    welcome_text = (
        "Welcome to RestfulLlama Inference!"
        + "\n"
        + "RestfulLlama is a fine-tuned Large Language Model specifically designed to interacting with RESTFul based APIs!"
    )
    stream_output(welcome_text)

    # Wait for the model to load
    stream_output("Loading the model...")
    while not model_task.done():
        await asyncio.sleep(5)
        # clear console
        print("\033[H\033[J")
        print("Loading the model...")

    llm = model_task.result()
    stream_output("Model loaded successfully!")
    default_api = "https://petstore.swagger.io/v2/swagger.json"
    response = input(
        f"Enter the URL of the API Docs you would like to interact with or press enter to chose the default, (default is {default_api}): "
    )
    if not response:
        user_api = default_api
    api_docs_file = download_docs(user_api)
    _conversation_loop(llm, api_docs_file)


if __name__ == "__main__":
    asyncio.run(inference())
