import asyncio
import json
import logging
import os
import random
import sys
import time
import uuid

import colorama
import llama_cpp
import requests
import structlog
from colorama import Fore
from llama_cpp import Llama

from data_generation.schema_processor import (
    APIInfo,
    _generate_path_to_response,
    parse_file,
)

logging.basicConfig(level=logging.CRITICAL)

logger = structlog.get_logger(__name__)

repo_name = "the1ullneversee"
model_name = "RestfulLlama-8B-Instruct"
colorama.init()


async def load_language_model(repo_name: str, model_name: str) -> Llama:
    def _load_model():
        return llama_cpp.Llama.from_pretrained(
            repo_id=f"{repo_name}/{model_name}",
            filename="restful_llama_8_Q8.gguf",
            n_threads=24,  # CPU cores
            n_batch=512,  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
            n_gpu_layers=-1,  # Change this value based on your model and your GPU VRAM pool.
            n_ctx=8096,  # Context window
            verbose=False,
        )

    # Run the CPU-bound model loading in a thread pool
    llm = await asyncio.to_thread(_load_model)
    return llm


def stream_output(output_text: str) -> None:
    sys.stdout.write(random.choice([Fore.CYAN, Fore.GREEN, Fore.MAGENTA, Fore.YELLOW]))
    for char in output_text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(0.02)
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
        logger.info(f"API Docs downloaded successfully to {tmp_file}")
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


def _api_context_summary(llm: Llama, api_context: APIInfo) -> str:
    system_message = "Here is some information about the API you are interacting with, please provide a summary of the API to confirm with the user that you understand the API"
    context = _generate_path_to_response(api_context)
    system_message += f"\n\n{context}"
    history = [{"role": "system", "content": system_message}]
    answer: dict = prompt_llm(llm, system_message, history)
    history.append(answer)
    return answer.get("content")


def _conversation_loop(llm: Llama, api_docs_json: dict) -> None:
    try:
        api_context = parse_file(api_docs_json)
        answer = _api_context_summary(llm, api_context)
        stream_output(answer)
        print(len(api_context.paths))
    except Exception as e:
        stream_output(f"Error: {e}")


async def main() -> None:
    model_task = asyncio.create_task(load_language_model(repo_name, model_name))

    welcome_text = (
        "Welcome to RestfulLlama!  "
        + "\n"
        + "RestfulLlama is a fine-tuned Large Language Model specifically designed to interacting with RESTFul based APIs!"
    )
    stream_output(welcome_text)

    # Wait for the model to load
    while not model_task.done():
        if further_text_blocks:
            stream_output(further_text_blocks.pop())
        else:
            stream_output("Loading the model...")
        await asyncio.sleep(2)
