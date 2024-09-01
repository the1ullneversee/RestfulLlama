
import os
import random
import sys
import time

from colorama import Fore
import asyncio
import os

from huggingface_hub import hf_hub_download
from llama_cpp import Llama

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
