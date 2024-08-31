import asyncio
import random
import sys
import time

import torch
from colorama import Fore

from model_converser import inference


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


def get_gpu_info():
    gpu_information = {}
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"Number of available GPUs: {gpu_count}")
        for i in range(gpu_count):
            gpu = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}:")
            print(f"  Name: {gpu.name}")
            print(f"  VRAM: {gpu.total_memory / 1024**3:.2f} GB")
            gpu_information[i] = gpu.total_memory / 1024**3

    return gpu_information


def pre_flight_checks() -> None:
    stream_output("Running pre-flight checks...", Fore.YELLOW)
    gpu_information = get_gpu_info()
    checklist = [
        {"label": "Dataset Generation", "vram": 40, "type": "Multi-GPU"},
        {"label": "Model Fine-Tuning", "vram": 40, "type": "Single-GPU"},
        {"label": "Evaluation", "vram": 24, "type": "Single-GPU"},
        {"label": "CLI Inference", "vram": 24, "type": "Single-GPU"},
    ]
    # Check for Dataset Generation, GPU is LLama-3-70B model which needs 40GB VRAM or 2 * 24GB VRAM
    valid_config = f"✅"
    invalid_config = f"❌"
    check_to_print = []
    for check in checklist:
        label = check.get("label")
        vram_requirement = check.get("vram")
        gpu_config = check.get("type")
        valid_check = False
        if gpu_information:
            multi_gpu = len(gpu_information) > 1
            if gpu_config == "Multi-GPU":
                combined_vram = round(sum(gpu_information.values()))
                if vram_requirement > combined_vram:
                    stream_output(
                        f"{label} requires {vram_requirement}GB VRAM, which is more than the {combined_vram}GB VRAM available in the multi-GPU configuration.",
                        Fore.RED,
                    )
                else:
                    valid_check = True
            else:
                # single gpu available, so make sure that single GPU has enough VRAM
                available_vram = round(min(gpu_information.values()))
                if available_vram < vram_requirement:
                    stream_output(
                        f"{label} requires {vram_requirement}GB VRAM, which is more than the {available_vram}GB VRAM available in the single GPU configuration.",
                        Fore.RED,
                    )
                else:
                    valid_check = True
        check_to_print.append(
            f"{label}: {valid_config if valid_check else invalid_config}"
        )
    stream_output("Pre-flight checks completed successfully!", Fore.GREEN)
    for check in check_to_print:
        print(check)
    return check_to_print


async def main() -> None:
    welcome_text = """Welcome to the RestfulLlama Project! \n This project has two sides: \n
        First is the Machine Learning Pipeline, taking you from raw API documentation to a trained model. \n
        Second is the interactive CLI, where you can ask the model questions about the API documentation. \n
        """
    stream_output(welcome_text)
    pre_flight_checks()

    stream_output("Please select an option from the following:", Fore.CYAN)
    options = [
        "1. Generate Synthetic Dataset",
        "2. Fine-tune Model",
        "3. Evaluate Model",
        "4. CLI Inference",
        "5. Exit",
    ]
    for option in options:
        print(option)
    selected_option = input("Enter the option number: ")

    match selected_option:
        case "1":
            stream_output("Generating Synthetic Dataset...", Fore.CYAN)
            # await generate_synthetic_dataset()
        case "2":
            stream_output("Fine-tuning Model...", Fore.CYAN)
            # await fine_tune_model()
        case "3":
            stream_output("Evaluating Model...", Fore.CYAN)
            # await evaluate_model()
        case "4":
            stream_output("Starting CLI Inference...", Fore.CYAN)
            await inference()
        case "5":
            stream_output("Exiting...", Fore.CYAN)
            sys.exit(0)
        case _:
            stream_output("Invalid option selected. Please try again.", Fore.RED)


if __name__ == "__main__":
    # asyncio.run(load_sloth_model())
    asyncio.run(main())
