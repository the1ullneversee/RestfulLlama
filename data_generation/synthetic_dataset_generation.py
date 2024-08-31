import json
import math
import os
import subprocess
import tarfile
import zipfile

import llama_cpp
import requests
import structlog
from helper import (
    conversation_simulation,
    create_path_hash,
    dump_data,
    extract_resource_name,
    format_questions_response,
    get_questions,
    load_language_model,
    multi_stage_questioning,
    single_stage_questioning,
)
from schema_processor import (
    _generate_full_path_context,
    _generate_path_to_response,
    parse_file,
)
from transformers import AutoTokenizer

logger = structlog.get_logger()

inference_params = {
    "do_sample": True,
    "top_p": 0.6,
    "temperature": 0.1,
    "top_k": 50,
    "max_new_tokens": 512,
    "repetition_penalty": 1.03,
    "stop": ["</s>"],
    "return_full_text": False,
}


# input_file = "./data/0_output.json"
# f = open(input_file, "r")
# lines = f.readlines()
# f.close()

# output_file = "0_questions.jsonl"
# previous_questions, start_index = get_questions(output_file=output_file)

# line_index = start_index
# try:
#     paths = []
#     multi_stage_questions = []
#     single_stage_questions = []
#     for line in lines[start_index:]:
#         data = json.loads(line)
#         full_schema_context = json.loads(data["full_schema_context"])
#         path_context = json.loads(data["path_context"])
#         path_parameters = {}
#         response_parameters = defaultdict(list)
#         request_params = defaultdict(list)
#         index = 0
#         first_context = ""
#         second_context = ""
#         third_context = ""
#         for path, values in full_schema_context.items():
#             method = values.get("method")

#             resource = extract_resource_name(path)
#             if not values.get("parameters") and not values.get("request_body"):
#                 continue
#             full_line_item = f"{index}. [{{method: {values.get('method')}], path: {path}, summary: {values.get('summary')}, parameters: {values.get('parameters')}, request_body: {values.get('request_body') or []}, response_body: {values.get('response_body') or []}}}"
#             short_line_item = f"{index}. | {{'\method': \"{values.get('method')}\", \"path\": \"{path}\", \"response_body\": {values.get('response_body') or []}}}\n"
#             tiny_line_item = f'{index}. | {{"path": "{path}"}}\n'

#             second_context += short_line_item
#             first_context += full_line_item
#             third_context += tiny_line_item
#             index += 1
#             paths.append(path)

#         paths_hash = create_path_hash(paths=paths)
#         multi_part_llm_response = multi_stage_questioning(first_context, second_context)
#         single_part_llm_response = single_stage_questioning(first_context)
#         multi_stage_questions = format_questions_response(
#             input_str=multi_part_llm_response,
#             output_file=output_file,
#             paths_hash=paths_hash,
#         )
#         single_stage_questions = format_questions_response(
#             input_str=single_part_llm_response,
#             output_file=output_file,
#             paths_hash=paths_hash,
#         )
#         multi_stage_questions = previous_questions[line_index].get("questions")
#         single_stage_questions = previous_questions[line_index].get(
#             "single_stage_questions", []
#         )
#         conversation_simulation(
#             llm=llm,
#             multi_stage_questions=multi_stage_questions,
#             single_stage_questions=single_stage_questions,
#             short_path_context=third_context,
#             full_path_context=full_schema_context,
#         )

#         dump_data(
#             output_file=output_file,
#             paths_hash=paths_hash,
#             multi_stage_questions=multi_stage_questions,
#             single_stage_questions=single_stage_questions,
#         )
#         line_index += 1
# except Exception as exc:
#     print(exc)
#     dump_data(
#         output_file=output_file,
#         paths_hash=paths_hash,
#         multi_stage_questions=[],
#         single_stage_questions=[],
#     )
#     line_index += 1


def _conversational_generation(
    full_schema_context: dict, llm: llama_cpp.Llama, output_file: str
):
    try:
        paths = []
        multi_stage_questions = []
        single_stage_questions = []
        first_context = ""
        second_context = ""
        third_context = ""
        index = 0
        for path, values in full_schema_context.items():
            method = values.get("method")
            resource = extract_resource_name(path)
            if not values.get("parameters") and not values.get("request_body"):
                continue
            full_line_item = f"{index}. [{{method: {values.get('method')}], path: {path}, summary: {values.get('summary')}, parameters: {values.get('parameters')}, request_body: {values.get('request_body') or []}, response_body: {values.get('response_body') or []}}}"
            short_line_item = f"{index}. | {{'\method': \"{values.get('method')}\", \"path\": \"{path}\", \"response_body\": {values.get('response_body') or []}}}\n"
            tiny_line_item = f'{index}. | {{"path": "{path}"}}\n'

            second_context += short_line_item
            first_context += full_line_item
            third_context += tiny_line_item
            paths.append(path)
            index += 1

        paths_hash = create_path_hash(paths=paths)
        multi_part_llm_response = multi_stage_questioning(first_context, second_context)
        single_part_llm_response = single_stage_questioning(first_context)
        multi_stage_questions = format_questions_response(
            input_str=multi_part_llm_response,
            output_file=output_file,
            paths_hash=paths_hash,
        )
        single_stage_questions = format_questions_response(
            input_str=single_part_llm_response,
            output_file=output_file,
            paths_hash=paths_hash,
        )
        # multi_stage_questions = previous_questions[line_index].get("questions")
        # single_stage_questions = previous_questions[line_index].get(
        #     "single_stage_questions", []
        # )
        conversation_simulation(
            llm=llm,
            multi_stage_questions=multi_stage_questions,
            single_stage_questions=single_stage_questions,
            short_path_context=third_context,
            full_path_context=full_schema_context,
        )

        dump_data(
            output_file=output_file,
            paths_hash=paths_hash,
            multi_stage_questions=multi_stage_questions,
            single_stage_questions=single_stage_questions,
        )
    except Exception as exc:
        print(exc)
        dump_data(
            output_file=output_file,
            paths_hash=paths_hash,
            multi_stage_questions=[],
            single_stage_questions=[],
        )


def pull_and_process_files(git_repo_path, input_folder):
    # Change to the git repository directory
    zip_prefix = "grouped_files"
    # Pull files using Git LFS
    try:
        subprocess.run(["git", "lfs", "pull"], check=True)
        print("Git LFS pull completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error during Git LFS pull: {e}")
        return

    os.makedirs(input_folder, exist_ok=True)
    # Counter for extracted files
    total_files = 0

    # Loop through the expected zip files
    for i in range(1, 5):  # Assuming 4 groups as before
        zip_filename = f"{git_repo_path}{zip_prefix}_group_{i}.zip"

        # Check if the zip file exists
        if not os.path.exists(zip_filename):
            print(f"Warning: {zip_filename} not found. Skipping.")
            continue

        # Extract files from the zip
        with zipfile.ZipFile(zip_filename, "r") as zipf:
            zipf.extractall(input_folder)
            total_files += len(zipf.namelist())

        print(f"Extracted files from {zip_filename}")

    print(f"Extraction complete. Total files extracted: {total_files}")


def preflight_check(
    parent_path: str,
    data_folder: str,
    input_folder: str,
    broken_files_folder: str,
    data_set_parsed_folder: str,
    raw_data_set_url: str,
):
    if not os.path.exists(data_folder):
        os.mkdir(data_folder)

    if not os.path.exists(input_folder):
        pull_and_process_files(git_repo_path=parent_path, input_folder=input_folder)

    if not os.path.exists(broken_files_folder):
        os.mkdir(broken_files_folder)
    if not os.path.exists(data_set_parsed_folder):
        os.mkdir(data_set_parsed_folder)


def _move_file(folder: str, file: str, destination: str):
    os.rename(folder + file, f"{destination}{file}")


def process_file(
    file: str,
    input_folder: str,
    output_file: str,
    broken_files_folder: str,
    data_set_parsed_folder: str,
    llm: llama_cpp.Llama,
):
    processed_file = False
    try:
        f = open(input_folder + "/" + file)
        data = json.load(f)
        api_information = parse_file(data)
        if not api_information:
            f.close()
            _move_file(
                folder=input_folder,
                file=file,
                destination=broken_files_folder,
            )
            return processed_file
        path_context = _generate_path_to_response(api_information)
        full_schema_context = _generate_full_path_context(api_information)
        if not path_context or not full_schema_context:
            f.close()
            _move_file(
                folder=input_folder,
                file=file,
                destination=broken_files_folder,
            )
            return processed_file

        values = {
            "path_context": path_context,
            "full_schema_context": full_schema_context,
        }
        _conversational_generation(
            full_schema_context, llm=llm, output_file=output_file
        )

        # _dump_to_file(output_file=output_file, values=values)
        f.close()
        _move_file(
            folder=input_folder,
            file=file,
            destination=data_set_parsed_folder,
        )
        api_information = None
        processed_file = True
        return processed_file
    except Exception as e:
        # logger.debug(f"Error: {e}")
        f.close()
        _move_file(
            folder=input_folder,
            file=file,
            destination=broken_files_folder,
        )
    return processed_file


async def generation_main():
    parent_folder = "./data_generation/"
    data_folder = parent_folder + "data/"
    input_folder = parent_folder + "input/"
    broken_files_folder = parent_folder + "broken_files/"
    data_set_parsed_folder = parent_folder + "processed_files/"
    output_file = data_folder + "/data_set.json"
    raw_data_set_url = (
        "https://wordlists-cdn.assetnote.io/rawdata/kiterunner/swagger-files.tar"
    )
    # tokenizer = AutoTokenizer.from_pretrained(
    #     "NousResearch/Hermes-2-Theta-Llama-3-70B", trust_remote_code=True
    # )

    file_name = "Meta-Llama-3-70B-Instruct-Q4_K_M.gguf"
    local_dir = "../models/"
    repo_name = "bartowski"
    model_name = "Meta-Llama-3-70B-Instruct-GGUF"

    preflight_check(
        parent_path=parent_folder,
        data_folder=data_folder,
        input_folder=input_folder,
        broken_files_folder=broken_files_folder,
        data_set_parsed_folder=data_set_parsed_folder,
        raw_data_set_url=raw_data_set_url,
    )
    llm = await load_language_model(
        local_dir=local_dir,
        repo_name=repo_name,
        model_name=model_name,
        file_name=file_name,
    )

    print("Starting data generation...")
    files = os.listdir(input_folder)
    print(f"Found {len(files)} files to process.")
    for file in files:
        process_file(
            file=file,
            input_folder=input_folder,
            output_file=output_file,
            broken_files_folder=broken_files_folder,
            data_set_parsed_folder=data_set_parsed_folder,
            llm=llm,
        )
        print(f"Processed {file} \n")


if __name__ == "__main__":
    import asyncio

    asyncio.run(generation_main())
