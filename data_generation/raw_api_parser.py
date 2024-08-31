import json
import os
import tarfile
import threading as th
from queue import Queue

import requests
import structlog
from schema_processor import (
    _generate_full_path_context,
    _generate_path_to_response,
    parse_file,
)

logger = structlog.get_logger(__name__)


def log_to_console(file_count, valid_files):
    # clear console
    print("\033[H\033[J")
    print(f"Files remaining: {file_count}, Valid files: {valid_files}", end="\r")


def _dump_to_file(output_file: str, values: dict):
    with open(output_file, "a") as f:
        f.write(json.dumps(values) + "\n")


def _move_file(folder: str, file: str, destination: str):
    os.rename(folder + file, f"{destination}{file}")


def download_data_set(url, save_path, file_name):
    logger.info(f"Downloading {url} to {save_path + file_name}")
    response = requests.get(url, stream=True, timeout=10)
    with open(save_path + file_name, "wb") as f:
        for chunk in response.iter_content(chunk_size=128):
            f.write(chunk)


def untar_file(file_path: str, destination: str):
    try:
        with tarfile.open(file_path) as tar:
            tar.extractall(path=destination)
        print(f"Extracted {file_path} to {destination}")
    except Exception as e:
        print(f"Error extracting {file_path}: {e}")


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
        file_name = "openapi-data.tar"
        download_data_set(raw_data_set_url, parent_path, file_name)
        untar_file(file_name, parent_path)

    if not os.path.exists(broken_files_folder):
        os.mkdir(broken_files_folder)
    if not os.path.exists(data_set_parsed_folder):
        os.mkdir(data_set_parsed_folder)


def process_file(file: str, input_folder: str, output_file: str):
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
        _conversational_generation()

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


def worker(queue, input_folder, output_file):
    while True:
        file = queue.get()
        if file is None:
            break
        processed = process_file(file, input_folder, output_file)
        # if processed:
        #     with valid_files.get_lock():
        #         valid_files.value += 1
        # with files_remaining.get_lock():
        #     files_remaining.value -= 1
        print(
            f"Processed {file} status {processed} \n",
        )
        queue.task_done()


if __name__ == "__main__":
    # folders to store the data
    parent_folder = "./data_generation/"
    data_folder = parent_folder + "data/"
    input_folder = parent_folder + "input/"
    broken_files_folder = parent_folder + "broken_files/"
    data_set_parsed_folder = parent_folder + "processed_files/"
    output_file = data_folder + "/data_set.json"
    raw_data_set_url = (
        "https://wordlists-cdn.assetnote.io/rawdata/kiterunner/swagger-files.tar"
    )

    preflight_check(
        parent_path=parent_folder,
        data_folder=data_folder,
        input_folder=input_folder,
        broken_files_folder=broken_files_folder,
        data_set_parsed_folder=data_set_parsed_folder,
        raw_data_set_url=raw_data_set_url,
    )

    files = os.listdir(input_folder)
    # valid_files = 0
    # total_files = len(files)
    # for file in files:
    #     processed = process_file(file, input_folder, output_file)
    #     total_files -= 1
    #     if processed:
    #         valid_files += 1
    #     log_to_console(total_files, valid_files)

    total_files = len(files)
    output_index = 0
    worker_count = 10
    workers = []
    queue = Queue()

    for _ in range(worker_count):
        t = th.Thread(
            target=worker,
            args=(queue, input_folder, output_file),
        )
        t.start()
        workers.append(t)

    for file in files:
        queue.put(file)

    for _ in range(worker_count):
        # Add a None for each worker to signal them to stop
        queue.put(None)

    queue.join()
    for t in workers:
        t.join()

    print("All files processed")
