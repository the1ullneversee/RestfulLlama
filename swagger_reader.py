from swagger_parser import SwaggerParser
import os 
import datetime
import threading
import time
files = os.listdir('./broken_parsed')
import multiprocessing
worker_count = 24
workers = []
worker_start_times = {}


def file_reader(file):
    try:
        #print(f"{datetime.datetime.now().isoformat()} Processing {file}")
        parser = SwaggerParser(swagger_path=f'./parsed/{file}')
        #print(f"{datetime.datetime.now().isoformat()}_{parser.specification['info']['description']} has {len(parser.paths)} paths")
        os.rename(f'./broken_parsed/{file}', f'./data/parsed/{file}')
    except Exception as exc:
        #print(f"Error processing {file}: {exc}")
        os.rename(f'./broken_parsed/{file}', f'./data/broken_files/{file}')

if __name__ == '__main__':
    while len(files) > 0:
        if len(workers) <= worker_count:
            file = files.pop(0)
            worker = multiprocessing.Process(target=file_reader, args=(file,))
            #worker = threading.Thread(target=file_reader, args=(file,))
            workers.append(worker)
            worker.start()
            worker_start_times[worker] = datetime.datetime.now()
        else:
            for worker in workers:
                w_start_time = worker_start_times[worker]
                if w_start_time <= datetime.datetime.now() - datetime.timedelta(minutes=5):
                    print(f"Worker {worker} has been running since {w_start_time}, and needs killing")
                    worker.terminate()
                    print(f"Worker {worker} has been terminated")
                    workers.remove(worker)
                if not worker.is_alive():
                    #print(f"Worker {worker} has finished")
                    worker.join()
                    if worker in workers:
                        workers.remove(worker)
        time.sleep(1)
        # every minute print out how many files are left
        if datetime.datetime.now().second == 0:
            print(f"Files left: {len(files)}")