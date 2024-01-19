import os
import shutil
import json

base_file = 'swagger-files/rest_fine_tune.jsonl'


with open(base_file) as file:
    with open('rest_fine_tune_cleaned.jsonl', mode='w') as data_file:
        for file_line in file.readlines():
            line_json = json.loads(file_line)
            question: str = line_json['question']
            answer: str = line_json['answer']
            end_token = '[/INST]'
            answer = answer[answer.find(end_token) + len(end_token):]
            data_file.write(json.dumps({"question": question, "answer": answer})+"\n")


    # for line in contents:
    #     
    #     print(question)
    #     print(answer)
    #     break