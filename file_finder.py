import os
import json

train_data_set = {}
with open('rest_fine_tune_cleaned.jsonl', mode='r') as cleaned:
    for line in cleaned.readlines():
        question_answer = json.loads(line)
        train_data_set[question_answer['question']] = question_answer['answer']

test_data_set = {}
with open('rest_fine_tune (1).jsonl', mode='r') as raw:
    for line in raw.readlines():
        question_answer = json.loads(line)
        if question_answer['question'] in train_data_set:
            print("Duplicate found " + question_answer['question'])
            del train_data_set[question_answer['question']]
        test_data_set[question_answer['question']] = question_answer['answer']

print(f"train {len(train_data_set)} test {len(test_data_set)}")
print(f"whole = {len(train_data_set) + len(test_data_set)}")


with open('test_dataset.jsonl', 'w', newline='\n') as f:
    for question, answer in test_data_set.items():
        f.write(json.dumps({"question": question, "answer": answer})+"\n")
