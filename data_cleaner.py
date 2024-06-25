import hashlib
import json

def create_path_hash(paths: list):
    concat = ''.join(paths)

    hash_object = hashlib.sha256(concat.encode())
    hash_hex = hash_object.hexdigest()
    return hash_hex



input_file = './data/0_output.json'
f = open(input_file, "r")
lines = f.readlines()
f.close()

input_file = 'question.jsonl'
f = open(input_file, 'r')
output_lines = f.readlines()
f.close()

output_file = "0_questions.jsonl"
with open(output_file, 'w') as f_o:
    index = 0
    for line in lines:
        data = json.loads(line)
        output_data = json.loads(output_lines[index])
        full_schema_context = json.loads(data["full_schema_context"])
        paths = []
        for path, values in full_schema_context.items():
            paths.append(path)
        paths_hash = create_path_hash(paths=paths)
        f_o.write(json.dumps({"hash": paths_hash, "questions": output_data['questions']}) + '\n')
        index += 1