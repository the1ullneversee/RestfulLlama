import json
from collections import defaultdict

import llama_cpp
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "NousResearch/Hermes-2-Theta-Llama-3-70B", trust_remote_code=True
)

model_path = "Meta-Llama-3-70B-Instruct.Q4_K_M.gguf"

llm = llama_cpp.Llama(
    model_path=model_path,
    n_threads=16,  # CPU cores
    n_batch=512,  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    n_gpu_layers=-1,  # Change this value based on your model and your GPU VRAM pool.
    n_ctx=8000,  # Context window
    verbose=True,
)

inference_params = {
    "do_sample": True,
    "top_p": 0.6,
    "temperature": 0.9,
    "top_k": 50,
    "max_new_tokens": 512,
    "repetition_penalty": 1.03,
    "stop": ["</s>"],
    "return_full_text": False,
}


input_file = "./data/0_output.json"
f = open(input_file, "r")
lines = f.readlines()
f.close()

output_file = "0_questions.jsonl"
previous_questions, start_index = get_questions(output_file=output_file)

line_index = start_index
try:
    paths = []
    multi_stage_questions = []
    single_stage_questions = []
    for line in lines[start_index:]:
        data = json.loads(line)
        full_schema_context = json.loads(data["full_schema_context"])
        path_context = json.loads(data["path_context"])
        path_parameters = {}
        response_parameters = defaultdict(list)
        request_params = defaultdict(list)
        index = 0
        first_context = ""
        second_context = ""
        third_context = ""
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
            index += 1
            paths.append(path)

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
        multi_stage_questions = previous_questions[line_index].get("questions")
        single_stage_questions = previous_questions[line_index].get(
            "single_stage_questions", []
        )
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
        line_index += 1
except Exception as exc:
    print(exc)
    dump_data(
        output_file=output_file,
        paths_hash=paths_hash,
        multi_stage_questions=[],
        single_stage_questions=[],
    )
    line_index += 1
