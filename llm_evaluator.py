import json
import sys
import sys
from openai import OpenAI

import json

from schema_processor import _generate_full_path_context, _generate_path_to_response, execute_context_call, parse_file

def save_response(history, questions_to_metadata):
    print("Saving response")

if __name__ == '__main__':
    # open directory and read the files
    input_file = 'API_docs.json'
    with open(input_file) as f:
        try:
            data = json.load(f)
            api_information = parse_file(data)
            if not api_information:
                sys.exit(0)
            #api_information= process_paths(api_information=api_information)
            path_context = _generate_path_to_response(api_information)
            full_schema_context = _generate_full_path_context(api_information)

            system_context = f"""You are a helpful assistant that can generate Python code to interact with an application's API. You have access to the application's API endpoints and their corresponding schemas.
                When a user asks a question, your task is to ask for additional context about the API endpoints as needed, reason about the schema context, and how it can answer the question or part of it.
                You can ask for context about an API endpoint by saying get_context(path='path', method='method'), where path is the endpoint path and method is the verb.
                Look at the schema context to understand what PARAMETERS are required to call each endpoint.
                The application's API endpoints are: {path_context}
                Please respond with a get_context request to clarify the API endpoint needed to answer the user's question.
                If you need further context on multi-part questions, you can ask for it by saying get_context(path='path', method='method').
                """
        except Exception as e:
            print('Error: Invalid JSON file')

        # Chat with an intelligent assistant in your terminal
    
    questions_to_metadata = {}
    questions_to_metadata = [
        {"questions": ["Can you get me a list of events?"], "target_context_calls": ["get_context(path='/event/', method='get')"], "parameters": [api_information.paths.get('/event/').methods.get('get').parameters]},
        {"questions": ["Get a list of products that are archived"], "target_context_calls": ["get_context(path='/product/', method='get')"], "parameters": [api_information.paths.get('/product/').methods.get('get').parameters]},
        {"questions": ["Can you get me all performances for an event that start after tomorrow?"], "target_context_calls": ["get_context(path='/event/{event_id}/performance/', method='get')"], "parameters": [api_information.paths.get('/event/{event_id}/performance/').methods.get('get').parameters]},
        {"questions": ["I have a discount wiht ID 5, and I want to get the visbility for the discount.", "Now I want to update the visibility to start from tomorrow and end in a months time."], "target_context_calls": ["get_context(path='/discount/{discount_id}/visibility/', method='get')", "get_context(path='/discount/{discount_id}/visibility/', method='put')"], "parameters": [api_information.paths.get('/discount/{discount_id}/visibility/').methods.get('get').parameters, api_information.paths.get('/discount/{discount_id}/visibility/').methods.get('put').parameters]},
        {"questions": ["Can you change all performances that start after tomorrow to start at 10:00 AM for Event 1?"], "target_context_calls": ["get_context(path='/event/{event_id}/performance/bulk/', method='put')"], "parameters": [api_information.paths.get('/event/{event_id}/performance/bulk/').methods.get('put').parameters]},
        {"questions": ["I would like to see the price tables for event 1, with 10 results per page", "Now select one and put in draft mode"], "target_context_calls": ["get_context(path='/event/{event_id}/price-table/', method='get')", "get_context(path='/event/{event_id}/price-table/{price_table_id}/draft/', method='post')"], "parameters": [api_information.paths.get('/event/{event_id}/price-table/').methods.get('get').parameters, api_information.paths.get('/event/{event_id}/price-table/{price_table_id}/draft/').methods.get('post').parameters]},
        {"questions": ["Can you update the product with ID 1 to have a title of 'New Product' and description of 'This is a great product'?"], "target_context_calls": ["get_context(path='/product/{product_id}/', method='put')"], "parameters": [api_information.paths.get('/product/{product_id}/').methods.get('put').parameters]},
        {"questions": ["Get me the pricing for all products that are not archived with 50 prices per page"], "target_context_calls": ["get_context(path='/product/{product_id}/price/', method='get')"], "parameters": [api_information.paths.get('/product/{product_id}/price/').methods.get('get').parameters]},
        {"questions": ["I have a product with ID 1, and performance with ID 12. Get me the product-performance-inventory.", "Now update the inventory with a value of 50"], "target_context_calls": ["get_context(path='/product-performance-inventory/', method='get')", "get_context(path='/product-performance-inventory/', method='put')"], "parameters": [api_information.paths.get('/product-performance-inventory/').methods.get('get').parameters, api_information.paths.get('/product-performance-inventory/').methods.get('put').parameters]},
        {"questions": ["I want to create a new Event called 'Spectacular Show' with a description of 'A show like no other'", "Now setup 10 performances for this event, starting from tomorrow at 10:00 AM, and ending the day after at 10:00 PM"], "target_context_calls": ["get_context(path='/event/', method='post')", "get_context(path='/event/{event_id}/performance/', method='post')"], "parameters": [api_information.paths.get('/event/').methods.get('post').parameters, api_information.paths.get('/event/{event_id}/performance/').methods.get('post').parameters]},
    ]
    # Point to the local server
    client = OpenAI(base_url="http://172.29.0.1:1234/v1", api_key="lm-studio")
    for question_metadata in questions_to_metadata:
        history = [
            {"role": "system", "content": system_context},
        ]
        covno = question_metadata["questions"]
        question = covno[0]
        history.append({"role": "user", "content": question})
        completion = client.chat.completions.create(
                model="the1ullneversee/Restful-Llama-Instruct-8b-gguf",
                messages=history,
                temperature=0.7,
                stream=True,
            )
        while True:
            new_message = {"role": "assistant", "content": ""}
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    print(chunk.choices[0].delta.content, end="", flush=True)
                    new_message["content"] += chunk.choices[0].delta.content

            history.append(new_message)
            
            # Uncomment to see chat history
            
            gray_color = "\033[90m"
            reset_color = "\033[0m"
            print(f"{gray_color}\n{'-'*20} History dump {'-'*20}\n")
            print(json.dumps(history, indent=2))
            print(f"\n{'-'*55}\n{reset_color}")

            print()
            if 'get_context' in new_message['content']:
                #context_path = extract_context(llm_response=new_message["content"])
                schema_context = execute_context_call(full_schema_context=full_schema_context, context_call=new_message["content"])
                if not schema_context:
                    message = {"role": "assistant", "content": "Couldn't find that path and method in the schema context. Are you sure it's that path and method? Could be a bulk operation or patch? Do not put params in the URL path during context calls."}
                else:
                    message = {"role": "assistant", "content": f"Here is the schema context, now use it to answer the question: {schema_context}"}
                history.append(message)
            else:
                if len(covno) == 2:
                    history.append({"role": "user", "content": covno[1]})
                else:
                    save_response(history, questions_to_metadata)
                    break
            
            completion = client.chat.completions.create(
                model="the1ullneversee/Restful-Llama-Instruct-8b-gguf",
                messages=history,
                temperature=0.7,
                stream=True,
            )
    