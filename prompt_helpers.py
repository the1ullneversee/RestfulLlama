def get_instructions(user_content, system_content):
    instructions = [
        { "role": "system","content": f"{system_content} "},
    ]

    instructions.append({"role": "user", "content": f"{user_content}"})

    return instructions

def build_llama2_prompt(instructions):
    stop_token = "</s>"
    start_token = "<s>"
    startPrompt = f"{start_token}[INST] "
    endPrompt = " [/INST]"
    conversation = []
    for index, instruction in enumerate(instructions):
        if instruction["role"] == "system" and index == 0:
            conversation.append(f"<<SYS>>\n{instruction['content']}\n<</SYS>>\n\n")
        elif instruction["role"] == "user":
            conversation.append(instruction["content"].strip())
        else:
            conversation.append(f"{endPrompt} {instruction['content'].strip()} {stop_token}{startPrompt}")

    return startPrompt + "".join(conversation) + endPrompt