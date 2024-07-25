import json

def modify_jsonl(input_file, output_file):
    chat_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{SYSTEM} <|eot_id|>{CONVERSATION}"""

    conversation_template = """<|start_header_id|>{ROLE}<|end_header_id|>

{CONTENT} <|eot_id|>"""

    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            try:
                data = json.loads(line.strip())
                messages = data['messages']
                
                system_content = messages[0]['content']
                conversation = ""
                
                for message in messages[1:]:
                    role = message['role']
                    content = message['content']
                    conversation += conversation_template.format(ROLE=role, CONTENT=content)
                
                modified_content = chat_template.format(
                    SYSTEM=system_content,
                    CONVERSATION=conversation
                )
                
                outfile.write(modified_content + '\n')
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                print(f"Problematic line: {line}")
                continue

if __name__ == "__main__":
    input_file = "data/bastos.jsonl"
    output_file = "modified_bastos.jsonl"
    modify_jsonl(input_file, output_file)
    print(f"Modified content has been written to {output_file}")