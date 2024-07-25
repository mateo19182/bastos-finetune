import json

def convert_to_chat_template(input_file, output_file):
    chat_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>  {SYSTEM} <|eot_id|><|start_header_id|>user<|end_header_id|>  {INPUT} <|eot_id|><|start_header_id|>assistant<|end_header_id|>  {OUTPUT} <|eot_id|>"""

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            messages = item['messages']
            system_content = next(msg['content'] for msg in messages if msg['role'] == 'system')
            user_content = next(msg['content'] for msg in messages if msg['role'] == 'user')
            assistant_content = next(msg['content'] for msg in messages if msg['role'] == 'assistant')

            formatted_chat = chat_template.format(
                SYSTEM=system_content,
                INPUT=user_content,
                OUTPUT=assistant_content
            )
            f.write(formatted_chat + '\n\n')

    print(f"Conversion complete. Output written to {output_file}")

# Usage
input_file = 'bastos.jsonl'
output_file = 'output.txt'
convert_to_chat_template(input_file, output_file)