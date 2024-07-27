import re

def extract_user_prompts(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Regular expression to match user prompts
    pattern = r'<\|start_header_id\|>user<\|end_header_id\|>\s*(.*?)\s*<\|eot_id\|>'
    
    # Find all matches
    matches = re.findall(pattern, content, re.DOTALL)

    # Write matches to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        for match in matches:
            f.write(match.strip() + '\n')

if __name__ == "__main__":
    input_file = "bastos/bastos-llama.txt"  # Change this to your input file name
    output_file = "user_prompts.txt"  # Change this to your desired output file name
    
    extract_user_prompts(input_file, output_file)
    print(f"User prompts have been extracted and saved to {output_file}")