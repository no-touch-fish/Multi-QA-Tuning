import json

def load_prompts_by_category(file_path):
    """
    Load prompts from a file and organize them into a dictionary by category.
    """
    prompts = {}
    current_category = None
    current_prompt = []
    
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line.startswith("#"):  # Category marker
                if current_category:  # Save the previous category
                    prompts[current_category] = "\n".join(current_prompt).strip()
                current_category = line[1:].strip()  # Get category name
                current_prompt = []
            elif line:  # Add non-empty lines to the current prompt
                current_prompt.append(line)
        if current_category:  # Save the last category
            prompts[current_category] = "\n".join(current_prompt).strip()
    
    return prompts

def get_prompt_by_category(prompts, category):
    """
    Get the prompt by its category name.
    """
    if category not in prompts:
        raise ValueError(f"Category '{category}' not found in the prompts.")
    return prompts[category]

def process_qa_pair(qa_pair):
    question = qa_pair['question']
    answer = qa_pair['answer']
    index = answer.rfind('#')
    answer = answer[index + 2:]
    new_qa_pair = {
        'question': question,
        'answer' : answer,
        'cot' : reduce_CoT(qa_pair['answer']).replace('\n',' ')
    }
    return new_qa_pair

def process_json(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    print(f'the number of prompts is:{len(lines)}')
    processed_data = [process_qa_pair(json.loads(line)) for line in lines if process_qa_pair(json.loads(line)) is not None]
    prompts = load_prompts_by_category("context.txt")
    category = "GSM"
    context = get_prompt_by_category(prompts, category)
    for data in processed_data:
        data["context"] = context
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)
    print(f'save to {output_file}')

def reduce_CoT(cot):
    output = []
    while '**' in cot:
        index_1 = cot.find('**')
        index_2 = cot.find('\n')
        output.append(cot[index_1+2:index_2])
        cot = cot[index_2+1:]
    output.append(" " + cot)
    return ("").join(output)

def process_mc_pair(qa_pair):
    question = qa_pair['Question']
    answer = qa_pair['Answer']
    options = [qa_pair['A'], qa_pair['B'], qa_pair['C'], qa_pair['D']]
    new_qa_pair = {
        'question': question,
        'options': options,
        'answer' : answer
    }
    return new_qa_pair

def process_mc(input_file,output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    print(f'the number of prompts is:{len(lines)}')
    processed_data = [process_mc_pair(json.loads(line)) for line in lines if process_mc_pair(json.loads(line)) is not None]
    context = ""
    for data in processed_data:
        data["context"] = context
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)
    print(f'save to {output_file}')


input_file = 'dataset/gsm_train.jsonl'
output_file_1 = 'dataset/blank/gsm_train.json'
process_json(input_file,output_file_1)

input_file = 'dataset/gsm_test.jsonl'
output_file_1 = 'dataset/blank/gsm_test.json'
process_json(input_file,output_file_1)

# input_file = 'dataset/gsm_train_mc.jsonl'
# output_file_2 = 'dataset/multiple_choice/gsm_train.json'
# process_mc(input_file, output_file_2)

# input_file = 'dataset/gsm_test_mc.jsonl'
# output_file_2 = 'dataset/multiple_choice/gsm_test.json'
# process_mc(input_file, output_file_2)


