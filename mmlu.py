import json

answer_map = {'A':0,'B':1,'C':2,'D':3}

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

def mmlu(input_file, output_file):
    # read json file
    with open(input_file, 'r') as file:
        data = json.load(file)

    # get the question, choice and answer
    my_dataset = []
    for key in data:
        data_list = data[key]
        for item in data_list:
            question = item[0]
            choice = [item[1],item[2],item[3],item[4]]
            answer = item[5]
            my_dataset.append({
                'question': question,
                'options': choice,
                'answer':answer
            })
    prompts = load_prompts_by_category("context.txt")
    category = "MMLU"
    # context = get_prompt_by_category(prompts, category)
    context = ""
    for data in my_dataset:
        data['context'] = context
    # save to local file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(my_dataset, f, ensure_ascii=False, indent=4)

    print(f'save to {output_file}')

input_file = 'dataset/mmlu_test.json'
output_file = 'dataset/multiple_choice/mmlu_test.json'
mmlu(input_file, output_file)

input_file = 'dataset/mmlu_train.json'
output_file = 'dataset/multiple_choice/mmlu_train.json'
mmlu(input_file, output_file)

