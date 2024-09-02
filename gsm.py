import json
import random

input_file = 'dataset/gsm_test.jsonl'
output_file = 'dataset/multiple_choice/gsm_test.json'

def process_qa_pair(qa_pair):
    question = qa_pair['question']
    answer = qa_pair['answer']
    index = answer.rfind('#')
    answer = answer[index + 2:]
    if random.random() > 0.5:
        option = [f'{answer}','None of the above is true']
        answer = 'A'
    else:
        option = ['None of the above is true',f'{answer}']
        answer = 'B'
    new_qa_pair = {
        'question': question,
        'options' : option,
        'answer' : answer
    }
    return new_qa_pair

def process_json(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    print(f'the number of prompts is:{len(lines)}')
    processed_data = [process_qa_pair(json.loads(line)) for line in lines if process_qa_pair(json.loads(line)) is not None]
    # print(processed_data)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)




process_json(input_file, output_file)

