import json

def process_qa_pair(qa_pair):
    question = qa_pair['question']
    answer = qa_pair['answer']
    index = answer.rfind('#')
    answer = answer[index + 2:]
    new_qa_pair = {
        'question': question,
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
    print(f'save to {output_file}')

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
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)
    print(f'save to {output_file}')


input_file = 'dataset/gsm_train_mc.jsonl'
# output_file_1 = 'dataset/blank/gsm_test.json'
output_file_2 = 'dataset/multiple_choice/gsm_train.json'

process_mc(input_file, output_file_2)

input_file = 'dataset/gsm_test_mc.jsonl'
# output_file_1 = 'dataset/blank/gsm_test.json'
output_file_2 = 'dataset/multiple_choice/gsm_test.json'

process_mc(input_file, output_file_2)


