import json
import re

def process_qa_pair(qa_pair):

    question = qa_pair['question']
    answer = qa_pair['answer']

    # find the question (last sentense) in question
    index = question.rfind('?')
    while (question[index] != '.'):
        index -= 1
        # the case that maybe we can just ignore
        if index == 0:
            return None
    # add 1 to avoid the "."
    last_question = question[index + 1:].strip()
    context = question[:index+1].strip()

    # get the question in answer
    answer_parts = answer.split('\n')
    question_list = []
    answer_list = []
    clean_answer_list = []
    for part in answer_parts:
        index = part.rfind('?')
        question_list.append(part[:index + 1].strip())
        answer_list.append(part[index + 1:].strip())
    # only left the result in answer
    for answer in answer_list:
        index = answer.rfind('>>')
        numbers = re.findall(r'\d+', answer[index+1:])
        if numbers:
            clean_answer_list.append(max(numbers, key=int))
        else: 
            index = answer.rfind('=')
            numbers = re.findall(r'\d+', answer[index+1:])
            if numbers:
                clean_answer_list.append(max(numbers, key=int))
            else:
                return None
    # create new Q-A pair
    additional_part = ' Answer these questions. Please mark the answer with ** so I can quickly find it.'
    new_qa_pair = {
        'question': context + ' ' + ' '.join(question_list).strip() + ' ' + additional_part,
        'answer': ' \n '.join(clean_answer_list[:-1]).strip()
    }
    # print(new_qa_pair)
    return new_qa_pair

def process_json(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    print(f'the number of prompts is:{len(lines)}')
    processed_data = [process_qa_pair(json.loads(line)) for line in lines if process_qa_pair(json.loads(line)) is not None]
    # print(processed_data)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)


input_file = 'dataset/test_socratic.jsonl'
output_file = 'dataset/processed_math_test.json'

process_json(input_file, output_file)

