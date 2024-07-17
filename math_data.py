import json

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
    for part in answer_parts:
        index = part.rfind('?')
        question_list.append(part[:index + 1].strip())
        answer_list.append(part[index + 1:].strip())

    # create new Q-A pair
    additional_part = 'Answer the question one by one.'
    new_qa_pair = {
        'question': context + ' ' + ' '.join(question_list).strip() + ' ' + last_question + ' ' + additional_part,
        'answer': ' '.join(answer_list).strip()
    }
    # print(new_qa_pair)
    return new_qa_pair

def process_json(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    processed_data = [process_qa_pair(json.loads(line)) for line in lines if process_qa_pair(json.loads(line)) is not None]
    # print(processed_data)
    with open(output_file, 'w', encoding='utf-8') as f:
        for qa_pair in processed_data:
            f.write(json.dumps(qa_pair, ensure_ascii=False) + '\n')


input_file = 'dataset/test_socratic.jsonl'
output_file = 'dataset/processed_test.json'

process_json(input_file, output_file)

