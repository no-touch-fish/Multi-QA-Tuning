from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import argparse
import json
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description="""Compare the result and the answer."""
)

parser.add_argument(
    "--data_path",
    type = str,
    default = 'result/gsm.json',
    help = "Path to the dataset used.",
)
parser.add_argument(
    "--gpu",
    type = str,
    default = '0',
    help = "which gpu to use",
)
parser.add_argument(
    "--case",
    type = str,
    default = None,
    help = "which case of result we are comparing",
)

args = parser.parse_args()
data_path = args.data_path
gpu = args.gpu
# case = args.case
# os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu}"
device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

# 加载预训练的问答模型和句子嵌入模型
qa_model_name = "deepset/roberta-base-squad2"
qa_pipeline = pipeline("question-answering", model=qa_model_name,device = int(gpu))
model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
embedding_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def get_sentence_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = embedding_model(**inputs, output_hidden_states=True)
    last_hidden_state = outputs.hidden_states[-1]
    sentence_embedding = last_hidden_state.mean(dim=1)
    return sentence_embedding

def cosine_similarity(embedding1, embedding2):
    return torch.nn.functional.cosine_similarity(embedding1, embedding2).item()

def extract_answer_from_output(question, output):
    answer = qa_pipeline(question=question, context=output)
    return answer['answer']

# 判断输出是否正确
def is_output_correct(question, correct_answer, generated_output, threshold=0.8):
    extracted_answer = extract_answer_from_output(question, generated_output)
    if extracted_answer:
        correct_embedding = get_sentence_embedding(correct_answer)
        extracted_embedding = get_sentence_embedding(extracted_answer)
        
        similarity = cosine_similarity(correct_embedding, extracted_embedding)
        return similarity > threshold, similarity, extracted_answer
    else:
        return False, 0, None

def score(data):
    total = 0
    correct = 0
    for entry in tqdm(data, desc="Processing data"):
        questions = entry.get('question','').split('\n')
        answers = entry.get('answer', '').lower().split('\n')
        output = entry.get('output', '').lower()
        extracted_answers = []
        labels = []
        # get the extracted answer and the label
        for question,answer in zip(questions[:-1],answers):
            label,_,extracted_answer = is_output_correct(question,answer,output)
            labels.append(label)
            extracted_answers.append(extracted_answer)
        # calculate the successful rate
        for label in labels:
            if label == 1:
                correct += 1
                total += 1
            else:
                total += 1
        entry['extracted_answer'] = extracted_answers
        entry['labels'] = labels
    return data

# read the data
with open(data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

updated_data = score(data)

with open(data_path, 'w', encoding='utf-8') as f:
    json.dump(updated_data, f, ensure_ascii=False, indent=4)

print("Finish comparing")
