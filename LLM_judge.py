from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import json
import panda as pd

parser = argparse.ArgumentParser(
    description="""parameter"""
)
parser.add_argument(
    "--data_path",
    type = str,
    default = 'dataset/processed_pararel_test.json',
    help = "Path to the dataset used.",
)
parser.add_argument(
    "--gpu",
    type = str,
    help = "which gpu to use",
)
args = parser.parse_args()

data_file = args.data_path
gpu = args.gpu
# load the data
with open(data_file, 'r') as file:
    data = json.load(file)
df = pd.DataFrame(data)
questions = df['question'].tolist()
answers = df['answer'].tolist()
outputs = df['output'].tolist()
output_data = []

# 加载 T5 模型和分词器
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def is_output_correct(question, correct_answer, generated_output):
    input_text = f"question: {question} correct_answer: {correct_answer} generated_output: {generated_output} Give me a score from 0 to 10 to measure how the generated output is correct and relavant to the question."
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    
    # 生成输出
    outputs = model.generate(input_ids, max_length=50, num_beams=4, early_stopping=True)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return result

for question,answer,output in zip(questions,answers,outputs):
    score = is_output_correct(question, answer, output)
    output_data.append({
        'question': question, 
        'output': output, 
        'answer': answer,
        'score' : score
    })
#   print(response)
with open(data_file,'w') as file:
    json.dump(output_data, file, ensure_ascii=False, indent=4)

print(f"save to {data_file}")
