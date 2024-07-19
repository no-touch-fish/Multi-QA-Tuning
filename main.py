import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json

# load dataset
data_file = 'dataset/pararel_test.json'
with open(data_file, 'r') as file:
    data = json.load(file)
df = pd.DataFrame(data)
# load model and tokenizer
model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# generate output
def generate_output(question):
    inputs = tokenizer.encode(question, return_tensors='pt')
    outputs = model.generate(inputs, max_length=50)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

output_data = []
for question in df['question']:
    output = generate_output(question)
    output_data.append({'question': question, 'output': output})

# save result
output_file = 'result/llama3.json'
with open(output_file, 'w') as file:
    json.dump(output_data, file, ensure_ascii=False, indent=4)

print(f"结果已保存到 {output_file}")