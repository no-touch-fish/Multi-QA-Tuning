from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import Dataset
import torch
import json
import argparse
import os
from fastchat.model import get_conversation_template

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
output_file = "models/llama3_gsm/checkpoint-1314"

# model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
# model = PeftModel.from_pretrained(model, output_file).to(device)

tokenizer = AutoTokenizer.from_pretrained(model_name)
# 假设你的 question 列表如下
questions = [
    "What is the capital of France? Answer: ",
    "How many continents are there? Answer: ",
    "Who wrote 'Pride and Prejudice'? Answer: ",
]
for question in questions:
    conversation = get_conversation_template(model_name)
    conversation.append_message(
        conversation.roles[0],
        question,
    )
    conversation.append_message(
        conversation.roles[1],
        None,
    )
    print(conversation.get_prompt())
# 逐个输入进行处理（不使用padding）
# results = []
# for question in questions:
#     # 对每个问题进行分词
#     inputs = tokenizer(question, return_tensors="pt")

#     # 将输入数据移动到设备上
#     input_ids = inputs['input_ids'].to(device)
#     attention_mask = inputs['attention_mask'].to(device)

#     # 推理
#     outputs = model.generate(
#         input_ids=input_ids,
#         attention_mask=attention_mask,
#         max_length=256,  # 确保有足够空间生成新内容
#         pad_token_id=tokenizer.pad_token_id  # 设置pad_token_id，确保生成过程正确处理padding
#     )

#     # 解码并存储生成的文本
#     generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     results.append(generated_text)

# # 输出结果
# for result in results:
#     print(result)



