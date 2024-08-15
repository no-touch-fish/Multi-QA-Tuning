from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import Dataset
import torch
import json
import argparse
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
output_file = "models/llama3_gsm"

sampling_params = SamplingParams(
    temperature=0,
    max_tokens=256,
)

model = LLM(
    model=model_name, 
    enable_lora=True,
    max_num_seqs = 1
    )

tokenizer = AutoTokenizer.from_pretrained(model_name)
# 假设你的 question 列表如下
questions = [
    "What is the capital of France? Answer: ",
    "How many continents are there? Answer: ",
    "Who wrote 'Pride and Prejudice'? Answer: ",
]
# 逐个输入进行处理（不使用padding）
results = []
for i,question in enumerate(questions):
    question = {
        'role': 'user',
        'content' : question,
    }
    questions[i] = question

# 对每个问题进行分词
inputs = tokenizer.apply_chat_template(
    conversation=questions,
    padding = False,
    add_generation_prompt=True, 
    return_tensors=None,
    return_dict=False,
    )

# 推理
results = model.generate(
        prompt_token_ids=inputs,
        sampling_params=sampling_params,
        use_tqdm=True,
        lora_request=LoRARequest("lora_adapter", 1, output_file)
        )
# 输出结果
for result in results:
    print(result.outputs[0].text)



