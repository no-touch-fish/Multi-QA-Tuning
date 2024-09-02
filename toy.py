from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import Dataset
import torch
import json
import argparse
import os
import numpy as np
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
output_file = "models/llama3_gsm"

pattern = r"1:\s*([^2]*)2:\s*([^3]*)3:\s*([^1]*)"
confidence = '1: a 2: b 3: c 1: I am sure 2: I am sure 3: I am sure'
# confidence = '1: I am sure 2: I am sure 3: I am sure'
matches = re.findall(pattern, confidence)
if matches:
    # match = re.search(pattern, matches[-1])
    match = matches[-1]
    print(matches)
    print(match)






