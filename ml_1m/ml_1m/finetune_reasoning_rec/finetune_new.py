import os
import sys
from typing import List

import numpy as np 
import fire
import torch
import transformers
from datasets import load_dataset
from transformers import EarlyStoppingCallback

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer
from sklearn.metrics import roc_auc_score

def generate_prompt(example):
    text = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
    ### Instruction: {example["instruction"]}
    ### Input: {example["input"]}
    ### Response: {example["output"]}"""
    return text

test_data = [{"instruction": "Given the user preference and unpreference, identify whether the user will like the target movie by answering \"Yes.\" or \"No.\".", "input": "\nUser Preference: \"Twister (1996)\", \"Con Air (1997)\", \"River Wild, The (1994)\", \"13th Warrior, The (1999)\", \"Batman Returns (1992)\", \"From Dusk Till Dawn (1996)\", \"Young Guns II (1990)\", \"Demolition Man (1993)\", \"Mummy, The (1999)\", \"Eraser (1996)\", \"Substitute, The (1996)\", \"Maximum Risk (1996)\", \"Sudden Death (1995)\", \"Outbreak (1995)\", \"Dick Tracy (1990)\",\nUser Unpreference:\"Cliffhanger (1993)\",\"Armageddon (1998)\",\"Mercury Rising (1998)\",\"U.S. Marshalls (1998)\",\"Shadow, The (1994)\",\n Whether the user will like the target movie \"Rocketeer, The (1991)\\?", "output": "No."}]
print(generate_prompt(test_data[0]))

base_model = "meta-llama/Llama-2-7b-chat-hf"
max_length = 512

tokenizer = LlamaTokenizer.from_pretrained(
    base_model,
    padding_side = "left",
    add_eos_token = True,
    add_bos_token = True,
    token = os.environ.get("HUGGINGFACE_ACCESS_TOKEN"),
    )
tokenizer.add_special_tokens({"pad_token":"<pad>"})

def generate_and_tokenize_prompt(prompt):
    result = tokenizer(
        generate_prompt(prompt),
        truncation = True,
        max_length = max_length,
        padding = "max_length",
    )
    result["labels"] = result["input_ids"].copy()

train_data_path = "./data/movie/train.json",
val_data_path = "./data/movie/valid.json",

# if train_data_path.endswith(".json"):
#     train_data = load_dataset("json", data_files=train_data_path)
# else:
#     train_data = load_dataset(train_data_path)

# if val_data_path.endswith(".json"):
#     val_data = load_dataset("json", data_files=val_data_path)
# else:
#     val_data = load_dataset(val_data_path)

