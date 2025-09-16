import torch
import transformers
import os
from typing import List, Optional, Tuple, Union
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer
from transformers import Trainer, TrainingArguments
from itertools import chain
from datasets import load_dataset
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from functools import partial



# 加载训练后的模型并测试生成
def main():
    # 设置文件路径 - 请根据您的环境修改这些路径
    TRAIN_FILE = './data/zh_train.jsonl'
    VALIDATION_FILE = './data/zh_dev.jsonl'
    TEST_FILE = './data/zh_test.jsonl'
    MODEL_FOLDER = "./llama-42m"
    OUTPUT_DIR = "llama-42m-zh-fairytales-cosine"

    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device type: {device}")

    # 加载模型和分词器
    model = LlamaForCausalLM.from_pretrained(MODEL_FOLDER).to(device)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_FOLDER)
    print("Loading trained model for testing...")
    model = LlamaForCausalLM.from_pretrained(OUTPUT_DIR).to(device)
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 测试生成中文故事
    prompts = [
        "小明今天过生日，",
        "本有一只会说话的小猫，",
        "下雨天，他捡到一把伞，",
        "学校里来了个新同学，",
        "妈妈做了一个奇怪的梦，",
        "他的玩具车晚上自己动了起来，",
        "暑假的第一天，",
        "他在阁楼发现了一个旧盒子，",
        "天气预报说今天是晴天，可是...",
        ""
    ]

    batch_size = 10  # If you have multiple data inputs, please control the batch size to prevent out-of-memory issues.
    max_new_tokens = 300
    do_sample = True
    temperature = 0.9

    for i in range(0, len(prompts), batch_size):
        batch_input = prompts[i:i + batch_size]
        tokenized_input = tokenizer(batch_input, return_tensors="pt", padding=True).to(device)

        # For decoder-only models, batched inputs of model.generate() should be in the format of input_ids.
        output_ids = model.generate(
            tokenized_input["input_ids"],
            max_new_tokens=max_new_tokens,
            eos_token_id=1,
            do_sample=do_sample,
            temperature=temperature,
        )
        output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        for idx, result in enumerate(output_text):
            print(f"{result}\n**********\n")

if __name__ == '__main__':
    main()