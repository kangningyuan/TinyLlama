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

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 加载中文数据集
    print("Loading Chinese dataset...")
    chinese_dataset = load_dataset('json', data_files={
        'train': TRAIN_FILE,
        'validation': VALIDATION_FILE,
        'test': TEST_FILE
    })

    print(f"Dataset structure: {chinese_dataset}")

    # 定义处理函数
    block_size = 512

    # 定义tokenize函数
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=block_size,
            padding=False,
            return_attention_mask=True
        )

    # 定义分组函数
    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    print("Tokenizing dataset...")
    # 应用tokenize函数（使用单进程避免多进程问题）
    tokenized_zh_datasets = chinese_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=["text"]
    )

    print("Grouping texts...")
    # 应用分组函数
    lm_datasets = tokenized_zh_datasets.map(
        group_texts,
        batched=True,
        batch_size=512,
        num_proc=4,
    )

    # 设置训练参数
    lr = 1e-4
    epochs = 8  # 为了演示，只训练1个epoch，您可以增加这个值
    save_steps = 200
    strategy = "steps"
    train_bsz = 32  # 较小的批次大小以减少内存使用
    eval_bsz = 16

    training_args = TrainingArguments(
        OUTPUT_DIR,
        eval_strategy=strategy,
        eval_steps=save_steps,
        save_strategy=strategy,
        save_steps=save_steps,
        logging_strategy="steps",
        logging_steps=10,
        learning_rate=lr,
        weight_decay=0.01,
        seed=42,
        per_device_train_batch_size=train_bsz,
        per_device_eval_batch_size=eval_bsz,
        save_total_limit=1,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        adam_beta1=0.9,
        adam_beta2=0.98,
        warmup_ratio=0.01,
        num_train_epochs=epochs,
        report_to=None,
        gradient_accumulation_steps=2,  # 累积梯度以减少内存使用
    )

    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"],
    )

    # 开始训练
    print("Starting training...")
    trainer.train()

    # 保存最终模型
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Training completed!")
    print(f"Model saved to: {OUTPUT_DIR}")

    # 加载训练后的模型并测试生成
    print("Loading trained model for testing...")
    model = LlamaForCausalLM.from_pretrained(OUTPUT_DIR).to(device)
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 测试生成中文故事
    prompt = "从前，有一只叫做汤姆的猫。汤姆和他的朋友们一起玩。"
    max_new_tokens = 100
    do_sample = True
    temperature = 0.3

    tokenized_input = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output_ids = model.generate(
        tokenized_input,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=do_sample,
        temperature=temperature,
    )
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print("Generated story:")
    print(output_text)

if __name__ == '__main__':
    main()