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


# The following code was adapted from the `evaluate` library. Licensed under the Apache License, Version 2.0 (the "License").
# We modify them to avoid causing serious memory issues in the Colab environment.

def compute_ppl(
        model, tokenizer, inputs, device, batch_size: int = 16, add_start_token: bool = True, max_length=None
):

    if device is not None:
        assert device in ["gpu", "cpu", "cuda"], "device should be either gpu or cpu."
        if device == "gpu":
            device = "cuda"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # if batch_size > 1 (which generally leads to padding being required), and
    # if there is not an already assigned pad_token, assign an existing
    # special token to also be the padding token
    if tokenizer.pad_token is None and batch_size > 1:
        existing_special_tokens = list(tokenizer.special_tokens_map_extended.values())
        # check that the model already has at least one special token defined
        assert (
            len(existing_special_tokens) > 0
        ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
        # assign one of the special tokens to also be the pad token
        tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

    if add_start_token and max_length:
        # leave room for <BOS> token to be added:
        assert (
            tokenizer.bos_token is not None
        ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
        max_tokenized_len = max_length - 1
    else:
        max_tokenized_len = max_length

    encodings = tokenizer(
        inputs,
        add_special_tokens=False,
        padding=True,
        truncation=True if max_tokenized_len else False,
        max_length=max_tokenized_len,
        return_tensors="pt",
        return_attention_mask=True,
    )

    encoded_texts = encodings["input_ids"]
    attn_masks = encodings["attention_mask"]

    # check that each input is long enough:
    if add_start_token:
        assert torch.all(torch.ge(attn_masks.sum(1), 1)), "Each input text must be at least one token long."
    else:
        assert torch.all(
            torch.ge(attn_masks.sum(1), 2)
        ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

    ppls = []
    loss_fct = CrossEntropyLoss(reduction="none")

    for start_index in tqdm(range(0, len(encoded_texts), batch_size)):
        end_index = min(start_index + batch_size, len(encoded_texts))
        encoded_batch = encoded_texts[start_index:end_index].to(device)
        attn_mask = attn_masks[start_index:end_index].to(device)

        if add_start_token:
            bos_tokens_tensor = torch.tensor([[tokenizer.bos_token_id]] * encoded_batch.size(dim=0)).to(device)
            encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
            attn_mask = torch.cat(
                [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(device), attn_mask], dim=1
            )

        labels = encoded_batch

        with torch.no_grad():
            out_logits = model(encoded_batch, attention_mask=attn_mask).logits

            shift_logits = out_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

            perplexity_batch = torch.exp(
                (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
                / shift_attention_mask_batch.sum(1)
            )

            ppls += perplexity_batch.tolist()

    del encoded_batch, attn_mask
    if device == "cuda":
        torch.cuda.empty_cache()

    return {"perplexities": ppls, "mean_perplexity": sum(ppls)/float(len(ppls))}


# 加载训练后的模型并测试生成
def main():
    # 设置文件路径 - 请根据您的环境修改这些路径
    TRAIN_FILE = './data/zh_train.jsonl'
    VALIDATION_FILE = './data/zh_dev.jsonl'
    TEST_FILE = './data/zh_test.jsonl'
    MODEL_PATH =  "./llama-42m-zh-fairytales-linear"


    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device type: {device}")

    model_dir = os.path.abspath(MODEL_PATH)
    # 检查是否存在检查点目录
    checkpoint_dir = os.path.join(model_dir, "checkpoint-2000")
    if os.path.exists(checkpoint_dir):
        # 如果检查点目录存在，使用它
        model_path_to_load = checkpoint_dir
        print(f"Loading from checkpoint: {model_path_to_load}")
    else:
        # 否则使用基础模型目录
        model_path_to_load = model_dir
        print(f"Loading from base model: {model_path_to_load}")

    # 加载模型和分词器
    try:
        model = LlamaForCausalLM.from_pretrained(model_path_to_load).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)  # 分词器从基础目录加载
    except Exception as e:
        print(f"Error loading model: {e}")
        # 尝试从基础目录加载
        try:
            model = LlamaForCausalLM.from_pretrained(model_dir).to(device)
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            print("Loaded from base directory instead")
        except Exception as e2:
            print(f"Failed to load model: {e2}")
            return

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data_file = TEST_FILE
    test_dataset = list(load_dataset('json', data_files={'test': data_file})["test"]["text"])

    results = compute_ppl(model=model, tokenizer=tokenizer, device=device, inputs=test_dataset, batch_size = 16)
    dataset_ppl = results['mean_perplexity']
    print(f"Test Perplexity: {dataset_ppl:.2f}")


if __name__ == '__main__':
    main()

