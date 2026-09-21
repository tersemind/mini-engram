"""按租户训练 LoRA adapter：QA 数据集 → 可挂载的参数化"记忆"。

用法: python -m engram.train_lora --tenant <租户名> [--epochs 4] [--r 16]
"""
import argparse
import json

import torch
from torch.utils.data import Dataset

from .common import adapter_dir, dataset_dir, resolve_model, system_prompt


class QADataset(Dataset):
    def __init__(self, path, tokenizer, company, max_len=1024):
        self.examples = []
        for line in open(path, encoding="utf-8"):
            ex = json.loads(line)
            prompt = tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt(company)},
                 {"role": "user", "content": ex["question"]}],
                tokenize=False, add_generation_prompt=True)
            full = prompt + ex["answer"] + tokenizer.eos_token
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(full, add_special_tokens=False)["input_ids"][:max_len]
            labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
            self.examples.append({"input_ids": full_ids, "labels": labels})

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def make_collate(pad_id):
    def collate(batch):
        width = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            pad = width - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [pad_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            attn.append([1] * len(b["input_ids"]) + [0] * pad)
        return {"input_ids": torch.tensor(input_ids),
                "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(attn)}
    return collate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tenant", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--epochs", type=float, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--company", default=None, help="公司名，默认用租户名")
    ap.add_argument("--data", default="train.jsonl",
                    help="训练文件名（dataset/<租户>/ 下），默认 train.jsonl")
    ap.add_argument("--tag", default="",
                    help="adapter 版本后缀，如 v2 → 保存到 adapters/<租户>_v2/")
    args = ap.parse_args()
    company = args.company or args.tenant

    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments)

    model_path = resolve_model(args.model)
    tok = AutoTokenizer.from_pretrained(model_path)
    ds = QADataset(dataset_dir(args.tenant) / args.data, tok, company)
    print(f"[train] tenant={args.tenant} 样本数={len(ds)}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=args.r, lora_alpha=args.r * 2, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    out = adapter_dir(args.tenant + (f"_{args.tag}" if args.tag else ""))
    trainer = Trainer(
        model=model, train_dataset=ds,
        data_collator=make_collate(tok.pad_token_id),
        args=TrainingArguments(
            output_dir=str(out) + "_ckpt",
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_steps=5,
            bf16=True,
            logging_steps=5,
            save_strategy="no",
            report_to=[],
            remove_unused_columns=False,
            optim="adamw_torch_fused",
        ))
    trainer.train()

    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"[train] adapter 已保存 -> {out}")


if __name__ == "__main__":
    main()
