"""训练跨租户共享的"技能" adapter（Engram 内容/技能分离的技能层）。

与 train_lora 的区别：样本来自多个租户，每个样本用该租户自己的
system_prompt（保留租户身份条件，避免技能绑定到单一身份）；
每租户贡献上限 max_per_tenant，类别均衡。训练出的 skill adapter 之后与
各租户 content adapter 用 engram.merge_adapters 精确相加合并。

用法: python -m engram.train_skill --tenants t1,t2,... [--r 32] [--epochs 2]
"""
import argparse
import json
import random

import torch
from torch.utils.data import Dataset

from .common import dataset_dir, resolve_model, system_prompt

from .train_lora import make_collate


class MultiTenantQADataset(Dataset):
    def __init__(self, rows, tokenizer, max_len=1024):
        self.examples = []
        for ex in rows:
            prompt = tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt(ex["tenant"])},
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


def collect(tenants: list[str], max_per: int, cap: int, seed: int,
            prefer_tagged: bool) -> list[dict]:
    """收集各租户 QA，每租户上限 max_per，总量上限 cap，偏置专项 pass。"""
    rng = random.Random(seed)
    rows = []
    for t in tenants:
        per = []
        for line in (dataset_dir(t) / "train.jsonl").open(encoding="utf-8"):
            r = json.loads(line)
            r["tenant"] = t
            per.append(r)
        if prefer_tagged:  # 专项 pass（cross/temporal/preference/update）优先
            tagged = [r for r in per if "#" in r.get("source", "")]
            plain = [r for r in per if "#" not in r.get("source", "")]
            rng.shuffle(tagged)
            rng.shuffle(plain)
            per = tagged + plain
        else:
            rng.shuffle(per)
        rows += per[:max_per]
    rng.shuffle(rows)
    return rows[:cap]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tenants", required=True, help="逗号分隔（或 @文件）")
    ap.add_argument("--r", type=int, default=32)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-per-tenant", type=int, default=2000)
    ap.add_argument("--cap", type=int, default=40000, help="总样本上限")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="skill", help="输出 adapters/<tag>/")
    args = ap.parse_args()

    spec = args.tenants
    if spec.startswith("@"):
        tenants = [l.strip() for l in open(spec[1:]) if l.strip()]
    else:
        tenants = [t.strip() for t in spec.split(",") if t.strip()]

    rows = collect(tenants, args.max_per_tenant, args.cap, args.seed,
                   prefer_tagged=True)
    by_t = {}
    for r in rows:
        by_t[r["tenant"]] = by_t.get(r["tenant"], 0) + 1
    print(f"[skill] {len(rows)} 样本 / {len(by_t)} 租户 "
          f"(每租户 {min(by_t.values())}~{max(by_t.values())})")

    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments)
    from .common import adapter_dir

    model_path = resolve_model()
    tok = AutoTokenizer.from_pretrained(model_path)
    ds = MultiTenantQADataset(rows, tok)

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

    out = adapter_dir(args.tag)
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
            warmup_steps=10,
            bf16=True,
            logging_steps=10,
            save_strategy="no",
            report_to=[],
            remove_unused_columns=False,
            optim="adamw_torch_fused",
        ))
    trainer.train()
    model.save_pretrained(out)
    print(f"[skill] adapter 已保存 -> {out}")


if __name__ == "__main__":
    main()
