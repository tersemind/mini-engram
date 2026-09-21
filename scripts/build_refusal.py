"""拒答校准数据：问的是别家/无关的事 → 学会说"不知道"，而不是自信瞎编。

数据来源（对租户 X）：
  1. 另一租户 val.jsonl + val_para.jsonl 的问题（带着别家事实的问题）
  2. 另一租户 train.jsonl 抽样（同领域但别家的事实）
目标回答为简短拒答。混入 X 的正样本后输出 train_v2.jsonl。

用法: python scripts/build_refusal.py --tenant xinglan [--other hanhai] [--sample 20]
"""
import argparse
import json
import random

from engram.common import DATA, dataset_dir

REFUSALS = [
    "抱歉，这与本公司无关，我没有这方面的信息，无法回答。",
    "这个问题涉及的不是本公司的信息，我没有相关资料，不能回答。",
    "我只掌握本公司的内部知识，这个问题不在我的知识范围内。",
    "这不是我们公司的事实，我无法确认，建议咨询对应公司。",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tenant", required=True)
    ap.add_argument("--other", default=None, help="默认自动取另一租户")
    ap.add_argument("--sample", type=int, default=20, help="对方 train 集抽样条数")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    other = args.other
    if not other:
        others = [p.name for p in (DATA / "dataset").iterdir()
                  if p.is_dir() and p.name != args.tenant
                  and not p.name.endswith("_ckpt")]
        other = others[0]
    rng = random.Random(args.seed)

    off_topic: list[str] = []
    for name in ("val.jsonl", "val_para.jsonl"):
        p = dataset_dir(other) / name
        if p.exists():
            off_topic += [json.loads(l)["question"] for l in p.open(encoding="utf-8")]
    train_other = [json.loads(l)["question"]
                   for l in (dataset_dir(other) / "train.jsonl").open(encoding="utf-8")]
    off_topic += rng.sample(train_other, min(args.sample, len(train_other)))

    pos = (dataset_dir(args.tenant) / "train.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(l) for l in pos]
    for i, q in enumerate(off_topic):
        rows.append({"question": q, "answer": REFUSALS[i % len(REFUSALS)],
                     "source": f"refusal:{other}"})

    out = dataset_dir(args.tenant) / "train_v2.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_ref = len(off_topic)
    print(f"[refusal] {args.tenant}: 正样本 {len(pos)} + 拒答 {n_ref}（来源 {other}）"
          f" = {len(rows)} 条 -> {out}")


if __name__ == "__main__":
    main()
