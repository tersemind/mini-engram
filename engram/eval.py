"""评测"知识是否烤进了参数"：同一套题，闭卷(无 adapter) vs 烤入后(带 adapter) 对比。

用法: python -m engram.eval --tenant <租户名> [--limit 0(全部)]
"""
import argparse
import json
import re
import string
from collections import Counter

from .common import adapter_dir, dataset_dir, resolve_model, system_prompt


def norm(s: str) -> list[str]:
    s = s.lower()
    s = re.sub(r"[{}]".format(re.escape(string.punctuation)), " ", s)
    # 中英文混排：按字符切，兼容中文答案
    return [ch for ch in s if not ch.isspace()]


def f1(pred: str, gold: str) -> float:
    p, g = norm(pred), norm(gold)
    if not p or not g:
        return 0.0
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tenant", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--quiz", default=None,
                    help="用指定租户的 val 集测本 adapter（串味检查）")
    ap.add_argument("--company", default=None, help="公司名，默认用租户名")
    args = ap.parse_args()
    company = args.company or args.tenant

    quiz_src = args.quiz or args.tenant
    quiz = [json.loads(l) for l in
            (dataset_dir(quiz_src) / "val.jsonl").open(encoding="utf-8")]
    if args.limit:
        quiz = quiz[: args.limit]
    adapter = adapter_dir(args.tenant)
    if not (adapter / "adapter_config.json").exists():
        raise SystemExit(f"没有找到 adapter: {adapter}，请先跑 train_lora")
    model_path = resolve_model(args.model)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    tok = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(model=model_path, enable_lora=True, max_lora_rank=64,
              max_model_len=4096, gpu_memory_utilization=0.85,
              enforce_eager=True)
    sp = SamplingParams(temperature=0.0, max_tokens=200)
    prompts = [
        tok.apply_chat_template(
            [{"role": "system", "content": system_prompt(company)},
             {"role": "user", "content": q["question"]}],
            tokenize=False, add_generation_prompt=True)
        for q in quiz
    ]

    lora_req = LoRARequest(args.tenant, 1, str(adapter))
    ans_base = [o.outputs[0].text.strip()
                for o in llm.generate(prompts, sp)]                    # 闭卷
    ans_lora = [o.outputs[0].text.strip()
                for o in llm.generate(prompts, sp, lora_request=lora_req)]  # 烤入后

    rows, f1_base_sum, f1_lora_sum = [], 0.0, 0.0
    for q, ab, al in zip(quiz, ans_base, ans_lora):
        fb, fl = f1(ab, q["answer"]), f1(al, q["answer"])
        f1_base_sum += fb
        f1_lora_sum += fl
        rows.append({"question": q["question"], "gold": q["answer"],
                     "base": ab, "lora": al, "f1_base": fb, "f1_lora": fl})

    n = len(rows)
    report = {"tenant": args.tenant, "quiz": quiz_src, "n": n,
              "f1_base": round(f1_base_sum / n, 4),
              "f1_lora": round(f1_lora_sum / n, 4), "rows": rows}
    out = dataset_dir(args.tenant) / "eval_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n===== 评测结果 tenant={args.tenant} (n={n}) =====")
    print(f"闭卷 (base model)      平均 F1: {report['f1_base']:.3f}")
    print(f"烤入后 (base + LoRA)   平均 F1: {report['f1_lora']:.3f}")
    print("\n--- 示例对比 ---")
    for r in rows[:5]:
        print(f"Q: {r['question']}")
        print(f"  标准答案: {r['gold']}")
        print(f"  闭卷:     {r['base'][:80]}")
        print(f"  烤入后:   {r['lora'][:80]}")
    print(f"\n完整报告 -> {out}")


if __name__ == "__main__":
    main()
