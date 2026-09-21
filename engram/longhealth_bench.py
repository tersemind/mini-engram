"""LongHealth 竞品对比 runner：五选一医学选择题准确率（Cartridges 官方尺子）。

条件：base（闭卷）/ full（信件全文进上下文）/ rag5（BM25 top-5）/ lora（烤 adapter 闭卷）。
评分：抽取生成文本中的选项字母（A-E），与 correct 所属字母比对；抽不出字母时
回退做 correct 文本包含匹配。逐题明细 + 按条件/病人汇总。

用法: python -m engram.longhealth_bench [--conditions base,full,rag5,lora]
输出: data/bench/longhealth_rows__<cond>.jsonl + longhealth_summary.json
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from .common import DATA, adapter_dir, corpus_dir, dataset_dir, resolve_model, system_prompt
from .rag import BM25

LETTERS = "ABCDE"


def tenants() -> list[str]:
    ds = DATA / "dataset"
    return sorted(p.name for p in ds.iterdir()
                  if p.is_dir() and p.name.startswith("lh_")
                  and (p / "qa.jsonl").exists())


def load_qa(tenant: str) -> list[dict]:
    return [json.loads(l) for l in
            (dataset_dir(tenant) / "qa.jsonl").open(encoding="utf-8")]


def mc_prompt(q: dict) -> str:
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(q["options"]))
    return (f"{q['question']}\n{opts}\n\n"
            "Answer with the letter of the correct option (A-E).")


def extract_letter(pred: str) -> str | None:
    m = re.search(r"\b([A-E])\b", pred.upper())
    return m.group(1) if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--conditions", default="base,full,rag5,lora")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--rag-k", type=int, default=5)
    ap.add_argument("--rag-chars", type=int, default=2000)
    ap.add_argument("--full-len", type=int, default=8192)
    args = ap.parse_args()
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]

    tids = tenants()
    if not tids:
        raise SystemExit("没有找到 lh_* 租户，先跑 python -m engram.longhealth_prep")
    all_qa = {t: load_qa(t) for t in tids}
    print(f"[longhealth_bench] {len(tids)} 病人, "
          f"{sum(len(v) for v in all_qa.values())} 题; 条件: {conds}")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    model_path = resolve_model(args.model)
    tok = AutoTokenizer.from_pretrained(model_path)
    sp = SamplingParams(temperature=0.0, max_tokens=60)
    outdir = DATA / "bench"
    outdir.mkdir(exist_ok=True)
    rows: dict[str, list[dict]] = {c: [] for c in conds}

    def render(system: str, user: str) -> str:
        return tok.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True)

    def score_rows(cond: str, tid: str, preds: list[str]):
        for q, pred in zip(all_qa[tid], preds):
            letter = extract_letter(pred)
            gold_idx = next((i for i, o in enumerate(q["options"])
                             if o == q["correct"]), -1)
            hit = (letter == LETTERS[gold_idx] if letter and gold_idx >= 0
                   else q["correct"].lower() in pred.lower())
            rows[cond].append({"tenant": tid, "question": q["question"],
                               "gold": q["correct"], "pred": pred,
                               "letter": letter, "correct": int(bool(hit))})

    def dump(cond: str):
        rr = rows[cond]
        acc = sum(r["correct"] for r in rr) / max(len(rr), 1)
        with (outdir / f"longhealth_rows__{cond}.jsonl").open("w", encoding="utf-8") as f:
            for r in rr:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[longhealth_bench] {cond:5s} acc = {acc:.4f} (n={len(rr)})")

    llm = LLM(model=model_path, enable_lora=True, max_lora_rank=64,
              max_model_len=args.full_len, gpu_memory_utilization=args.gpu_mem,
              enforce_eager=True)
    for cond in conds:
        for tid in tids:
            if cond == "full":
                chunks = [json.loads(l) for l in
                          (corpus_dir(tid) / "chunks.jsonl").open(encoding="utf-8")]
                body, n_tok = [], 0
                for c in chunks:  # 超预算丢尾部块
                    t = len(tok(c["text"])["input_ids"])
                    if n_tok + t > args.full_len - 1500 and body:
                        break
                    body.append(f"({c['source']})\n{c['text']}")
                    n_tok += t
                prompts = [render(system_prompt(tid),
                                  "Reference medical record:\n\n" + "\n\n".join(body)
                                  + "\n\n———\n" + mc_prompt(q))
                           for q in all_qa[tid]]
                outs = llm.generate(prompts, sp)
            elif cond == "rag5":
                bm = BM25([json.loads(l) for l in
                           (corpus_dir(tid) / "chunks.jsonl").open(encoding="utf-8")])
                prompts = []
                for q in all_qa[tid]:
                    hits = bm.search(q["question"], args.rag_k)
                    ctx = "\n\n".join(
                        f"[{i+1}] ({h['source']})\n{h['text'][:args.rag_chars]}"
                        for i, h in enumerate(hits)) or "(no relevant excerpt)"
                    prompts.append(render(
                        system_prompt(tid),
                        "Reference medical record excerpts:\n\n" + ctx
                        + "\n\n———\nAnswer the question based only on the "
                          "record. " + mc_prompt(q)))
                outs = llm.generate(prompts, sp)
            elif cond == "lora":
                adir = adapter_dir(tid)
                if not (adir / "adapter_config.json").exists():
                    print(f"[longhealth_bench] 跳过 {tid}（无 adapter）")
                    continue
                req = LoRARequest(tid, tids.index(tid) + 1, str(adir))
                prompts = [render(system_prompt(tid), mc_prompt(q))
                           for q in all_qa[tid]]
                outs = llm.generate(prompts, sp, lora_request=req)
            else:  # base
                prompts = [render(system_prompt(tid), mc_prompt(q))
                           for q in all_qa[tid]]
                outs = llm.generate(prompts, sp)
            score_rows(cond, tid, [o.outputs[0].text.strip() for o in outs])
        dump(cond)

    summary: dict[str, dict] = {"overall": {}, "by_patient": {}}
    for c in conds:
        rr = rows.get(c) or []
        if not rr:
            continue
        summary["overall"][c] = round(sum(r["correct"] for r in rr) / len(rr), 4)
        by: dict[str, list[int]] = defaultdict(list)
        for r in rr:
            by[r["tenant"]].append(r["correct"])
        summary["by_patient"][c] = {k: round(sum(v) / len(v), 4)
                                    for k, v in sorted(by.items())}
    (outdir / "longhealth_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n===== LongHealth 汇总（MC accuracy）=====")
    print(f"{'condition':12s}" + "".join(f"{c:>10s}" for c in conds))
    print(f"{'overall':12s}" + "".join(f"{summary['overall'].get(c, float('nan')):10.4f}"
                                       for c in conds))
    print(f"明细 -> {outdir}")


if __name__ == "__main__":
    main()
