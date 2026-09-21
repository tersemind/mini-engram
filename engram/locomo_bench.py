"""LoCoMo 竞品对比 runner：真实长对话（avg ~17k tokens）上的"记忆方式"对比。

条件（对照 Mem0 论文的基线设置）：
  base   闭卷 —— 无记忆
  full   整段对话塞进上下文（~上界参照；超长截断，记入 truncated）
  rag5   BM25 检索 top-5 session 进上下文 —— Mem0/RAG 式检索记忆
  lora   每段对话烤一个 adapter（engram 链路：prep→synth→train），闭卷作答

评分：locomo_score.score_sample 官方口径（normalize→Porter stem→token 多重集 F1；
adversarial 判 "no information available"/"not mentioned"）。

用法: python -m engram.locomo_bench [--conditions base,full,rag5,lora]
输出: data/bench/locomo_rows__<cond>.jsonl + locomo_summary.json
"""
import argparse
import json
from pathlib import Path

from .common import DATA, adapter_dir, corpus_dir, dataset_dir, resolve_model, system_prompt
from .locomo_score import CATEGORIES, score_sample
from .rag import BM25

FAITHFUL_SUFFIX = (
    "\n\n———\nAnswer the question based only on the above information. "
    "If it does not contain the answer, respond exactly "
    "'No information available'.\n\nQuestion: {q}")


def tenants() -> list[str]:
    ds = DATA / "dataset"
    return sorted(p.name for p in ds.iterdir()
                  if p.is_dir() and p.name.startswith("locomo_")
                  and (p / "qa.jsonl").exists())


def load_qa(tenant: str) -> list[dict]:
    return [json.loads(l) for l in
            (dataset_dir(tenant) / "qa.jsonl").open(encoding="utf-8")]


def load_chunks(tenant: str) -> list[dict]:
    return [json.loads(l) for l in
            (corpus_dir(tenant) / "chunks.jsonl").open(encoding="utf-8")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--conditions", default="base,full,rag5,lora")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--rag-k", type=int, default=5)
    ap.add_argument("--rag-chars", type=int, default=2000)
    ap.add_argument("--full-len", type=int, default=32768)
    args = ap.parse_args()
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]

    tids = tenants()
    if not tids:
        raise SystemExit("没有找到 locomo_* 租户，先跑 python -m engram.locomo_prep")
    all_qa = {t: load_qa(t) for t in tids}
    print(f"[locomo_bench] {len(tids)} 段对话, "
          f"{sum(len(v) for v in all_qa.values())} 题; 条件: {conds}")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    model_path = resolve_model(args.model)
    tok = AutoTokenizer.from_pretrained(model_path)
    sp = SamplingParams(temperature=0.0, max_tokens=200)
    outdir = DATA / "bench"
    outdir.mkdir(exist_ok=True)

    rows: dict[str, list[dict]] = {c: [] for c in conds}
    truncated: dict[str, bool] = {}

    def render(system: str, user: str) -> str:
        return tok.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True)

    def score_rows(cond: str, tid: str, preds: list[str]):
        for q, pred in zip(all_qa[tid], preds):
            s = score_sample(q["category"], pred, q["answer"] or "")
            rows[cond].append({"tenant": tid, "category": CATEGORIES[q["category"]],
                               "question": q["question"], "gold": q["answer"],
                               "pred": pred, "score": round(s, 4)})

    def dump(cond: str):
        rr = rows[cond]
        with (outdir / f"locomo_rows__{cond}.jsonl").open("w", encoding="utf-8") as f:
            for r in rr:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[locomo_bench] {cond:5s} done, n={len(rr)}, "
              f"F1={sum(r['score'] for r in rr) / max(len(rr), 1):.4f}")

    # ---- 短上下文会话：base / rag5 / lora ----
    short = [c for c in conds if c != "full"]
    if short:
        llm = LLM(model=model_path, enable_lora=True, max_lora_rank=64,
                  max_model_len=4096, gpu_memory_utilization=args.gpu_mem,
                  enforce_eager=True)
        if "base" in short:
            for tid in tids:
                prompts = [render(system_prompt(tid), q["question"])
                           for q in all_qa[tid]]
                outs = llm.generate(prompts, sp)
                score_rows("base", tid, [o.outputs[0].text.strip() for o in outs])
            dump("base")
        if "rag5" in short:
            for tid in tids:
                bm = BM25(load_chunks(tid))
                prompts = []
                for q in all_qa[tid]:
                    hits = bm.search(q["question"], args.rag_k)
                    ctx = "\n\n".join(
                        f"[{i+1}] ({h['source']})\n{h['text'][:args.rag_chars]}"
                        for i, h in enumerate(hits)) or "(no relevant excerpt found)"
                    user = ("Here are the most relevant excerpts from the chat "
                            f"history:\n\n{ctx}" + FAITHFUL_SUFFIX.format(q=q["question"]))
                    prompts.append(render(system_prompt(tid), user))
                outs = llm.generate(prompts, sp)
                score_rows("rag5", tid, [o.outputs[0].text.strip() for o in outs])
            dump("rag5")
        if "lora" in short:
            for tid in tids:
                adir = adapter_dir(tid)
                if not (adir / "adapter_config.json").exists():
                    print(f"[locomo_bench] 跳过 {tid}（无 adapter）")
                    continue
                req = LoRARequest(tid, tids.index(tid) + 1, str(adir))
                prompts = [render(system_prompt(tid), q["question"])
                           for q in all_qa[tid]]
                outs = llm.generate(prompts, sp, lora_request=req)
                score_rows("lora", tid, [o.outputs[0].text.strip() for o in outs])
            dump("lora")
        del llm
        import gc, torch
        gc.collect()
        torch.cuda.empty_cache()

    # ---- 长上下文会话：full ----
    if "full" in conds:
        llm = LLM(model=model_path, max_model_len=args.full_len,
                  gpu_memory_utilization=args.gpu_mem, enforce_eager=True)
        budget = args.full_len - 800
        for tid in tids:
            chunks = load_chunks(tid)  # 按 session 顺序 = 时间顺序
            hist, n_tok = [], 0
            for c in chunks:
                t = len(tok(c["text"])["input_ids"])
                if n_tok + t > budget and hist:
                    truncated[tid] = True
                    break
                hist.append(f"({c['source']})\n{c['text']}")
                n_tok += t
            body = "\n\n".join(hist)
            prompts = [
                render(system_prompt(tid),
                       "Here is the full chat history in chronological order:"
                       f"\n\n{body}" + FAITHFUL_SUFFIX.format(q=q["question"]))
                for q in all_qa[tid]]
            outs = llm.generate(prompts, sp)
            score_rows("full", tid, [o.outputs[0].text.strip() for o in outs])
        del llm
        dump("full")

    # ---- 汇总 ----
    summary: dict[str, dict] = {"overall": {}, "by_category": {}, "by_conversation": {},
                                "truncated": truncated}
    for c in conds:
        rr = rows.get(c) or []
        if not rr:
            continue
        summary["overall"][c] = round(sum(r["score"] for r in rr) / len(rr), 4)
        by_cat: dict[str, list[float]] = {}
        for r in rr:
            by_cat.setdefault(r["category"], []).append(r["score"])
        summary["by_category"][c] = {k: round(sum(v) / len(v), 4)
                                     for k, v in sorted(by_cat.items())}
        by_conv: dict[str, list[float]] = {}
        for r in rr:
            by_conv.setdefault(r["tenant"], []).append(r["score"])
        summary["by_conversation"][c] = {k: round(sum(v) / len(v), 4)
                                         for k, v in sorted(by_conv.items())}
    (outdir / "locomo_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n===== LoCoMo 汇总（official token-F1）=====")
    print(f"{'condition':12s}" + "".join(f"{c:>10s}" for c in conds))
    print(f"{'overall':12s}" + "".join(f"{summary['overall'].get(c, float('nan')):10.4f}"
                                       for c in conds))
    cats = [CATEGORIES[k] for k in sorted(CATEGORIES)]
    for cat in cats:
        line = f"{cat:12s}"
        for c in conds:
            v = summary["by_category"].get(c, {}).get(cat)
            line += f"{v:10.4f}" if v is not None else f"{'—':>10s}"
        print(line)
    if truncated:
        print(f"full 截断租户: {list(truncated)}")
    print(f"明细 -> {outdir}")


if __name__ == "__main__":
    main()
