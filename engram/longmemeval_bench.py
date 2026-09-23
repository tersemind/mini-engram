"""LongMemEval-S(50) 竞品对比 runner：每题一租户，四条件生成答案（判分走 lme_judge）。

条件：base（闭卷）/ full（haystack 时间序头部 32K 截断）/ rag5（BM25 top-5）/ lora（烤 adapter 闭卷）。
生成结果写入 data/bench/lme_rows__<cond>.jsonl，之后用官方 judge 模板判分。

用法: python -m engram.longmemeval_bench [--conditions base,full,rag5,lora]
"""
import argparse
import json
from pathlib import Path

from .common import DATA, adapter_dir, corpus_dir, dataset_dir, resolve_model, system_prompt
from .rag import BM25

FAITHFUL_SUFFIX = (
    "\n\n———\nAnswer the question based only on the above information. "
    "If it does not contain the answer, say you don't have that "
    "information.\n\nQuestion: {q}")

ROUTER_TMPL = (
    "You are a routing classifier for a memory assistant. Given only the "
    "user's question, decide which subsystem should answer:\n"
    "- FACT: it asks about a specific fact, person, date, number, preference, "
    "plan or relationship the assistant should remember about the user.\n"
    "- SCAN: it requires scanning conversation history, references something "
    "possibly never mentioned (\"Did I ever mention ...?\", \"Have I ever "
    "...?\"), asks for verbatim or open-ended content, or asks about "
    "absence/negation.\n\n"
    "Reply with exactly one word: FACT or SCAN.\n\nQuestion: {q}")


def tenants() -> list[str]:
    """只 bench 有完整 adapter 的租户（各条件同题集）。"""
    ds = DATA / "dataset"
    return sorted(p.name for p in ds.iterdir()
                  if p.is_dir() and p.name.startswith("lme_")
                  and (p / "qa.jsonl").exists()
                  and (adapter_dir(p.name) / "adapter_config.json").exists())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--conditions", default="base,full,rag5,lora")
    ap.add_argument("--oracle-src", default=str(
        DATA / "longmemeval" / "longmemeval_oracle.json"),
                    help="oracle 条件用（只给证据 session 的上界参照）")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--rag-k", type=int, default=5)
    ap.add_argument("--rag-chars", type=int, default=2000)
    ap.add_argument("--full-len", type=int, default=32768)
    ap.add_argument("--tenants", default=None,
                    help="只 bench 指定租户（逗号分隔），默认全部有 adapter 的 lme_*")
    ap.add_argument("--adapter-root", default=None,
                    help="adapter 根目录（默认 data/adapters），merged 条件用 data/adapters_merged")
    ap.add_argument("--yarn", action="store_true",
                    help="启用 YaRN 把 32K 原生窗口扩到 128K（LongMemEval 完整 full 条件用）")
    args = ap.parse_args()
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]

    tids = tenants()
    if args.adapter_root:
        want_all = not args.tenants
        base = [p.name for p in (DATA / "dataset").iterdir()
                if p.name.startswith("lme_") and (p / "qa.jsonl").exists()]
        if want_all:
            tids = [t for t in sorted(base)
                    if (Path(args.adapter_root) / t / "adapter_config.json").exists()]
        else:
            want = {x.strip() for x in args.tenants.split(",") if x.strip()}
            tids = [t for t in base if t in want]
    if args.tenants:
        want = {t.strip() for t in args.tenants.split(",") if t.strip()}
        tids = [t for t in tids if t in want]
    if not tids:
        raise SystemExit("没有找到 lme_* 租户，先跑 python -m engram.longmemeval_prep")
    qa = {}
    for t in tids:
        rows = [json.loads(l) for l in
                (dataset_dir(t) / "qa.jsonl").open(encoding="utf-8")]
        qa[t] = rows[0]
    print(f"[lme_bench] {len(tids)} 题; 条件: {conds}")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    model_path = resolve_model(args.model)
    tok = AutoTokenizer.from_pretrained(model_path)
    sp = SamplingParams(temperature=0.0, max_tokens=200)
    outdir = DATA / "bench"
    outdir.mkdir(exist_ok=True)

    def render(system: str, user: str) -> str:
        return tok.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True)

    def chunks(tid: str) -> list[dict]:
        return [json.loads(l) for l in
                (corpus_dir(tid) / "chunks.jsonl").open(encoding="utf-8")]

    def dump(cond: str, rows: list[dict]):
        with (outdir / f"lme_rows__{cond}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[lme_bench] {cond:5s} done, n={len(rows)}")

    llm_kwargs = dict(model=model_path, enable_lora=True, max_lora_rank=64,
                      max_model_len=args.full_len,
                      gpu_memory_utilization=args.gpu_mem, enforce_eager=True)
    if args.yarn:
        llm_kwargs["hf_overrides"] = {"rope_scaling": {
            "rope_type": "yarn", "factor": 4.0,
            "original_max_position_embeddings": 32768}}
        llm_kwargs["kv_cache_dtype"] = "fp8"  # 112K 长文下 KV cache 减半，24GB 卡可承载
    llm = LLM(**llm_kwargs)
    oracle_map = {}
    if "oracle" in conds:
        for s in json.load(open(args.oracle_src, encoding="utf-8")):
            oracle_map[f"lme_{s['question_id'][:6]}"] = s

    def gen_lora(tid: str, q: str):
        adir = (Path(args.adapter_root) / tid) if args.adapter_root \
            else adapter_dir(tid)
        if not (adir / "adapter_config.json").exists():
            return None
        req = LoRARequest(tid, tids.index(tid) + 1, str(adir))
        return llm.generate([render(system_prompt(tid), q)], sp,
                            lora_request=req)

    def gen_rag(tid: str, q: str):
        bm = BM25(chunks(tid))
        hits = bm.search(q, args.rag_k)
        ctx = "\n\n".join(
            f"[{i+1}] ({h['source']})\n{h['text'][:args.rag_chars]}"
            for i, h in enumerate(hits)) or "(no relevant excerpt)"
        user = ("Here are the most relevant excerpts from the user's "
                f"chat history:\n\n{ctx}" + FAITHFUL_SUFFIX.format(q=q))
        return llm.generate([render(system_prompt(tid), user)], sp)

    def route(q: str, tid: str) -> str:
        sp_r = SamplingParams(temperature=0.0, max_tokens=5)
        user = ROUTER_TMPL.format(q=q)
        out = llm.generate([render(system_prompt(tid), user)], sp_r)
        verdict = out[0].outputs[0].text.strip().upper()
        return "scan" if "SCAN" in verdict else "fact"

    for cond in conds:
        rows = []
        for tid in tids:
            q, tr = qa[tid]["question"], qa[tid]
            if cond == "oracle":
                s = oracle_map.get(tid)
                if not s:
                    continue
                parts = []
                for idx, sess in enumerate(s["haystack_sessions"]):
                    dt = (s["haystack_dates"][idx]
                          if idx < len(s["haystack_dates"]) else "")
                    parts.append((f"[session {idx+1}, {dt}]\n" if dt else "")
                                 + "\n".join(f"{t['role']}: {t['content']}"
                                             for t in sess))
                user = ("Here are the relevant excerpts from the user's chat "
                        "history with an AI assistant:\n\n" + "\n\n".join(parts)
                        + FAITHFUL_SUFFIX.format(q=q))
                outs = llm.generate([render(system_prompt(tid), user)], sp)
            elif cond == "full":
                hist, n_tok = [], 0
                for c in chunks(tid):  # 时间序头部，超预算即截断
                    t = len(tok(c["text"])["input_ids"])
                    if n_tok + t > args.full_len - 1500 and hist:
                        break
                    hist.append(f"({c['source']})\n{c['text']}")
                    n_tok += t
                user = ("Here is the user's chat history with an AI assistant "
                        "in chronological order:\n\n" + "\n\n".join(hist)
                        + FAITHFUL_SUFFIX.format(q=q))
                outs = llm.generate([render(system_prompt(tid), user)], sp)
            elif cond == "rag5":
                outs = gen_rag(tid, q)
            elif cond == "lora" or cond == "hybrid" or cond.startswith("lora"):
                outs = gen_lora(tid, q)
                if cond == "hybrid":
                    if outs is None:
                        print(f"[lme_bench] 跳过 {tid}（无 adapter）")
                        continue
                    verdict = route(q, tid)
                    outs = gen_rag(tid, q) if verdict == "scan" else outs
                    rows.append({"tenant": tid, "question_type": tr["question_type"],
                                 "question": q, "gold": tr["answer"],
                                 "pred": outs[0].outputs[0].text.strip(),
                                 "judge": None, "route": verdict})
                    continue
                if outs is None:
                    print(f"[lme_bench] 跳过 {tid}（无 adapter）")
                    continue
            else:  # base
                outs = llm.generate([render(system_prompt(tid), q)], sp)
            rows.append({"tenant": tid, "question_type": tr["question_type"],
                         "question": q, "gold": tr["answer"],
                         "pred": outs[0].outputs[0].text.strip(),
                         "judge": None})
        dump(cond, rows)

    print(f"明细 -> {outdir}/lme_rows__*.jsonl；下一步: python -m engram.lme_judge")


if __name__ == "__main__":
    main()
