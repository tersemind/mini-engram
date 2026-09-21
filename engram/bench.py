"""竞品对比 bench：一个 vLLM 会话跑完所有"记忆方式 × 题集"组合。

记忆方式（condition）：
  base   闭卷，什么都不给 —— 无记忆基线
  full   把租户语料全文塞进上下文 —— 小语料下的最强对手（上界参照）
  rag3   BM25 检索 top-3 文档块塞进上下文 —— Mem0/RAG 式检索记忆
  rag5   BM25 检索 top-5
  lora   本租户 adapter（参数化记忆）；--adapters "v1,v2" 可同场对比多个版本

题集（quizset）：
  own   本租户 val.jsonl（事实回忆，主指标）
  para  本租户 val_para.jsonl（换问法鲁棒性，需先用 scripts/gen_paraphrase.py 生成）
  cross 另一租户 val.jsonl（串味/幻觉压力测试：问的是别家的事）

用法: python -m engram.bench --tenant xinglan [--adapters v1,v2] [--skip rag5]
输出: data/bench/<tenant>__<quizset>__<cond>.json + summary_<tenant>.json
"""
import argparse
import json
from pathlib import Path

from .common import DATA, adapter_dir, corpus_dir, dataset_dir, resolve_model, system_prompt
from .eval import f1
from .rag import rag_context


def full_context(tenant: str) -> str:
    chunks = sorted((corpus_dir(tenant) / "chunks.jsonl")
                    .open(encoding="utf-8"))
    return "\n\n".join(json.loads(c)["text"] for c in chunks)


def other_tenant(tenant: str) -> str | None:
    ds = DATA / "dataset"
    others = [p.name for p in ds.iterdir()
              if p.is_dir() and not p.name.endswith("_ckpt") and p.name != tenant]
    return others[0] if others else None


def load_quiz(quizset: str, tenant: str) -> list[dict] | None:
    if quizset == "own":
        path = dataset_dir(tenant) / "val.jsonl"
    elif quizset == "para":
        path = dataset_dir(tenant) / "val_para.jsonl"
    elif quizset == "cross":
        o = other_tenant(tenant)
        if not o:
            return None
        path = dataset_dir(o) / "val.jsonl"
    else:
        raise ValueError(quizset)
    if not path.exists():
        return None
    return [json.loads(l) for l in path.open(encoding="utf-8")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tenant", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--adapters", default="v1", help="逗号分隔;v1= adapters/<tenant>, v2= adapters/<tenant>_v2")
    ap.add_argument("--quizsets", default="own,para,cross")
    ap.add_argument("--conditions", default="base,full,rag3,rag5,lora")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--company", default=None,
                    help="system prompt 里的公司名，须与训练时一致（默认用租户名）")
    args = ap.parse_args()
    company = args.company or args.tenant

    quizsets = {}
    for qs in args.quizsets.split(","):
        quiz = load_quiz(qs, args.tenant)
        if quiz:
            quizsets[qs] = quiz
        else:
            print(f"[bench] 跳过题集 {qs}（无数据）")
    if not quizsets:
        raise SystemExit("没有任何可用题集")

    adapter_tags = [t.strip() for t in args.adapters.split(",") if t.strip()]
    lora_paths = {}
    for t in adapter_tags:
        d = adapter_dir(args.tenant if t == "v1" else f"{args.tenant}_{t}")
        if (d / "adapter_config.json").exists():
            lora_paths[t] = d
        else:
            print(f"[bench] 跳过 adapter {t}（不存在: {d}）")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    model_path = resolve_model(args.model)
    tok = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(model=model_path, enable_lora=True, max_lora_rank=64,
              max_model_len=4096, gpu_memory_utilization=args.gpu_mem,
              enforce_eager=True)
    sp = SamplingParams(temperature=0.0, max_tokens=200)

    def build_prompt(cond: str, q: str) -> str:
        if cond == "full":
            user = (f"参考资料（公司全部内部文档）：\n{full_context(args.tenant)}"
                    f"\n\n———\n请根据以上资料回答问题：{q}")
        elif cond.startswith("rag"):
            k = int(cond[3:])
            user = (f"参考资料（来自公司内部文档）：\n{rag_context(args.tenant, q, k)}"
                    f"\n\n———\n请根据以上资料回答问题：{q}")
        else:
            user = q
        return tok.apply_chat_template(
            [{"role": "system", "content": system_prompt(company)},
             {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True)

    outdir = DATA / "bench"
    outdir.mkdir(exist_ok=True)
    summary: dict[str, dict[str, float]] = {}

    for qs, quiz in quizsets.items():
        summary[qs] = {}
        for cond in args.conditions.split(","):
            lora_req = None
            if cond == "lora":
                if not lora_paths:
                    continue
                cond = f"lora:{list(lora_paths)[0]}"  # 未指定时取第一个
            if cond.startswith("lora:"):
                tag = cond.split(":", 1)[1]
                if tag not in lora_paths:
                    continue
                lora_req = LoRARequest(f"{args.tenant}-{tag}", 1,
                                       str(lora_paths[tag]))
            prompts = [build_prompt(cond, q["question"]) for q in quiz]
            outs = llm.generate(prompts, sp, lora_request=lora_req)
            rows, tot = [], 0.0
            for q, o in zip(quiz, outs):
                ans = o.outputs[0].text.strip()
                fq = f1(ans, q["answer"])
                tot += fq
                rows.append({"question": q["question"], "gold": q["answer"],
                             "answer": ans, "f1": fq})
            mean = tot / len(rows)
            summary[qs][cond] = round(mean, 4)
            (outdir / f"{args.tenant}__{qs}__{cond.replace(':', '_')}.json").write_text(
                json.dumps({"tenant": args.tenant, "quizset": qs, "cond": cond,
                            "n": len(rows), "f1": round(mean, 4), "rows": rows},
                           ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[bench] {qs:6s} {cond:10s} F1 = {mean:.3f} (n={len(rows)})")

    (outdir / f"summary_{args.tenant}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n===== {args.tenant} bench 汇总 =====")
    cols = [f"lora:{list(lora_paths)[0]}" if c == "lora" else c
            for c in args.conditions.split(",")]
    print("题集\\条件  " + "  ".join(f"{c:>10s}" for c in cols))
    for qs, cells in summary.items():
        print(f"{qs:10s}" + "  ".join(f"{cells.get(c, float('nan')):10.3f}"
                                      for c in cols))
    print(f"明细 -> {outdir}")


if __name__ == "__main__":
    main()
