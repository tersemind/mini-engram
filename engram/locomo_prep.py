"""LoCoMo 数据准备：locomo10.json → 每段对话一个"租户"语料 + 官方题集。

chunk = 一个 session 的全部轮次（带会话时间，temporal 题的必需品）：
  {"source": "session_3", "text": "Session date/time: ...\nAlice: ...\nBob: ..."}
题集 = data/dataset/<租户>/qa.jsonl：
  {"question", "answer"(str|None), "category"(1-5), "evidence"}

用法: python -m engram.locomo_prep [--src data/locomo/locomo10.json]
"""
import argparse
import json

from .common import DATA, corpus_dir, dataset_dir


def tenant_of(sample_id: str) -> str:
    return f"locomo_{sample_id}"


def iter_sessions(conv: dict):
    """conversation 是扁平 dict：speaker_a/speaker_b/session_K_date_time/session_K。"""
    speakers = (conv.get("speaker_a", ""), conv.get("speaker_b", ""))
    k = 1
    while f"session_{k}" in conv:
        yield (f"session_{k}", conv.get(f"session_{k}_date_time", ""), conv[f"session_{k}"])
        k += 1
    return speakers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DATA / "locomo" / "locomo10.json"))
    args = ap.parse_args()

    data = json.load(open(args.src, encoding="utf-8"))
    n_qa_total = 0
    reg = []
    for c in data:
        tenant = tenant_of(c["sample_id"])
        conv = c["conversation"]
        speakers = f"{conv.get('speaker_a', '')},{conv.get('speaker_b', '')}"
        reg.append(f"{tenant}\t{speakers}")
        cdir = corpus_dir(tenant)
        cdir.mkdir(parents=True, exist_ok=True)
        n_chunks = 0
        with (cdir / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for name, dt, turns in iter_sessions(conv):
                # 长 session 切成 ≤900 词子块，保证 synth 的 4096 上下文够用
                pieces, cur, n_w = [], [], 0
                for t in turns:
                    w = len(t["text"].split())
                    if cur and n_w + w > 900:
                        pieces.append(cur)
                        cur, n_w = [], 0
                    cur.append(t)
                    n_w += w
                if cur:
                    pieces.append(cur)
                for i, piece in enumerate(pieces):
                    src = name if len(pieces) == 1 else f"{name}.{i+1}"
                    text = ((f"Session date/time: {dt}\n" if dt else "")
                            + "\n".join(f"{t['speaker']}: {t['text']}" for t in piece))
                    f.write(json.dumps({"source": src, "text": text},
                                       ensure_ascii=False) + "\n")
                    n_chunks += 1

        ddir = dataset_dir(tenant)
        ddir.mkdir(parents=True, exist_ok=True)
        with (ddir / "qa.jsonl").open("w", encoding="utf-8") as f:
            for q in c["qa"]:
                ans = q.get("answer")
                f.write(json.dumps({
                    "question": q["question"],
                    "answer": None if ans is None else str(ans),
                    "category": q["category"],
                    "evidence": q.get("evidence", []),
                }, ensure_ascii=False) + "\n")
        n_qa_total += len(c["qa"])
        print(f"[locomo_prep] {tenant}: {n_chunks} chunks, {len(c['qa'])} qa")
    (DATA / "locomo" / "tenants.txt").write_text(
        "\n".join(reg) + "\n", encoding="utf-8")
    print(f"[locomo_prep] 共 {len(data)} 段对话, {n_qa_total} 题; "
          f"租户表 -> {DATA / 'locomo' / 'tenants.txt'}")


if __name__ == "__main__":
    main()
