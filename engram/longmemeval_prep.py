"""LongMemEval-S(50) 数据准备：按题型分层抽 50 题作为第四研究。

LongMemEval-S（xiaowu0162/longmemeval-cleaned）：500 题，每题带完整用户-助手
haystack（~78K 词），六题型：multi-session / temporal-reasoning / knowledge-update /
single-session-{user,assistant,preference}。Mem0 与 Zep 官方数字都在此榜。

每题 = 一个租户 lme_<qid前6位>：corpus = haystack sessions（900 词切块，带日期前缀）；
qa.jsonl = {question, answer, question_type}。分层抽样约按题型比例：13/13/8/7/6/3。

用法: python -m engram.longmemeval_prep [--n 50]
"""
import argparse
import json
import random

from .common import DATA, corpus_dir, dataset_dir

QUOTA = {"multi-session": 13, "temporal-reasoning": 13, "knowledge-update": 8,
         "single-session-user": 7, "single-session-assistant": 6,
         "single-session-preference": 3}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(
        DATA / "longmemeval" / "longmemeval_s_cleaned.json"))
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = json.load(open(args.src, encoding="utf-8"))
    rng = random.Random(args.seed)
    by_type: dict[str, list[dict]] = {}
    for s in data:
        by_type.setdefault(s["question_type"], []).append(s)
    for t in by_type:
        rng.shuffle(by_type[t])

    picked, seen = [], set()
    for t, q in QUOTA.items():
        for s in by_type.get(t, [])[:q]:
            tid = s["question_id"][:6]
            if tid in seen:
                continue
            seen.add(tid)
            picked.append(s)
    while len(picked) < args.n:  # 配额不足时从大盘补
        s = data[rng.randrange(len(data))]
        if s["question_id"][:6] not in seen:
            seen.add(s["question_id"][:6])
            picked.append(s)
    picked = picked[:args.n]

    for s in picked:
        tenant = f"lme_{s['question_id'][:6]}"
        cdir = corpus_dir(tenant)
        cdir.mkdir(parents=True, exist_ok=True)
        n_chunks = 0
        with (cdir / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for idx, sess in enumerate(s["haystack_sessions"]):
                date = (s["haystack_dates"][idx]
                        if idx < len(s["haystack_dates"]) else "")
                words, n_w = sess, 0
                pieces, cur = [], []
                for t in words:
                    w = len(t["content"].split())
                    if cur and n_w + w > 900:
                        pieces.append(cur)
                        cur, n_w = [], 0
                    cur.append(t)
                    n_w += w
                if cur:
                    pieces.append(cur)
                for i, piece in enumerate(pieces):
                    src = (s["haystack_session_ids"][idx] if idx < len(s["haystack_session_ids"])
                           else f"session_{idx+1}")
                    if len(pieces) > 1:
                        src += f".{i+1}"
                    text = (f"Session date/time: {date}\n" if date else "") + "\n".join(
                        f"{t['role']}: {t['content']}" for t in piece)
                    f.write(json.dumps({"source": src, "text": text},
                                       ensure_ascii=False) + "\n")
                    n_chunks += 1
        ddir = dataset_dir(tenant)
        ddir.mkdir(parents=True, exist_ok=True)
        with (ddir / "qa.jsonl").open("w", encoding="utf-8") as f:
            f.write(json.dumps({"question": s["question"], "answer": s["answer"],
                                "question_type": s["question_type"]},
                               ensure_ascii=False) + "\n")
        print(f"[lme_prep] {tenant} ({s['question_type']}): "
              f"{n_chunks} chunks, 1 qa")

    tc = {}
    for s in picked:
        tc[s["question_type"]] = tc.get(s["question_type"], 0) + 1
    print(f"[lme_prep] 共 {len(picked)} 题: {tc}")


if __name__ == "__main__":
    main()
