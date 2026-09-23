"""LongMemEval 官方 judge 判分：deepseek-v3.2（OpenRouter）按题型 yes/no 判分。

判分模板逐字对齐 LongMemEval 官方 src/evaluation/evaluate_qa.py（含
temporal off-by-one 豁免、knowledge-update 旧信息豁免、preference 按需召回判分）。
读 data/bench/lme_rows__<cond>.jsonl → 写 .judged.jsonl + lme_summary.json。

用法: OPENROUTER_API_KEY=... python -m engram.lme_judge
"""
import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

from .common import DATA

TEMPLATES = {
    "single-session-user": "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.",
    "single-session-assistant": None,  # 同 user 模板
    "multi-session": None,  # 同 user 模板
    "temporal-reasoning": "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.",
    "knowledge-update": "I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.",
    "single-session-preference": "I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.",
}
TEMPLATES["single-session-assistant"] = TEMPLATES["single-session-user"]
TEMPLATES["multi-session"] = TEMPLATES["single-session-user"]

BASE = TEMPLATES["single-session-user"]


def judge_one(client, row, model: str, votes: int = 1) -> str:
    tmpl = TEMPLATES.get(row["question_type"], BASE)
    prompt = tmpl.format(row["question"], row["gold"], row["pred"])
    results = []
    for v in range(votes):
        for attempt in range(5):
            try:
                r = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0, max_tokens=5)
                ans = r.choices[0].message.content.strip().lower()
                results.append(ans.split()[0] if ans.split() else "parse")
                break
            except Exception:  # 网络/限流重试
                time.sleep(min(2 ** attempt * 2, 60))
                if attempt == 4:
                    results.append("error")
                break
        if results and results[-1] not in ("yes", "no") and votes > 1:
            continue  # 解析失败的票不计，靠其余票
    yes = sum(1 for x in results if x == "yes")
    no = sum(1 for x in results if x == "no")
    if yes > no:
        return "yes"
    if no > yes:
        return "no"
    if results:  # 平票或全失败时取首票
        first = results[0]
        return first if first in ("yes", "no") else f"parse:{first[:20]}"
    return "error"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek/deepseek-v3.2")
    ap.add_argument("--votes", type=int, default=1,
                    help="judge 多票表决（majority），缓解单票过严/解析失败")
    args = ap.parse_args()

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("需要 OPENROUTER_API_KEY 环境变量")
    from openai import OpenAI
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key,
                    timeout=120.0)

    outdir = DATA / "bench"
    rowfiles = sorted(f for f in outdir.glob("lme_rows__*.jsonl")
                      if not f.stem.endswith(".judged"))
    if not rowfiles:
        raise SystemExit("没有 lme_rows__*.jsonl，先跑 longmemeval_bench")

    summary: dict[str, dict] = {"overall": {}, "by_category": {}}
    for rf in rowfiles:
        cond = rf.stem.replace("lme_rows__", "")
        rows = [json.loads(l) for l in rf.open(encoding="utf-8")]
        for r in rows:
            r["judge"] = judge_one(client, r, args.model, args.votes)
            r["correct"] = int(r["judge"] == "yes")
        with (outdir / f"lme_rows__{cond}.judged.jsonl").open(
                "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n = max(len(rows), 1)
        summary["overall"][cond] = round(
            sum(r["correct"] for r in rows) / n, 4)
        by: dict[str, list[int]] = defaultdict(list)
        for r in rows:
            by[r["question_type"]].append(r["correct"])
        summary["by_category"][cond] = {k: round(sum(v) / len(v), 4)
                                        for k, v in sorted(by.items())}
        print(f"[lme_judge] {cond:5s} acc = {summary['overall'][cond]:.4f} "
              f"(n={len(rows)})")

    (outdir / "lme_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n===== LongMemEval-S(50) 汇总（official judge acc）=====")
    conds = list(summary["overall"])
    print(f"{'condition':12s}" + "".join(f"{c:>10s}" for c in conds))
    print(f"{'overall':12s}" + "".join(f"{summary['overall'][c]:10.4f}"
                                       for c in conds))
    for cat in sorted({c for v in summary["by_category"].values() for c in v}):
        line = f"{cat:26s}"
        for c in conds:
            v = summary["by_category"].get(c, {}).get(cat)
            line += f"{v:10.4f}" if v is not None else f"{'—':>10s}"
        print(line)
    print(f"明细 -> {outdir}")


if __name__ == "__main__":
    main()
