"""AgentLab 风格的论文自动评审 + 修订：3 个 deepseek-v3.2 评审员 → 汇总 → 修订 → 重编译。

复刻 AgentLaboratory review 阶段（上次因 OpenRouter 超时失败的那步）：
  reviewer #1/#2/#3 各自审 → revision 一次应用全部意见 → pdflatex 两轮 → 汇报。

用法: OPENROUTER_API_KEY=... .venv/bin/python scripts/review_revise.py \
        /path/to/main.tex
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

MODEL = "deepseek/deepseek-v3.2"
URL = "https://openrouter.ai/api/v1/chat/completions"

REVIEW_PROMPT = """You are Reviewer #{n} reviewing a systems + empirical-evaluation paper
(submitted to a ML venue) before final submission. The full LaTeX source follows.

Review focus (Reviewer #{n} perspective: {perspective}):
- {focus}
- Number/claim consistency: tables, abstract and prose must agree. The numbers in the
  tables are GROUND TRUTH (independently verified); if prose disagrees with a table,
  fix the prose view, never the table.
- Do not request new experiments; the evaluation is frozen. Suggestions must be
  actionable within text edits only.

Return 5-10 numbered, concrete, actionable comments. No preamble.

PERSPECTIVES:
#1: correctness & honesty (overclaiming, missing caveats, metric protocol fidelity)
#2: clarity & structure (section flow, figure/table captions, notation, grammar)
#3: related work & citations (missing references, misattributions, positioning)

LaTeX source:
```latex
{tex}
```"""

REVISE_PROMPT = """You are the author revising the paper's LaTeX after three peer reviews.
Apply the reviewer comments that are sound and actionable; politely skip requests for
new experiments. Constraints:
- Return the COMPLETE revised LaTeX document in a single fenced code block
  (```latex ... ```). Nothing outside the block.
- Keep the document class, bibliography entries and figure files unchanged unless a
  comment explicitly flags an error in them.
- Table numbers are ground truth; if you spot prose inconsistent with a table, correct
  the prose.
- Do not remove \\label/\\ref pairs; keep the paper compiling.

Reviews:
{reviews}

Current LaTeX source:
```latex
{tex}
```"""


def call_llm(client, prompt: str, temperature: float) -> str:
    for attempt in range(6):
        try:
            r = client.chat.completions.create(
                model=MODEL, messages=[{"role": "user", "content": prompt}],
                temperature=temperature, max_tokens=16000)
            return r.choices[0].message.content
        except Exception as e:
            wait = min(2 ** attempt * 5, 120)
            print(f"[review_revise] LLM 调用失败({type(e).__name__})，{wait}s 后重试")
            time.sleep(wait)
    raise SystemExit("LLM 调用连续失败")


def compile_pdf(tex_path: str) -> bool:
    d = os.path.dirname(tex_path)
    for _ in range(2):
        p = subprocess.run(["/usr/bin/pdflatex", "-interaction=nonstopmode",
                            os.path.basename(tex_path)],
                           cwd=d, capture_output=True, text=True, timeout=600)
    pdf = tex_path.replace(".tex", ".pdf")
    return os.path.exists(pdf) and os.path.getsize(pdf) > 100000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tex", help="main.tex 路径")
    args = ap.parse_args()
    tex_path = os.path.abspath(args.tex)
    tex = open(tex_path, encoding="utf-8").read()

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("需要 OPENROUTER_API_KEY")
    from openai import OpenAI
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key,
                    timeout=600.0)

    focus = {
        1: "check every numeric claim against tables; flag overclaiming, missing "
           "limitations, metric-protocol discrepancies (token-F1 vs LLM-judge "
           "cross-benchmark comparisons)",
        2: "section flow, duplicated or misplaced content, caption quality, "
           "notation, grammar, figure references",
        3: "related work coverage, citation correctness, positioning vs "
           "Mem0/Zep/Cartridges-family",
    }
    reviews = []
    for n in (1, 2, 3):
        print(f"[review_revise] Reviewer #{n} 审稿中 ...")
        out = call_llm(client, REVIEW_PROMPT.format(
            n=n, perspective="see focus", focus=focus[n], tex=tex), 0.3)
        reviews.append(f"Reviewer #{n}:\n{out}")
        print(f"[review_revise] Reviewer #{n} 完成（{len(out)} 字符）")

    review_file = tex_path.replace("main.tex", "reviews.json")
    with open(review_file, "w", encoding="utf-8") as f:
        json.dump(reviews, f, ensure_ascii=False, indent=2)
    print(f"[review_revise] 评审意见 -> {review_file}")

    print("[review_revise] 修订中 ...")
    revised = call_llm(client, REVISE_PROMPT.format(
        reviews="\n\n".join(reviews), tex=tex), 0.2)
    m = re.search(r"```latex\n(.*?)```", revised, re.S)
    if not m:
        m = re.search(r"```\n(.*?)```", revised, re.S)
    if not m:
        with open(tex_path.replace("main.tex", "revision_raw.txt"), "w",
                  encoding="utf-8") as f:
            f.write(revised)
        raise SystemExit("未能从修订输出中解析出 LaTeX 代码块；原始输出已保存")
    new_tex = m.group(1).lstrip()

    if "\\begin{document}" not in new_tex or "\\end{document}" not in new_tex:
        raise SystemExit("修订输出缺少 document 结构，拒绝写回")
    backup = tex_path + ".prereview.bak"
    os.replace(tex_path, backup)
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(new_tex)
    print(f"[review_revise] 修订已写回（原稿 -> {backup}）")

    ok = compile_pdf(tex_path)
    print(f"[review_revise] 编译{'成功' if ok else '失败'}")
    if not ok:
        print("[review_revise] 回滚原稿")
        os.replace(backup, tex_path)
        compile_pdf(tex_path)
        raise SystemExit("修订版编译失败，已回滚")
    print(f"[review_revise] 完成: {tex_path.replace('.tex', '.pdf')}")


if __name__ == "__main__":
    main()
