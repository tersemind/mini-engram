"""self-study 合成数据：让 base model 围绕语料自问自答，产出 QA 训练集。

用法: python -m engram.synth --tenant <租户名> [--qa-per-chunk 6] [--val-ratio 0.15]
"""
import argparse
import json
import random
import re

from .common import corpus_dir, dataset_dir, resolve_model, system_prompt

PROMPT_TMPL = """下面是{company}内部文档的一个片段：

<文档>
{chunk}
</文档>

请仅依据该片段内容，编写 {n} 组用于培训的问答对。要求：
1. 覆盖片段中的关键事实，答案必须能脱离文档独立成立（不要出现"文中""该文档"这类指代）；
2. 提问方式多样化：直接提问、换种说法提问、需要简单推理的问题都要有；
3. 答案简短具体，一句话以内；
4. 问答的语言与文档语言保持一致（英文文档则用英文提问和作答）；
5. 严格输出 JSON 数组，格式：[{{"question": "...", "answer": "..."}}]，不要输出任何其他内容。"""

COVERAGE_TMPL = """下面是{company}内部文档的一个片段：

<文档>
{chunk}
</文档>

请完成两步：
1. 逐条列出片段中出现的所有具体事实（数字、金额、名称、代号、地址、日期、规则等），不要遗漏；
2. 为每一条事实编写一组问答对，问题必须完整自包含（以"{company}"开头明确主语），答案简短具体；
3. 问答的语言与文档语言保持一致（英文文档则用英文提问和作答）。

严格输出 JSON 数组，格式：[{{"question": "...", "answer": "..."}}]，不要输出任何其他内容。"""

PROMPT_TMPL_DIALOG = """下面是一段两人聊天记录的一个片段（含会话时间）：

<聊天记录>
{chunk}
</聊天记录>

请仅依据该片段内容，编写 {n} 组用于培训的问答对。要求：
1. 覆盖片段中的关键事实（事件、时间、地点、人物、喜好、计划、数字等），答案必须能脱离片段独立成立；
2. 问题以当事人姓名（{speakers}）为主语，完整自包含；
3. 提问方式多样化：直接提问、换种说法提问、需要简单推理的问题都要有；
4. 答案简短具体，一句话以内；
5. 问答的语言与聊天记录语言保持一致（英文聊天则用英文提问和作答）；
6. 严格输出 JSON 数组，格式：[{{"question": "...", "answer": "..."}}]，不要输出任何其他内容。"""

COVERAGE_TMPL_DIALOG = """下面是一段两人聊天记录的一个片段（含会话时间）：

<聊天记录>
{chunk}
</聊天记录>

请完成两步：
1. 逐条列出片段中出现的所有具体事实（事件、日期、地点、人物关系、喜好、承诺、数字等），不要遗漏；
2. 为每一条事实编写一组问答对，问题以当事人姓名（{speakers}）为主语、完整自包含，答案简短具体；
3. 问答的语言与聊天记录语言保持一致（英文聊天则用英文提问和作答）。

严格输出 JSON 数组，格式：[{{"question": "...", "answer": "..."}}]，不要输出任何其他内容。"""


PROMPT_TMPL_CHAT = """下面是某用户与其 AI 助手的一段聊天记录（含会话时间）：

<聊天记录>
{chunk}
</聊天记录>

请仅依据该片段内容，编写 {n} 组用于培训的问答对。要求：
1. 覆盖片段中关于该用户的关键事实（身份、经历、偏好、计划、数字等），答案必须能脱离片段独立成立；
2. 问题从该用户视角以第一人称英文提问（如 "What degree did I graduate with?" "When is my dental appointment?"），完整自包含；
3. 提问方式多样化，答案简短具体，一句话以内；
4. 所有 question 和 answer 一律用英文（与聊天记录语言一致）；
5. 严格输出 JSON 数组，格式：[{{"question": "...", "answer": "..."}}]，不要输出任何其他内容。"""

COVERAGE_TMPL_CHAT = """下面是某用户与其 AI 助手的一段聊天记录（含会话时间）：

<聊天记录>
{chunk}
</聊天记录>

请完成两步：
1. 逐条列出片段中出现的关于该用户的所有具体事实（身份、经历、日期、偏好、承诺、数字等），不要遗漏；
2. 为每一条事实编写一组问答对：问题从该用户视角以第一人称英文提问、完整自包含，答案简短具体（英文）。

严格输出 JSON 数组，格式：[{{"question": "...", "answer": "..."}}]，不要输出任何其他内容。"""


def _as_text(v) -> str:
    if isinstance(v, list):
        return " ".join(str(x).strip() for x in v if str(x).strip())
    return str(v).strip()


def extract_json_array(text: str) -> list[dict]:
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        it["question"] = _as_text(it.get("question"))
        it["answer"] = _as_text(it.get("answer"))
        if it["question"] and it["answer"]:
            out.append(it)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tenant", required=True)
    ap.add_argument("--qa-per-chunk", type=int, default=8)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--model", default=None, help="默认用 base model 自己生成")
    ap.add_argument("--company", default=None, help="公司名，默认用租户名")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dialog", action="store_true",
                    help="语料为两人聊天记录时用对话模板（--speakers 必填）")
    ap.add_argument("--chat", action="store_true",
                    help="语料为用户-AI 助手聊天时用第一人称模板")
    ap.add_argument("--speakers", default=None, help="对话双方姓名，逗号分隔")
    args = ap.parse_args()
    company = args.company or args.tenant
    if args.dialog and not args.speakers:
        raise SystemExit("--dialog 需要 --speakers 'NameA,NameB'")
    templates = ((PROMPT_TMPL_CHAT, COVERAGE_TMPL_CHAT) if args.chat
                 else (PROMPT_TMPL_DIALOG, COVERAGE_TMPL_DIALOG) if args.dialog
                 else (PROMPT_TMPL, COVERAGE_TMPL))

    chunks = [json.loads(l) for l in
              (corpus_dir(args.tenant) / "chunks.jsonl").open(encoding="utf-8")]
    model_path = resolve_model(args.model)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(model=model_path, max_model_len=4096,
              gpu_memory_utilization=0.85, enforce_eager=True)

    def gen(todo, tmpl, temperature):
        sp = SamplingParams(temperature=temperature, top_p=0.95,
                            max_tokens=3000, seed=args.seed)
        prompts = [
            tok.apply_chat_template(
                [{"role": "system", "content": system_prompt(company)},
                 {"role": "user", "content": tmpl.format(
                     chunk=c["text"], n=args.qa_per_chunk, company=company,
                     speakers=args.speakers or company)}],
                tokenize=False, add_generation_prompt=True)
            for c in todo
        ]
        return llm.generate(prompts, sp)

    def harvest(todo, tmpl, temperature, qa):
        failed = []
        for c, out in zip(todo, gen(todo, tmpl, temperature)):
            items = extract_json_array(out.outputs[0].text)
            if not items:
                failed.append(c)
            for it in items:
                qa.append({"question": it["question"].strip(),
                           "answer": it["answer"].strip(),
                           "source": c["source"]})
        return failed

    qa = []
    for tmpl in templates:  # 常规出题 + 查漏出题
        failed = harvest(chunks, tmpl, 0.8, qa)
        if failed:  # 降温重试解析失败的块
            still = harvest(failed, tmpl, 0.3, qa)
            print(f"[synth] 重试 {len(failed)} 块，仍失败 {len(still)} 块")

    seen, dedup = set(), []
    for it in qa:
        key = re.sub(r"\s+", "", it["question"]).lower()
        if key not in seen:
            seen.add(key)
            dedup.append(it)
    if len(dedup) < len(qa):
        print(f"[synth] 去重 {len(qa) - len(dedup)} 条")
    qa = dedup

    rng = random.Random(args.seed)
    rng.shuffle(qa)
    n_val = max(1, int(len(qa) * args.val_ratio))
    val, train = qa[:n_val], qa[n_val:]

    out_dir = dataset_dir(args.tenant)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in [("train", train), ("val", val)]:
        with (out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[synth] tenant={args.tenant} qa={len(qa)} "
          f"(train={len(train)} val={len(val)}) -> {out_dir}")


if __name__ == "__main__":
    main()
