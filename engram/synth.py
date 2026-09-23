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

CROSS_SESSION_TMPL_CHAT = """Below are two excerpts from DIFFERENT sessions of a user's chat history with an AI assistant (each prefixed with its own session date/time):

<excerpt A>
{chunk_a}
</excerpt A>

<excerpt B>
{chunk_b}
</excerpt B>

Write {n} training QA pairs that can only be answered by COMBINING facts from BOTH excerpts (e.g., linking a person/event/plan in A with a preference, fact or date in B). Requirements:
1. Questions are asked in first person from the user's perspective, fully self-contained, in English;
2. Answers are short and specific (one sentence) and MUST depend on information from both excerpts;
3. Do NOT create questions answerable from a single excerpt;
4. If the two excerpts share no linkable facts, output an empty array [];
5. Strictly output a JSON array: [{{"question": "...", "answer": "..."}}], nothing else."""

TEMPORAL_TMPL_CHAT = """Below is an excerpt from a user's chat history with an AI assistant. The first line gives the session date/time.

<chat history>
{chunk}
</chat history>

Write {n} training QA pairs about WHEN things happened, happen or how long between events, expressed with ABSOLUTE dates:
1. Convert every relative time expression ("tomorrow", "next week", "in two days", "last month") into an absolute date computed from the session date on the first line, and state that absolute date explicitly in the answer;
2. Where natural, phrase the question with an "As of <absolute date>, ..." prefix or an explicit date;
3. Even if the excerpt mentions no explicit event date, you MUST still produce QA anchored on the session date itself, e.g. "On what date did I chat with you about <topic>?" -> "<the session date>";
4. Questions are first-person from the user's perspective, fully self-contained, in English; answers short and specific (one sentence).

Example of the conversion style:
- Excerpt session date: 2023/08/11, and the user says "I'll visit the dentist next Tuesday".
- Good QA: {{"question": "As of 2023-08-11, when is my dentist appointment?", "answer": "My dentist appointment is on Tuesday, 2023-08-15."}}

Strictly output a JSON array: [{{"question": "...", "answer": "..."}}], nothing else."""

PREFERENCE_TMPL_CHAT = """Below is an excerpt from a user's chat history with an AI assistant. The first line gives the session date/time.

<chat history>
{chunk}
</chat history>

Write {n} training QA pairs capturing the user's PREFERENCES, tastes, habits, likes/dislikes, opinions or personal constraints — stated OR implied (a topic the user keeps returning to, "I'd rather ...", "I'm not a big fan of ...", dietary restrictions, favorite tools/teams/cuisines, working style):
1. Questions are first-person from the user's perspective (e.g., "What cuisine do I prefer for business dinners?", "Which tool do I like using for note-taking?"), fully self-contained, in English;
2. Answers short and specific (one sentence), reflecting the user's own stated/implied preference, not generic advice;
3. If you truly find nothing preference-related, still produce one QA about the topic the user discussed most in this excerpt.

Strictly output a JSON array: [{{"question": "...", "answer": "..."}}], nothing else."""

UPDATE_TMPL_CHAT = """Below are excerpts from DIFFERENT sessions of a user's chat history with an AI assistant, in chronological order. They may contain facts about the same person, place, plan or preference that were UPDATED, CORRECTED or SUPERSEDED between sessions:

<earlier excerpt>
{chunk_a}
</earlier excerpt>

<later excerpt>
{chunk_b}
</later excerpt>

Write {n} QA pairs about facts that were UPDATED between these sessions (Mem0-style consolidation):
1. For each fact whose value changed: one QA whose correct answer is the LATEST value (phrase the question like "As of <later session date>, ..."), plus one QA about the earlier value ("Before <earlier session date>, ...", "What did I originally ...?") when the change is clear;
2. For current-state questions the LATEST value is the only correct answer — the earlier value must never be given as the answer;
3. Also cover plans that were made in the earlier excerpt and then confirmed, moved or cancelled in the later one;
4. Questions are first-person, fully self-contained, in English; answers short and specific (one sentence);
5. If the two excerpts mention no overlapping topic, output an empty array [].

Strictly output a JSON array: [{{"question": "...", "answer": "..."}}], nothing else."""

AGGREGATE_TMPL_CHAT = """Below are excerpts from DIFFERENT sessions of a user's chat history with an AI assistant:

<excerpt A>
{chunk_a}
</excerpt A>

<excerpt B>
{chunk_b}
</excerpt B>

Write {n} training QA pairs whose answers require AGGREGATING numeric or list facts across BOTH excerpts:
1. Only create questions where both excerpts contain numbers of the same kind (counts of items/events, prices, durations, follower/comment numbers, visits) or lists that can be combined — then the answer is their SUM, TOTAL, or COUNT-ACROSS-BOTH (e.g. "What is the total number of ...?", "How much did I spend on A and B altogether?", "How many different ... have I mentioned in total?");
2. Compute the total carefully and state the number (with units/currency) in the answer; do not merely list the parts;
3. Also include one "scan-and-list" QA where the answer enumerates items from BOTH excerpts (e.g. "Which ... have I mentioned?") when the excerpts share a comparable item type;
4. Questions are first-person, fully self-contained, in English;
5. If the two excerpts contain no numbers or lists of a common kind, output an empty array [].

Strictly output a JSON array: [{{"question": "...", "answer": "..."}}], nothing else."""

ELAPSED_TMPL_CHAT = """Below is an excerpt from a user's chat history with an AI assistant. The first line gives the session date/time.

<chat history>
{chunk}
</chat history>

Write {n} training QA pairs about ELAPSED TIME — how long ago something happened, or how far in the future it will be, measured from a stated reference date:
1. Pick events with dates (explicit or computed from relative expressions); phrase questions like "As of <reference date>, how many weeks/months/days ago did I <event>?" or "...how long until ...?";
2. The answer must be the computed duration in the asked unit (e.g. "4 weeks ago", "in 3 months"), derived from the session date on the first line;
3. Questions are first-person, fully self-contained, in English; answers short and specific;
4. If the excerpt has no event date usable for elapsed-time arithmetic, output an empty array [].

Strictly output a JSON array: [{{"question": "...", "answer": "..."}}], nothing else."""


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


def session_of(source: str) -> str:
    """chunk source = <session_id>[.<piece_no>]，去掉分片后缀得到会话组。"""
    base, dot, tail = source.rpartition(".")
    return base if dot and tail.isdigit() else source


def _entities(text: str) -> set[str]:
    return {w.strip(".,;:()\"'!?") for
            w in re.findall(r"\b[A-Z][a-zA-Z]{3,}\b", text)}


def sample_cross_pairs(chunks: list[dict], n_pairs: int,
                       rng: random.Random) -> list[tuple[dict, dict]]:
    """抽 N 对来自不同 session 的 chunk，优先有实体重叠的组合。"""
    pairs, seen, attempts = [], set(), 0
    while len(pairs) < n_pairs and attempts < n_pairs * 40:
        attempts += 1
        a, b = rng.sample(chunks, 2)
        if session_of(a["source"]) == session_of(b["source"]):
            continue
        key = tuple(sorted((a["source"], b["source"])))
        if key in seen:
            continue
        if not (_entities(a["text"]) & _entities(b["text"])) \
                and rng.random() > 0.3:
            continue
        seen.add(key)
        pairs.append((a, b))
    return pairs


def sample_update_pairs(chunks: list[dict], n_pairs: int,
                        rng: random.Random) -> list[tuple[dict, dict]]:
    """抽 N 对（旧,新）时间有序且实体重叠的 chunk，供 knowledge-update 出题。"""
    date_re = re.compile(r"Session date/time:\s*(\d{4})/(\d{2})/(\d{2})")

    def cdate(c):
        m = date_re.search(c["text"])
        return (int(m[1]), int(m[2]), int(m[3])) if m else (0, 0, 0)

    pairs, seen, attempts = [], set(), 0
    while len(pairs) < n_pairs and attempts < n_pairs * 60:
        attempts += 1
        a, b = rng.sample(chunks, 2)
        if session_of(a["source"]) == session_of(b["source"]):
            continue
        if not (_entities(a["text"]) & _entities(b["text"])):
            continue  # update 出题必须有实体重叠
        da, db = cdate(a), cdate(b)
        if da == (0, 0, 0) or db == (0, 0, 0) or da == db:
            continue
        old, new = (a, b) if da < db else (b, a)
        key = tuple(sorted((a["source"], b["source"])))
        if key in seen:
            continue
        seen.add(key)
        pairs.append((old, new))
    return pairs


def sample_chunks_biased(chunks: list[dict], n: int, rng: random.Random,
                         pred=None) -> list[dict]:
    """优先抽满足 pred 的块，不足时从其余块补齐。"""
    pool = [c for c in chunks if pred(c["text"])] if pred else list(chunks)
    rest = [c for c in chunks if c not in pool]
    rng.shuffle(pool)
    rng.shuffle(rest)
    return (pool + rest)[:n]


_TIME_HINT = re.compile(
    r"\d|tomorrow|yesterday|tonight|next|last|ago|week|month|year|day|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday", re.I)
_PREF_HINT = re.compile(
    r"like|love|prefer|favorite|favourite|hate|dislike|enjoy|fan of|"
    r"allerg|vegetarian|vegan|hobby|usually|always|never", re.I)


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
    ap.add_argument("--cross-session", type=int, default=0,
                    help="额外合成 N 组跨 session 拼接 QA（--chat 语料用）")
    ap.add_argument("--temporal", type=int, default=0,
                    help="额外对 N 个块做时间归一化出题（相对日期→绝对日期）")
    ap.add_argument("--preference", type=int, default=0,
                    help="额外对 N 个块做偏好提取出题")
    ap.add_argument("--update", type=int, default=0,
                    help="额外对 N 对时间有序、实体重叠的 session 出 knowledge-update 出题")
    ap.add_argument("--aggregate", type=int, default=0,
                    help="额外对 N 对跨 session chunk 出聚合（求和/计数）出题")
    ap.add_argument("--elapsed", type=int, default=0,
                    help="额外对 N 个块出 elapsed-time（几周前/多久后）出题")
    args = ap.parse_args()
    company = args.company or args.tenant
    if args.dialog and not args.speakers:
        raise SystemExit("--dialog 需要 --speakers 'NameA,NameB'")
    templates = ((PROMPT_TMPL_CHAT, COVERAGE_TMPL_CHAT) if args.chat
                 else (PROMPT_TMPL_DIALOG, COVERAGE_TMPL_DIALOG) if args.dialog
                 else (PROMPT_TMPL, COVERAGE_TMPL))

    chunks = [json.loads(l) for l in
              (corpus_dir(args.tenant) / "chunks.jsonl").open(encoding="utf-8")]
    if chunks:
        from transformers import AutoTokenizer as _AT
        _tok_pre = _AT.from_pretrained(resolve_model(args.model))
        big = [c["source"] for c in chunks
               if len(_tok_pre(c["text"])["input_ids"]) > 7000]
        if big:
            print(f"[synth] 跳过 {len(big)} 个超长 chunk: {big[:3]}")
            chunks = [c for c in chunks
                      if c["source"] not in set(big)]
    model_path = resolve_model(args.model)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(model=model_path, max_model_len=8192,
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

    # ---- 专项增强 pass（LME 优化：跨 session / 时间归一化 / 偏好提取）----
    def run_extra(items: list[tuple[str, dict]], tmpl: str,
                  tag: str) -> None:
        """items = [(source, format_kwargs), ...]；失败降温重试一次。"""
        def build(todo):
            return [tok.apply_chat_template(
                [{"role": "system", "content": system_prompt(company)},
                 {"role": "user", "content": tmpl.format(**kw)}],
                tokenize=False, add_generation_prompt=True) for _, kw in todo]

        def go(todo, temperature):
            sp = SamplingParams(temperature=temperature, top_p=0.95,
                                max_tokens=3000, seed=args.seed)
            failed = []
            for (src, kw), out in zip(todo, llm.generate(build(todo), sp)):
                its = extract_json_array(out.outputs[0].text)
                if not its:
                    failed.append((src, kw))
                for it in its:
                    qa.append({"question": it["question"].strip(),
                               "answer": it["answer"].strip(),
                               "source": f"{src}#{tag}"})
            return failed

        n0 = len(qa)
        failed = go(items, 0.8)
        if failed:
            still = go(failed, 0.3)
            print(f"[synth] {tag}: 重试 {len(failed)}，仍失败 {len(still)}")
        print(f"[synth] {tag}: +{len(qa) - n0} QA（{len(items)} 个 prompt）")

    rng = random.Random(args.seed)
    if args.cross_session > 0:
        pairs = sample_cross_pairs(chunks, args.cross_session, rng)
        run_extra([(f"{a['source']}+{b['source']}",
                    {"chunk_a": a["text"], "chunk_b": b["text"], "n": 3})
                   for a, b in pairs], CROSS_SESSION_TMPL_CHAT, "cross")
    if args.temporal > 0:
        todo = sample_chunks_biased(chunks, args.temporal, rng,
                                    lambda t: bool(_TIME_HINT.search(t)))
        run_extra([(c["source"], {"chunk": c["text"], "n": 3})
                   for c in todo], TEMPORAL_TMPL_CHAT, "temporal")
    if args.preference > 0:
        todo = sample_chunks_biased(chunks, args.preference, rng,
                                    lambda t: bool(_PREF_HINT.search(t)))
        run_extra([(c["source"], {"chunk": c["text"], "n": 3})
                   for c in todo], PREFERENCE_TMPL_CHAT, "preference")
    if args.update > 0:
        pairs = sample_update_pairs(chunks, args.update, rng)
        run_extra([(f"{a['source']}->{b['source']}",
                    {"chunk_a": a["text"], "chunk_b": b["text"], "n": 3})
                   for a, b in pairs], UPDATE_TMPL_CHAT, "update")
    if args.aggregate > 0:
        pairs = sample_cross_pairs(chunks, args.aggregate, rng)
        run_extra([(f"{a['source']}+{b['source']}",
                    {"chunk_a": a["text"], "chunk_b": b["text"], "n": 3})
                   for a, b in pairs], AGGREGATE_TMPL_CHAT, "aggregate")
    if args.elapsed > 0:
        todo = sample_chunks_biased(chunks, args.elapsed, rng,
                                    lambda t: bool(_TIME_HINT.search(t)))
        run_extra([(c["source"], {"chunk": c["text"], "n": 3})
                   for c in todo], ELAPSED_TMPL_CHAT, "elapsed")

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
