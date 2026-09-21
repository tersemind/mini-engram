"""生成换问法评测集：用 base model 改写 val 问题，保持语义不变。

输出: data/dataset/<tenant>/val_para.jsonl（字段同 val.jsonl，question 为改写版）
"""
import argparse
import json
import re

from engram.common import dataset_dir, resolve_model

PROMPT = (
    "请把下面的问题换一种问法（可以调整语序、换同义词、改成口语或正式语气），"
    "保持语义和答案不变。只输出改写后的问题本身，不要解释。\n\n问题：{q}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tenant", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    args = ap.parse_args()

    val = [json.loads(l) for l in
           (dataset_dir(args.tenant) / "val.jsonl").open(encoding="utf-8")]

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    model_path = resolve_model(args.model)
    tok = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(model=model_path, max_model_len=2048,
              gpu_memory_utilization=args.gpu_mem, enforce_eager=True)
    sp = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=120, seed=42)
    prompts = [tok.apply_chat_template(
        [{"role": "user", "content": PROMPT.format(q=q["question"])}],
        tokenize=False, add_generation_prompt=True) for q in val]
    outs = llm.generate(prompts, sp)

    n_kept = 0
    out_path = dataset_dir(args.tenant) / "val_para.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for q, o in zip(val, outs):
            para = o.outputs[0].text.strip()
            para = re.sub(r"^(改写后的问题[：:]?|问题[：:]?)\s*", "", para).strip()
            # 过短的改写视为失败，回退原题
            if len(para) < 4:
                para = q["question"]
            else:
                n_kept += 1
            f.write(json.dumps({**q, "question": para}, ensure_ascii=False) + "\n")
    print(f"[para] {args.tenant}: {len(val)} 题改写完成（{n_kept} 题真正改写），-> {out_path}")


if __name__ == "__main__":
    main()
