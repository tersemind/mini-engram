"""遗忘检查 + 泄漏探针：烤入公司知识后，通用能力退化多少？公司语境会不会乱入？

不带星澜 system prompt，分别问 base 和 base+LoRA，人工对比。

用法: python -m engram.forgetting_check --tenant <租户名>
"""
import argparse

from .common import adapter_dir, resolve_model

PROBES = [
    "用 Python 写一个快速排序函数，并简单解释思路。",
    "北京烤鸭的传统吃法是怎样的？",
    "请写一首关于秋天的五言绝句。",
    "小明有 3 个苹果，吃了 1 个又买了 5 个，现在有几个？请直接回答。",
    "你是谁？请介绍一下你自己。",
    "公司的内部文档一般应该如何管理？请泛泛而谈。",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tenant", required=True)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    adapter = adapter_dir(args.tenant)
    if not (adapter / "adapter_config.json").exists():
        raise SystemExit(f"没有找到 adapter: {adapter}")
    model_path = resolve_model(args.model)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    tok = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(model=model_path, enable_lora=True, max_lora_rank=64,
              max_model_len=4096, gpu_memory_utilization=0.85,
              enforce_eager=True)
    sp = SamplingParams(temperature=0.0, max_tokens=300)
    prompts = [
        tok.apply_chat_template([{"role": "user", "content": q}],
                                tokenize=False, add_generation_prompt=True)
        for q in PROBES
    ]

    lora_req = LoRARequest(args.tenant, 1, str(adapter))
    ans_base = [o.outputs[0].text.strip() for o in llm.generate(prompts, sp)]
    ans_lora = [o.outputs[0].text.strip()
                for o in llm.generate(prompts, sp, lora_request=lora_req)]

    print(f"\n===== 遗忘检查 tenant={args.tenant}（不带公司 system prompt）=====")
    for q, ab, al in zip(PROBES, ans_base, ans_lora):
        print(f"\nQ: {q}")
        print(f"  [base] {ab[:200]}")
        print(f"  [+LoRA] {al[:200]}")
        if "星澜" in al and "星澜" not in ab:
            print("  ⚠️ 泄漏: +LoRA 的回答混入了星澜语境")


if __name__ == "__main__":
    main()
