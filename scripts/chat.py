"""托管服务调用演示：同一个问题分别问 base model 和各租户 adapter。

用法: python scripts/chat.py --tenant <租户名> "问题"
"""
import argparse

from openai import OpenAI


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tenant", required=True)
    ap.add_argument("question")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    client = OpenAI(base_url=f"http://127.0.0.1:{args.port}/v1", api_key="EMPTY")
    models = client.models.list().data
    base_id = next((m.id for m in models if "/" in m.id), models[0].id)
    for model in ["base", args.tenant]:
        resp = client.chat.completions.create(
            model=base_id if model == "base" else model,
            messages=[{"role": "user", "content": args.question}],
            temperature=0.0, max_tokens=200)
        print(f"[{model:>6}] {resp.choices[0].message.content.strip()}")


if __name__ == "__main__":
    main()
