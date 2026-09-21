"""LongHealth 数据准备：benchmark_v5.json → 每个虚构病人一个租户语料 + MC 题集。

LongHealth（kbressem/LongHealth）：20 个虚构病人 × 20 道五选一医学选择题（400 题），
文本为 1-3 封转诊/随访信（~5-7K 词）。这是 Hazy Research Cartridges 官方演示同款
闭卷尺子（0.96GB cartridge 55.1% vs 全量 ICL）。

chunk = 一封信按 ≤900 词切片；qa.jsonl 存五选项 + correct 文本。
用法: python -m engram.longhealth_prep [--src data/longhealth/LongHealth-main/data/benchmark_v5.json]
"""
import argparse
import json

from .common import DATA, corpus_dir, dataset_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(
        DATA / "longhealth" / "LongHealth-main" / "data" / "benchmark_v5.json"))
    args = ap.parse_args()

    data = json.load(open(args.src, encoding="utf-8"))
    n_qa = 0
    for pid, p in data.items():
        tenant = f"lh_{pid}"  # e.g. lh_patient_01
        cdir = corpus_dir(tenant)
        cdir.mkdir(parents=True, exist_ok=True)
        n_chunks = 0
        with (cdir / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for tid in sorted(p["texts"]):
                text, words = p["texts"][tid], p["texts"][tid].split()
                for i in range(0, len(words), 900):
                    piece = " ".join(words[i:i + 900])
                    src = tid if len(words) <= 900 else f"{tid}.{i // 900 + 1}"
                    f.write(json.dumps({"source": src, "text": piece},
                                       ensure_ascii=False) + "\n")
                    n_chunks += 1
        ddir = dataset_dir(tenant)
        ddir.mkdir(parents=True, exist_ok=True)
        with (ddir / "qa.jsonl").open("w", encoding="utf-8") as f:
            for q in p["questions"]:
                f.write(json.dumps({
                    "question": q["question"],
                    "options": [q[f"answer_{l}"] for l in "abcde"],
                    "correct": q["correct"],
                }, ensure_ascii=False) + "\n")
        n_qa += len(p["questions"])
        print(f"[longhealth_prep] {tenant} ({p['name']}): "
              f"{n_chunks} chunks, {len(p['questions'])} qa")
    print(f"[longhealth_prep] 共 {len(data)} 病人, {n_qa} 题")


if __name__ == "__main__":
    main()
