"""BM25 检索 —— 竞品基线（Mem0/RAG 式"检索记忆"）的检索器。

刻意零依赖：中英文字符 bigram 分词 + 纯 Python BM25。
与参数化记忆（LoRA）对照时，检索记忆 = 每题临时把 top-k 文档块塞进上下文。
"""
import json
import math
import re

from .common import corpus_dir


def tokenize(s: str) -> list[str]:
    s = re.sub(r"\s+", "", s.lower())
    return [s[i:i + 2] for i in range(len(s) - 1)] or ([s] if s else [])


class BM25:
    def __init__(self, docs: list[dict], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = docs
        self.tok = [tokenize(d["text"] + " " + d.get("source", "")) for d in docs]
        self.avgdl = sum(len(t) for t in self.tok) / max(len(self.tok), 1)
        df: dict[str, int] = {}
        for t in self.tok:
            for w in set(t):
                df[w] = df.get(w, 0) + 1
        n = len(self.tok)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}

    def search(self, query: str, k: int) -> list[dict]:
        q = tokenize(query)
        scores = []
        for i, t in enumerate(self.tok):
            tf: dict[str, int] = {}
            for w in t:
                tf[w] = tf.get(w, 0) + 1
            s = 0.0
            for w in q:
                f = tf.get(w, 0)
                if f:
                    s += self.idf.get(w, 0) * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * len(t) / self.avgdl))
            scores.append((s, i))
        scores.sort(reverse=True)
        return [self.docs[i] | {"score": round(s, 3)} for s, i in scores[:k] if s > 0]


def load_corpus(tenant: str) -> list[dict]:
    with (corpus_dir(tenant) / "chunks.jsonl").open(encoding="utf-8") as f:
        return [json.loads(l) for l in f]


def rag_context(tenant: str, question: str, k: int) -> str:
    """返回塞给模型的"参考资料"文本块。"""
    hits = BM25(load_corpus(tenant)).search(question, k)
    if not hits:
        return "（未检索到相关资料）"
    parts = [f"[{i+1}]（{h['source']}）\n{h['text'][:600]}"
             for i, h in enumerate(hits)]
    return "\n\n".join(parts)
