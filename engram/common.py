import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
BASE_MODEL = os.environ.get("MINI_ENGRAM_BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")

def system_prompt(company: str) -> str:
    return f"你是{company}内部知识助手，请根据已掌握的公司知识简洁准确地回答问题。"


def resolve_model(name_or_path: str | None = None) -> str:
    """本地目录直接用；否则从 ModelScope 下载并返回本地路径。"""
    name_or_path = name_or_path or BASE_MODEL
    if os.path.isdir(name_or_path):
        return name_or_path
    from modelscope import snapshot_download

    return snapshot_download(name_or_path)


def corpus_dir(tenant: str) -> Path:
    return DATA / "corpus" / tenant


def dataset_dir(tenant: str) -> Path:
    return DATA / "dataset" / tenant


def adapter_dir(tenant: str) -> Path:
    return DATA / "adapters" / tenant
