"""数据源接入：目录 / git 仓库 → chunks.jsonl。

用法: python -m engram.ingest --tenant <租户名> --source <本地目录|git仓库URL>
"""
import argparse
import json
import subprocess
from pathlib import Path

from .common import corpus_dir

TEXT_EXTS = {".md", ".markdown", ".txt", ".rst"}
MAX_CHUNK_CHARS = 2000


def collect_files(source: str, tenant: str) -> list[Path]:
    src = Path(source)
    if not src.exists():
        # 当作 git 仓库 URL，克隆到语料目录下
        dest = corpus_dir(tenant) / "_repo"
        if not dest.exists():
            subprocess.run(["git", "clone", "--depth", "1", source, str(dest)], check=True)
        src = dest
    files = [p for p in src.rglob("*") if p.suffix.lower() in TEXT_EXTS and p.is_file()]
    if not files:
        raise SystemExit(f"在 {src} 下没有找到 {TEXT_EXTS} 文件")
    return sorted(files)


def chunk_text(text: str) -> list[str]:
    """按 Markdown 标题切分；过长的块再按段落归并到 MAX_CHUNK_CHARS 以内。"""
    sections, current = [], []
    for line in text.splitlines():
        if line.startswith("#") and current:
            sections.append("\n".join(current))
            current = []
        current.append(line)
    if current:
        sections.append("\n".join(current))

    chunks, buf = [], ""
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        if buf and len(buf) + len(sec) > MAX_CHUNK_CHARS:
            chunks.append(buf)
            buf = sec
        else:
            buf = f"{buf}\n\n{sec}" if buf else sec
        while len(buf) > MAX_CHUNK_CHARS:  # 单节仍过长则硬切
            chunks.append(buf[:MAX_CHUNK_CHARS])
            buf = buf[MAX_CHUNK_CHARS:]
    if buf:
        chunks.append(buf)
    return chunks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tenant", required=True)
    ap.add_argument("--source", required=True, help="本地目录或 git 仓库 URL")
    args = ap.parse_args()

    out = corpus_dir(args.tenant)
    out.mkdir(parents=True, exist_ok=True)
    out_file = out / "chunks.jsonl"

    n = 0
    with out_file.open("w", encoding="utf-8") as f:
        for path in collect_files(args.source, args.tenant):
            text = path.read_text(encoding="utf-8", errors="ignore")
            for i, chunk in enumerate(chunk_text(text)):
                f.write(json.dumps(
                    {"id": f"{path.stem}-{i}", "source": path.name, "text": chunk},
                    ensure_ascii=False) + "\n")
                n += 1
    print(f"[ingest] tenant={args.tenant} chunks={n} -> {out_file}")


if __name__ == "__main__":
    main()
