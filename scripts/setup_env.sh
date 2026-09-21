#!/bin/bash
set -e
export PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
command -v git >/dev/null || (dnf install -y git || yum install -y git)
python3 -m venv /root/train-study/mini-engram/.venv
/root/train-study/mini-engram/.venv/bin/pip install -U pip
/root/train-study/mini-engram/.venv/bin/pip install vllm transformers peft accelerate datasets modelscope pyyaml tqdm openai ninja
echo SETUP_DONE
