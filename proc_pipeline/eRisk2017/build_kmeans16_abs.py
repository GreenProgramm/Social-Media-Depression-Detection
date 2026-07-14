import os
import re
import json
from pathlib import Path
from tqdm import tqdm

import torch
from transformers import pipeline


# =========================================================
# 1. 路径配置
# =========================================================
ROOT = Path(
    "/media/wrf/b2ba2d39-9360-4508-b9e8-a2698f12ae67/zym/project/depression_project/eRisk2017/processed_2018"
)

K = 16

INPUT_ROOT = ROOT / f"kmeans{K}"
OUTPUT_ROOT = ROOT / f"kmeans{K}_abs256"

MODEL_DIR = Path(
    "/media/wrf/b2ba2d39-9360-4508-b9e8-a2698f12ae67/zym/project/depression_project/eRisk2017/pretrain_file/sshleifer/distilbart-cnn-12-6"
)

SPLITS = ["train", "test"]

# 你原来用的是 device=1
# 如果只有一张卡，改成 0；如果用 CPU，改成 -1
DEVICE = 1

MAX_SUMMARY_LEN = 256
MIN_SUMMARY_LEN = 20

# 输入太短时不做摘要，直接保留原文，避免 BART 对很短文本生成奇怪结果
MIN_INPUT_WORDS_FOR_SUMMARY = 30


# =========================================================
# 2. 工具函数
# =========================================================
def clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def read_user_kmeans_text(file_path: Path) -> str:
    """
    读取一个用户聚类筛选后的帖子。
    每行是一条代表性 post。
    这里将 16 条 post 合并成一个长文本，用于摘要。
    """
    posts = []

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = clean_text(line)
            if line:
                posts.append(line)

    # 用句号分隔不同 post，减少直接拼接造成的语义混乱
    text = " . ".join(posts)
    text = clean_text(text)

    return text


def count_words(text: str) -> int:
    return len(text.split())


def summarize_one_file(summarizer, input_file: Path, output_file: Path):
    text = read_user_kmeans_text(input_file)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    if len(text) == 0:
        summary = ""
    elif count_words(text) < MIN_INPUT_WORDS_FOR_SUMMARY:
        # 太短时直接保留原文本
        summary = text
    else:
        try:
            result = summarizer(
                text,
                truncation=True,
                max_length=MAX_SUMMARY_LEN,
                min_length=MIN_SUMMARY_LEN,
                do_sample=False
            )
            summary = result[0]["summary_text"].strip()
            summary = clean_text(summary)

        except Exception as e:
            print(f"[警告] 摘要失败，改为写入原文：{input_file}")
            print(f"       错误信息：{e}")
            summary = text

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(summary)

    return {
        "input_file": input_file.name,
        "output_file": output_file.name,
        "input_words": count_words(text),
        "summary_words": count_words(summary)
    }


# =========================================================
# 3. 处理 train / test
# =========================================================
def process_split(split: str, summarizer):
    input_dir = INPUT_ROOT / split
    output_dir = OUTPUT_ROOT / split

    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在：{input_dir}")

    files = sorted(input_dir.glob("*.txt"))

    print(f"\n========== Processing {split} ==========")
    print(f"Input dir : {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Files     : {len(files)}")

    metadata = []

    for file_path in tqdm(files, desc=f"kmeans{K}_abs256 {split}"):
        output_file = output_dir / file_path.name

        info = summarize_one_file(
            summarizer=summarizer,
            input_file=file_path,
            output_file=output_file
        )

        metadata.append(info)

    meta_path = OUTPUT_ROOT / f"{split}_metadata.json"

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"{split} metadata saved to: {meta_path}")


# =========================================================
# 4. 主函数
# =========================================================
def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        (OUTPUT_ROOT / split).mkdir(parents=True, exist_ok=True)

    if DEVICE >= 0:
        if not torch.cuda.is_available():
            print("[警告] 当前环境未检测到 CUDA，将自动使用 CPU")
            device = -1
        elif DEVICE >= torch.cuda.device_count():
            print(
                f"[警告] 指定 device={DEVICE}，但当前只有 {torch.cuda.device_count()} 张 GPU，"
                f"将自动使用 cuda:0"
            )
            device = 0
        else:
            device = DEVICE
    else:
        device = -1

    print("Loading summarization model...")
    print(f"Model dir: {MODEL_DIR}")
    print(f"Device   : {device}")

    summarizer = pipeline(
        "summarization",
        model=str(MODEL_DIR),
        tokenizer=str(MODEL_DIR),
        device=device
    )

    for split in SPLITS:
        process_split(split, summarizer)

    print("\n========== All Done ==========")
    print(f"Output root: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()