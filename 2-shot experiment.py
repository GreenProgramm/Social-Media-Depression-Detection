import os
import re
import json
import random
from pathlib import Path

from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

from vllm import LLM, SamplingParams

# =========================
# 1. 路径与参数配置
# =========================

BASE_DIR = Path(
    "../processed_2018/new_combined_maxsim16_ours"
)
TRAIN_DIR = BASE_DIR / "train"
TEST_DIR = BASE_DIR / "test"

# MODEL_PATH = "../pretrain_file/Qwen/Qwen3-8B"
MODEL_PATH = "../pretrain_file/Llama/Llama-3.1-8B-Instruct"

# OUTPUT_ROOT_DIR = BASE_DIR / "qwen3_8B_ours_results"
OUTPUT_ROOT_DIR = BASE_DIR / "llama_3.1_8B_ours_results"

MAX_CHARS_PER_USER = 12000
BATCH_SIZE = 128
MAX_TEST_FILES = None
SEED = 42


def get_label_from_filename(file_path: Path) -> int:
    """
    文件名格式：
    000000_0.txt
    004729_1.txt

    _0 表示 control
    _1 表示 depressed
    """
    match = re.search(r"_(0|1)\.txt$", file_path.name)
    if not match:
        raise ValueError(f"无法从文件名解析标签: {file_path.name}")
    return int(match.group(1))


def read_user_text(file_path: Path, max_chars: int = MAX_CHARS_PER_USER) -> str:
    text = file_path.read_text(encoding="utf-8", errors="ignore").strip()

    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    if len(text) > max_chars:
        text = text[:max_chars] + "\n...[TRUNCATED]"

    return text


def load_items(folder: Path):
    files = sorted(folder.glob("*.txt"))
    items = []

    for fp in files:
        items.append({
            "path": fp,
            "filename": fp.name,
            "label": get_label_from_filename(fp),
            "text": read_user_text(fp),
        })

    return items


def split_train_by_label(train_items):
    depressed_items = [x for x in train_items if x["label"] == 1]
    control_items = [x for x in train_items if x["label"] == 0]

    if len(depressed_items) == 0:
        raise ValueError("train 文件夹中没有 _1.txt 患病样本。")
    if len(control_items) == 0:
        raise ValueError("train 文件夹中没有 _0.txt 正常样本。")

    return depressed_items, control_items


def sample_2shot_from_pools(depressed_items, control_items):
    dep_example = random.choice(depressed_items)
    ctl_example = random.choice(control_items)
    return dep_example, ctl_example


def build_user_prompt(dep_example, ctl_example, test_item):
    prompt = f"""
Task: Classify the given social media post. Determine whether the author is in the depressed group or the control group. 

Two balanced samples from depressed group and control group are provided as references.

Only output "depressed" or "control".

Example 1 (Depressed sample):
Post: {dep_example["text"]}
Label: depressed

Example 2 (Control sample):
Post: {ctl_example["text"]}
Label: control

Now classify the following post:
Post: {test_item["text"]}
Label: 
"""

    return prompt


def build_chat_prompt(tokenizer, dep_example, ctl_example, test_item):
    system_prompt = (
        "You are a careful text classification model. "
        "You must output only valid JSON."
    )

    user_prompt = build_user_prompt(dep_example, ctl_example, test_item)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    return prompt


def parse_prediction(output_text: str):
    raw_text = output_text

    lower_text = raw_text.lower()

    if "depressed" in lower_text and "control" not in lower_text:
        pred = 1
        pred_label = "depressed"
    else:
        pred = 0
        pred_label = "control"

    return {
        "pred": pred,
        "pred_label": pred_label,
        "raw_output": raw_text,
    }


def batch_iter(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def run(seed):
    random.seed(seed)
    output_dir = OUTPUT_ROOT_DIR / f"output"
    output_dir.mkdir(parents=True, exist_ok=True)

    result_jsonl = output_dir / "predictions.jsonl"
    metric_txt = output_dir / "metrics.txt"

    print(f"\n{'=' * 50}")
    print(f"输出目录: {output_dir}")
    print(f"{'=' * 50}\n")

    # 加载数据（只加载一次）
    train_items = load_items(TRAIN_DIR)
    test_items = load_items(TEST_DIR)

    if MAX_TEST_FILES is not None:
        test_items = test_items[:MAX_TEST_FILES]

    print(f"训练样本数: {len(train_items)}")
    print(f"测试样本数: {len(test_items)}")

    depressed_pool, control_pool = split_train_by_label(train_items)

    print(f"Depressed pool size: {len(depressed_pool)}")
    print(f"Control pool size:   {len(control_pool)}")

    llm = LLM(
            model=MODEL_PATH,
            trust_remote_code=True,
            dtype="auto",
            tensor_parallel_size=2,
            gpu_memory_utilization=0.90,
            max_model_len=32768,
        )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=256,
    )

    y_true = []
    y_pred = []

    for batch_items in tqdm(list(batch_iter(test_items, BATCH_SIZE)), desc=f"Seed {seed} testing"):
        prompts = []
        shot_pairs = []

        for item in batch_items:
            dep_example, ctl_example = sample_2shot_from_pools(
                depressed_pool,
                control_pool
            )

            prompt = build_chat_prompt(
                tokenizer,
                dep_example,
                ctl_example,
                item
            )

            prompts.append(prompt)
            shot_pairs.append((dep_example, ctl_example))

        outputs = llm.generate(prompts, sampling_params)

        for item, output, shot_pair in zip(batch_items, outputs, shot_pairs):
            dep_example, ctl_example = shot_pair

            generated_text = output.outputs[0].text
            parsed = parse_prediction(generated_text)

            gold = int(item["label"])
            pred = int(parsed["pred"])

            y_true.append(gold)
            y_pred.append(pred)

            record = {
                "seed": seed,
                "test_file": item["filename"],
                "gold": gold,
                "pred": pred,
                "pred_label": parsed["pred_label"],
                "raw_output": parsed["raw_output"],
                "dep_shot_file": dep_example["filename"],
                "ctl_shot_file": ctl_example["filename"]
            }

            with open(result_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["control", "depressed"],
        digits=4,
        zero_division=0,
    )

    metric_text = f"""
Model: {MODEL_PATH}
Random Seed: {seed}
Train dir: {TRAIN_DIR}
Test dir: {TEST_DIR}

Classification report:
{report}
""".strip()

    with open(metric_txt, "w", encoding="utf-8") as f:
        f.write(metric_text)

    print(f"预测结果: {result_jsonl}")
    print(f"指标文件: {metric_txt}")


def main():
    run(SEED)
    print(f"结果目录: {OUTPUT_ROOT_DIR}")


if __name__ == "__main__":
    main()