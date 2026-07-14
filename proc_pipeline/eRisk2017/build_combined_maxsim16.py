import os
import re
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer


# =========================
# 1. 路径配置
# =========================
ROOT = Path(
    "/media/wrf/b2ba2d39-9360-4508-b9e8-a2698f12ae67/zym/project/depression_project/eRisk2017/processed_2017"
)

ORIGIN_DIR = ROOT / "origin"

TOP_K = 16

OUT_DIR = ROOT / f"combined_maxsim{TOP_K}_LLM_only"
MODEL_PATH = "/media/wrf/b2ba2d39-9360-4508-b9e8-a2698f12ae67/zym/project/depression_project/eRisk2017/paraphrase-MiniLM-L6-v2"

SPLITS = ["train", "test"]

BATCH_SIZE = 128


# =========================
# 2. Psych 模板
# =========================
# Psych = [
#     "I feel sad.",
#     "I am discouraged about my future.",
#     "I always fail.",
#     "I don't get pleasure from things.",
#     "I feel quite guilty.",
#     "I expected to be punished.",
#     "I am disappointed in myself.",
#     "I always criticize myself for my faults.",
#     "I have thoughts of killing myself.",
#     "I always cry.",
#     "I am hard to stay still.",
#     "It's hard to get interested in things.",
#     "I have trouble making decisions.",
#     "I feel worthless.",
#     "I don't have energy to do things.",
#     "I have changes in my sleeping pattern.",
#     "I am always irritable.",
#     "I have changes in my appetite.",
#     "I feel hard to concentrate on things.",
#     "I am too tired to do things.",
#     "I have lost my interest in sex.",
#     "I feel depressed.",
#     "I am diagnosed with depression.",
#     "I am treating my depression.",
# ]

Psych = [
    "I feel depressed.",
    "I have been feeling really down lately.",
    "I feel like I am stuck in a depressed mood.",
    "I cannot shake this heavy feeling inside.",
    "I have been feeling low for a long time.",
    "It feels like everything around me is dragging me down.",
    "I am diagnosed with depression.",
    "I was told I have depression.",
    "My doctor diagnosed me with depression.",
    "I have been officially diagnosed with depression.",
    "The diagnosis I got was depression.",
    "I found out that I have depression.",
    "I am treating my depression.",
    "I am getting help for my depression.",
    "I am working on treating my depression.",
    "I have started treatment for depression.",
    "I am trying to manage my depression with treatment.",
    "I am doing something to treat my depression.",
    "I feel sad.",
    "I feel sad today.",
    "I have been feeling really unhappy.",
    "I feel like I could cry for no clear reason.",
    "There is a sadness I cannot shake off.",
    "I feel emotionally low right now.",
    "I am discouraged about my future.",
    "I do not feel hopeful about my future.",
    "Thinking about the future makes me feel discouraged.",
    "I cannot see things getting better for me.",
    "My future feels really uncertain and bleak.",
    "I feel like there is not much to look forward to."
    "I always fail.",
    "I feel like I keep failing at everything.",
    "No matter what I try, I seem to mess it up.",
    "I keep thinking about all the things I failed at.",
    "It feels like failure follows me everywhere.",
    "I feel like I never do anything right.",
    "I don’t get pleasure from things.",
    "Things I used to enjoy do not feel fun anymore.",
    "I cannot really enjoy anything lately.",
    "Nothing feels satisfying the way it used to.",
    "Even good things do not bring me much joy now.",
    "I do not feel pleasure from the things I normally like.",
    "I feel quite guilty."
    "I keep feeling guilty about things.",
    "I feel like a lot of things are my fault.",
    "Guilt has been sitting with me all day.",
    "I cannot stop blaming myself inside.",
    "I feel bad about what I have done.",
    "I expected to be punished.",
    "I feel like I deserve to be punished.",
    "I keep expecting something bad to happen to me.",
    "It feels like I should pay for my mistakes.",
    "I feel like punishment is coming for me.",
    "I cannot stop thinking that I should be punished."
    "I am disappointed in myself.",
    "I feel really disappointed in myself.",
    "I do not like the person I am right now.",
    "I feel let down by myself.",
    "I am unhappy with who I have become.",
    "It is hard to feel okay about myself.",
    "I always criticize myself for my faults.",
    "I keep picking apart everything I do wrong.",
    "I am always hard on myself for my mistakes.",
    "I cannot stop criticizing myself.",
    "Every small fault makes me blame myself.",
    "I keep telling myself I should have done better.",
    "I have thoughts of killing myself.",
    "I have been having thoughts about ending my life.",
    "Sometimes I think I do not want to be alive.",
    "I keep having thoughts of hurting myself.",
    "I have thoughts about not being here anymore.",
    "I have been thinking about killing myself.",
    "I always cry.",
    "I keep crying lately.",
    "I cry much more than I used to.",
    "Tears come out even when I try to hold them back.",
    "I feel like I am crying all the time.",
    "I keep breaking down and crying.",
    "I am hard to stay still.",
    "I cannot sit still for long.",
    "I feel restless all the time.",
    "My body feels tense and unsettled.",
    "I keep moving around because I feel uneasy.",
    "It is hard for me to stay calm and still.",
    "It’s hard to get interested in things.",
    "It is hard to care about anything lately.",
    "I cannot get interested in things like before.",
    "Everything feels boring to me now.",
    "I have lost interest in most things.",
    "Nothing really catches my attention anymore.",
    "I have trouble making decisions.",
    "I struggle to make even simple decisions.",
    "Choosing anything feels hard for me lately.",
    "I keep second-guessing every decision.",
    "I cannot decide what to do most of the time.",
    "Making decisions feels overwhelming.",
    "I feel worthless.",
    "I feel like I have no value.",
    "I feel completely useless sometimes.",
    "It feels like I do not matter.",
    "I keep feeling like I am not worth much.",
    "I feel like I am not good for anything.",
    "I don’t have energy to do things.",
    "I do not have the energy to do anything.",
    "Even small tasks feel like too much effort.",
    "I feel drained before I even start doing things.",
    "I have no strength to get things done.",
    "Everything takes more energy than I have.",
    "I have changes in my sleeping pattern.",
    "My sleep has been really different lately.",
    "I cannot keep a normal sleep schedule anymore.",
    "My sleeping pattern has changed a lot.",
    "I either sleep too much or not enough.",
    "My sleep feels completely off these days.",
    "I am always irritable.",
    "I get irritated so easily lately.",
    "Small things make me annoyed.",
    "I feel on edge and impatient.",
    "I keep snapping over little things.",
    "Everything seems to bother me more than usual.",
    "I have changes in my appetite.",
    "My appetite has changed a lot lately.",
    "I do not feel like eating the way I used to.",
    "I have been eating much more or much less than usual.",
    "Food does not feel the same to me anymore.",
    "My eating habits have been really different recently.",
    "I feel hard to concentrate on things.",
    "I cannot focus on anything for long.",
    "My mind keeps drifting when I try to concentrate.",
    "It is hard to pay attention to things.",
    "I keep losing track of what I am doing.",
    "Focusing feels much harder than usual.",
    "I am too tired to do things.",
    "I feel too tired to get anything done.",
    "I am exhausted even when I have not done much.",
    "I feel worn out most of the time.",
    "I am too tired to deal with normal tasks.",
    "My body feels tired all day.",
    "I have lost my interest in sex.",
    "I do not feel interested in sex anymore.",
    "My interest in sex has really gone down.",
    "I have little desire for sex lately.",
    "Sex does not appeal to me the way it used to.",
    "I feel much less interested in intimacy now."
]

# =========================
# 3. 工具函数
# =========================
def parse_filename(file_path: Path):
    """
    输入文件名示例：
    train_subject96_1.txt
    test_subject25_0.txt

    返回：
    user_id = train_subject96
    label = 1
    """
    match = re.match(r"(.+)_([01])\.txt$", file_path.name)

    if not match:
        raise ValueError(f"文件名格式不符合 user_label.txt：{file_path.name}")

    user_id = match.group(1)
    label = int(match.group(2))

    return user_id, label


def read_user_posts(file_path: Path):
    """
    读取一个用户的所有 posts。
    origin 文件中默认一行是一条 post。
    """
    posts = []

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            post = line.strip()
            post = re.sub(r"\s+", " ", post)

            if len(post) > 0:
                posts.append(post)

    return posts


def select_topk_posts(posts, sbert, psych_emb, top_k=16):
    """
    对一个用户的 posts 做 Psych-guided screening。

    1. encode 每条 post
    2. 计算 post 与 Psych templates 的 cosine similarity
    3. 每条 post 取最大相似度
    4. 取 topK
    5. 再按原始时间顺序排序
    """
    if len(posts) == 0:
        return [], [], []

    post_emb = sbert.encode(
        posts,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    # 因为 post_emb 和 psych_emb 都做了 normalize，所以点积就是 cosine similarity
    sim_matrix = np.matmul(post_emb, psych_emb.T)

    # 每条 post 对所有 Psych 模板取最大相似度
    sim_scores = sim_matrix.max(axis=1)

    real_top_k = min(top_k, len(posts))

    # 先按相似度取 topK
    top_ids = np.argsort(sim_scores)[-real_top_k:]

    # 再按原始时间顺序排列
    top_ids = np.sort(top_ids)

    selected_posts = [posts[idx] for idx in top_ids]
    selected_scores = [float(sim_scores[idx]) for idx in top_ids]

    return selected_posts, top_ids.tolist(), selected_scores


# =========================
# 4. 处理 train / test
# =========================
def process_split(split, sbert, psych_emb):
    input_dir = ORIGIN_DIR / split
    output_dir = OUT_DIR / split
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在：{input_dir}")

    user_files = sorted(input_dir.glob("*.txt"))

    print(f"\n========== Processing {split} ==========")
    print(f"Input dir : {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"User files: {len(user_files)}")

    metadata = []

    for i, user_file in enumerate(user_files):
        user_id, label = parse_filename(user_file)

        posts = read_user_posts(user_file)

        selected_posts, top_ids, selected_scores = select_topk_posts(
            posts=posts,
            sbert=sbert,
            psych_emb=psych_emb,
            top_k=TOP_K
        )

        out_file = output_dir / f"{i:06}_{label}.txt"

        with open(out_file, "w", encoding="utf-8") as f:
            f.write("\n".join(x.replace("\n", " ") for x in selected_posts))

        metadata.append({
            "index": i,
            "output_file": out_file.name,
            "source_file": user_file.name,
            "user_id": user_id,
            "label": label,
            "total_posts": len(posts),
            "selected_posts": len(selected_posts),
            "top_ids": top_ids,
            "selected_scores": selected_scores
        })

        if (i + 1) % 20 == 0 or (i + 1) == len(user_files):
            print(
                f"[{split}] {i + 1}/{len(user_files)} "
                f"{user_file.name} -> {out_file.name}, "
                f"posts={len(posts)}, selected={len(selected_posts)}"
            )

    # 保存映射关系，方便之后排查 000001_1.txt 对应哪个原始用户
    meta_file = OUT_DIR / f"{split}_metadata.json"

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"{split} metadata saved to: {meta_file}")


# =========================
# 5. 主函数
# =========================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        (OUT_DIR / split).mkdir(parents=True, exist_ok=True)

    print("Loading SentenceTransformer...")
    print(f"Model path: {MODEL_PATH}")

    sbert = SentenceTransformer(MODEL_PATH)

    print("Encoding Psych templates...")

    psych_emb = sbert.encode(
        Psych,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    print(f"Psych template embedding shape: {psych_emb.shape}")

    for split in SPLITS:
        process_split(split, sbert, psych_emb)

    print("\n========== All Done ==========")
    print(f"Output root: {OUT_DIR}")


if __name__ == "__main__":
    main()