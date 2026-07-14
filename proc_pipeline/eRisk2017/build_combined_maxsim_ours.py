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

OUT_DIR = ROOT / f"combined_maxsim{TOP_K}_ours_extend"
MODEL_PATH = "/media/wrf/b2ba2d39-9360-4508-b9e8-a2698f12ae67/zym/project/depression_project/eRisk2017/paraphrase-MiniLM-L6-v2"

SPLITS = ["train", "test"]

BATCH_SIZE = 128

# =========================
# 2. Psych 模板
# =========================
Psych = [
    "I feel sad.",
    "I am discouraged about my future.",
    "I always fail.",
    "I don't get pleasure from things.",
    "I feel quite guilty.",
    "I expected to be punished.",
    "I am disappointed in myself.",
    "I always criticize myself for my faults.",
    "I have thoughts of killing myself.",
    "I always cry.",
    "I am hard to stay still.",
    "It's hard to get interested in things.",
    "I have trouble making decisions.",
    "I feel worthless.",
    "I don't have energy to do things.",
    "I have changes in my sleeping pattern.",
    "I am always irritable.",
    "I have changes in my appetite.",
    "I feel hard to concentrate on things.",
    "I am too tired to do things.",
    "I have lost my interest in sex.",
    "I feel depressed.",
    "I am diagnosed with depression.",
    "I am treating my depression.",
]

Ours = [
    'I keep scrolling like I always do, but nothing really lands anymore—it’s like my attention just slips away and everything feels kind of empty or pointless. Posts that used to interest me now just annoy me, and I catch myself rolling my eyes at almost everything online. I know I’m not even enjoying it, but I still stay there, mindlessly switching apps, feeling drained and weirdly disconnected, like I’m just going through the motions without actually caring.',
    'Lately it just feels like none of this really matters—like life itself doesn’t have any real value and we’re all just kind of here for no reason. Even the idea of death doesn’t feel scary anymore, more like it’s just part of the same meaningless cycle. I find myself thinking that people are replaceable and everything we go through doesn’t really add up to anything, and it’s hard to care about the future when it all seems so empty.',
    'Sometimes I feel so small compared to everything else, like I could disappear and nothing would really change. It’s like I’m just another random person in a huge world that doesn’t notice or need me, and whatever I do doesn’t leave any real mark. I keep wondering what the point is if I’m this insignificant, and it makes it hard to feel like anything I do actually matters.',
    'Things that should make me feel something just… don’t anymore. I can tell when something is supposed to be exciting or sad, but it’s like there’s a layer between me and the feeling. I just kind of observe it without reacting, and even moments I used to enjoy feel flat and distant. It’s not that everything is bad, it’s more like nothing really hits at all.',
    'I’ve started brushing things off like they don’t matter, even when they probably do. Conversations, problems, emotions—I just kind of shrug and think “what’s the point” and move on. It’s easier to stay detached than to get involved, so I keep everything at a distance and act like I don’t care, even if part of me knows I’m just avoiding it.',
    'Food is constantly on my mind, whether I’m trying to ignore it or thinking about it too much. I go back and forth between craving things and then feeling guilty for even wanting them, sometimes trying to control it and other times feeling like I’ve completely lost control. It’s exhausting, like my thoughts are stuck in this loop of eating, not eating, and regretting it either way.',
    'I feel stuck in the same place no matter what I do, like I’m not moving forward at all. Even when I try to change things, it doesn’t seem to make a difference, and eventually I just lose the energy to keep trying. It’s like being trapped in a situation that won’t shift, and over time it just turns into this quiet feeling that nothing I do will actually work.',
    'I can’t shake the feeling that people are judging me or talking about me, even when there’s no clear reason to think that. Small things start to feel intentional, like someone’s tone or a message that seems off, and my mind runs with it. It’s hard to trust what people say at face value because part of me always thinks there’s something else going on underneath.',
    'Even the smallest decisions feel overwhelming because I keep second-guessing myself. I’ll go back and forth in my head, wondering if I’m making the wrong choice, and even after I decide, I can’t stop thinking about the alternatives. It’s like I don’t trust my own judgment anymore, so everything takes longer and still doesn’t feel right in the end.',
    'It’s hard to believe that any system is actually fair or working the way it’s supposed to. Everything feels biased or broken in some way, like the outcome is already decided and people like me don’t really stand a chance. Over time I’ve just stopped expecting things to be just or meaningful, because it feels pointless to rely on something that never really delivers.',
    'I keep replaying things I’ve said or done, picking them apart and convincing myself I messed up somehow. Even small interactions turn into long mental loops where I imagine how others might have judged me. It’s like my mind won’t let go, constantly reminding me of my flaws and making me feel embarrassed or inadequate over things that probably don’t matter as much as I think.',
    'On the outside I can still joke around and have normal conversations, but it never goes deeper than that. I keep things light and avoid talking about how I actually feel, like I’m playing a role just to keep things smooth. It works enough to stay connected, but it also feels kind of empty, like no one really knows what’s going on with me.',
    'I’ve started expecting people to disappoint me, so I don’t really let myself get too close anymore. It’s easier to stay guarded than to trust and end up hurt again. Even when someone seems genuine, there’s this part of me that doubts it, like I’m just waiting for things to go wrong, so I keep my distance emotionally.',
    'I want to be close to people, but at the same time it kind of scares me. I overthink everything—what they said, how they feel, whether I’m too much or not enough—and it makes me hesitate to fully open up. I end up going back and forth between wanting connection and pulling away, like I can’t decide if it’s safe to let someone in.'
]  # 14

# Ours = Psych + Ours

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
        Ours,
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