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
    "/media/wrf/b2ba2d39-9360-4508-b9e8-a2698f12ae67/zym/project/depression_project/eRisk2017/processed_2018"
)

ORIGIN_DIR = ROOT / "origin"

TOP_K = 16

OUT_DIR = ROOT / f"new_combined_maxsim{TOP_K}_ours"
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
    "I feel like I’ve already tried so many ways to get better, but nothing really seems to stick. Meds might make me a little more functional or keep me from completely falling apart, but they don’t make me feel actually okay, and sometimes they just leave me numb or weirdly disconnected. Therapy, lifestyle changes, different prescriptions, all of it starts to feel like another round of hoping and then being disappointed. Even when I have a few better days, I’m already scared it’s temporary and that I’ll crash again. Part of me still wants help, but another part is just tired of trying and wondering if this is something that can’t really be fixed.",
    "I keep getting stuck in this loop where every little mistake turns into proof that I’m a bad person or that something is wrong with me. Even when I start to feel a bit better, my brain finds a way to ruin it, like I don’t deserve to be happy or I should feel guilty for feeling okay. I replay conversations, things I said, ways I reacted, and then I end up apologizing over and over because I’m scared I hurt someone or came across wrong. It’s exhausting because I can’t just let things go; my mind keeps dragging me back to shame, regret, and all the reasons I think I’ve messed everything up.",
    "I can look totally fine on the outside, like I’m going to work or school, talking to people, doing what I’m supposed to do, but inside I’m barely holding it together. It feels like all my energy goes into acting normal so nobody realizes how bad things really are. People assume I’m okay because I don’t always show it, and even when I try to talk about it, I end up hiding the worst parts or making it sound less serious. I keep pushing everything down just to get through the day, but eventually it leaks out as crying, anger, panic, shutting people out, or just completely breaking down when I can’t pretend anymore.",
    "Sometimes I feel like there’s so much going on inside me that I don’t even know how to explain it. It’s like this heavy pressure sitting on me, sadness and anxiety and confusion all mixed together, but at the same time I can feel weirdly numb, like I’m hurting but can’t actually cry or react properly. I’ll act like I don’t care, joke about it, avoid people, or shut down because the feelings are too much to deal with directly. But pushing it down only works for so long, and eventually it comes out as anger, breaking down, feeling detached from everything, or just being completely overwhelmed by emotions I don’t know what to do with.",
    "I know there are basic things I’m supposed to do, like get up, go to work or class, shower, eat, clean my room, answer messages, just function like a normal person, but sometimes I honestly can’t make myself start. Everything feels like it takes more energy than I have, and even small responsibilities pile up until they feel impossible. I might still force myself through it if someone is depending on me or if I absolutely have to, but it doesn’t feel satisfying or productive, it just feels draining. Then when I fall behind, I feel guilty and useless, which somehow makes it even harder to catch up, so I end up stuck in this cycle of exhaustion, avoidance, and feeling like I’m failing at basic life.",
    "My sleep is completely out of sync and it messes up everything else. I can be exhausted all day and still not fall asleep when I’m supposed to, then I end up lying there counting how few hours I’ll get before I have to wake up. Even when I do sleep for a long time, it doesn’t feel like real rest, and I wake up tired like I barely slept at all. My body seems to want sleep at the wrong times, but work, school, and normal life don’t care, so I’m stuck dragging myself through the day, wanting to nap, relying on sleep aids or random sleep whenever I can get it, and feeling anxious every night because I know the cycle is probably going to happen again.",
    "My brain just won’t shut up sometimes. I keep going over the same thoughts again and again, replaying conversations, imagining what people meant, worrying about what might happen, or taking one small thing and turning it into ten different ways everything could go wrong. Even if I try to distract myself, the thoughts come back the second I’m alone or things get quiet. I can get so stuck in my own head that I barely notice what’s happening around me, and it makes it hard to focus, relax, or actually do anything. It’s like I know I’m overthinking, but I still can’t stop the loop, and the more I think, the more anxious and trapped I feel.",
    "It feels like depression has been part of my life for so long that I don’t even know what it would be like without it. Some days I can tell myself I’ve made progress, but it still feels like this heavy thing that follows me around and slowly drains the point out of everything. Stuff I used to care about doesn’t really feel the same anymore, and life can start to feel like this endless stretch of just getting through one day after another. I’m tired of waiting for things to finally feel okay, and part of me is scared that this might never fully go away. It’s not always that I’ve completely given up, but I don’t know how to picture a future where happiness feels stable or where I’m not constantly fighting to keep my head above water.",
    "Even when my life looks like it should be good, I still feel weirdly disconnected from it. I can have people who care about me, things going well, or chances I should be excited about, but it all feels muted, like I’m watching it happen instead of actually feeling it. Hobbies and distractions might keep me busy for a while, but they don’t really make me happy, and when I do feel a little better, I’m already bracing for it to disappear. I don’t want to spend my whole life waiting for some future version of happiness, but right now I can’t seem to enjoy what’s in front of me either.",
    "Sometimes it feels like I’m only still here because I don’t want to hurt the people or even the pets who depend on me, not because I actually want to keep going. My mind keeps telling me that life is just this trap I have to endure, and that if I disappeared, maybe everyone else would be better off without having to deal with me. I don’t always know if I truly want to die or if I just want the pain, shame, and exhaustion to stop, but the thoughts keep coming back when I can’t see another way out. Part of me still holds on because I care about people and don’t want to destroy them, but another part feels useless, unwanted, and unable to imagine a future where I belong or become someone worth loving.",
    "Food feels like it takes up way too much space in my head. Sometimes I’m constantly hungry or obsessing over what I’m going to eat next, and other times I barely have an appetite or end up restricting because it gives me some weird sense of control. When I’m sad, anxious, or empty, eating can feel like the only thing that actually helps for a moment, but then I feel ashamed and frustrated with myself afterward. Trying to eat better just makes me think about junk food more, like I’m fighting some kind of addiction, and the whole thing turns into this cycle of craving, guilt, avoiding, overeating, or not eating enough. I know my relationship with food isn’t healthy, but changing it feels overwhelming when I’m already depressed and barely motivated to take care of myself.",
    "When I’m depressed or anxious, sex can start to feel complicated instead of natural. I might want closeness, affection, and reassurance, but the actual sexual part can feel stressful, tiring, or just kind of impossible when I already feel ugly, numb, ashamed, or disconnected from my body. Sometimes I feel guilty because my partner wants more intimacy than I can give, and I worry they’ll take it personally or eventually get tired of me. My desire depends so much on whether I feel emotionally safe and wanted, and when I don’t, even being touched can bring up pressure instead of comfort. It makes me scared that my low libido or discomfort with sex means I’m hard to love or not enough for a relationship."
]  # 12

Ours = Psych + Ours

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