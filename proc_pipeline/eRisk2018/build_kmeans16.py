import os
import re
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer


# =========================================================
# 1. 路径配置
# =========================================================
ROOT = Path(
    "/media/wrf/b2ba2d39-9360-4508-b9e8-a2698f12ae67/zym/project/depression_project/eRisk2017/processed_2018"
)

ORIGIN_DIR = ROOT / "origin"

MODEL_PATH = "/media/wrf/b2ba2d39-9360-4508-b9e8-a2698f12ae67/zym/project/depression_project/eRisk2017/paraphrase-MiniLM-L6-v2"

K_LIST = [16]

BATCH_SIZE = 64

SPLITS = ["train", "test"]


# =========================================================
# 2. 你的 KMeans 聚类筛选函数，已合并并稍微修正
# =========================================================
def get_kmeans_centroid_ids(embs, K=32):
    """
    对一个用户的所有帖子 embedding 做 KMeans 聚类，
    每个簇选择距离簇中心最近的一条帖子。

    返回：
    centroid_ids: List[int]
    """
    embs = np.asarray(embs)

    if len(embs) == 0:
        return []

    if len(embs) <= K:
        return list(range(len(embs)))

    kmeans = KMeans(
        n_clusters=K,
        random_state=42,
        n_init=10
    )

    # shape: [num_posts, K]
    # cluster_dists[i, k] 表示第 i 条帖子到第 k 个聚类中心的距离
    cluster_dists = kmeans.fit_transform(embs)

    labels = kmeans.labels_

    ret = []

    for k in range(K):
        curr_members = np.where(labels == k)[0]

        if len(curr_members) == 0:
            continue

        # 当前簇内，找距离该簇中心最近的帖子
        local_id = cluster_dists[curr_members, k].argmin()
        global_id = curr_members[local_id]

        ret.append(global_id)

    # 按原始发帖顺序排序，保证输入 HAN-BERT 时仍然是时间顺序
    # return sorted(list(set(ret)))
    return [int(x) for x in sorted(list(set(ret)))]


def get_cluster_summary(user_posts, user_embs, K=32):
    """
    输入：
    user_posts: 一个用户的全部 posts
    user_embs : 该用户全部 posts 的 sbert embedding

    输出：
    summaries: 聚类筛选后的代表性 posts
    """
    if len(user_posts) == 0:
        return []

    if len(user_posts) <= K:
        return user_posts

    centroid_ids = get_kmeans_centroid_ids(user_embs, K)

    return [user_posts[i] for i in centroid_ids]


# =========================================================
# 3. 工具函数
# =========================================================
def parse_filename(file_path: Path):
    """
    输入文件名：
    train_subject96_1.txt
    test_subject25_0.txt

    返回：
    user_id = train_subject96
    label = 1
    """
    match = re.match(r"(.+)_([01])\.txt$", file_path.name)

    if not match:
        raise ValueError(f"文件名格式错误，应该是 user_label.txt：{file_path.name}")

    user_id = match.group(1)
    label = int(match.group(2))

    return user_id, label


def read_posts(file_path: Path):
    """
    origin 中每一行是一条 post。
    """
    posts = []

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            post = line.strip()
            post = re.sub(r"\s+", " ", post)

            if len(post) > 0:
                posts.append(post)

    return posts


def clean_post_for_write(post: str):
    post = post.replace("\n", " ")
    post = re.sub(r"\s+", " ", post)
    return post.strip()


# =========================================================
# 4. 处理 train / test
# =========================================================
def process_split(split, sbert, K):
    input_dir = ORIGIN_DIR / split
    output_dir = ROOT / f"kmeans{K}" / split

    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在：{input_dir}")

    user_files = sorted(input_dir.glob("*.txt"))

    print(f"\n========== Processing {split}, K={K} ==========")
    print(f"Input dir : {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"User files: {len(user_files)}")

    metadata = []

    for id0, user_file in enumerate(tqdm(user_files, desc=f"K={K}, {split}")):
        user_id, label = parse_filename(user_file)

        user_posts = read_posts(user_file)

        if len(user_posts) == 0:
            summaries = []
            centroid_ids = []
        else:
            user_embs = sbert.encode(
                user_posts,
                batch_size=BATCH_SIZE,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False
            )

            centroid_ids = get_kmeans_centroid_ids(user_embs, K=K)
            summaries = [user_posts[i] for i in centroid_ids]

        out_file = output_dir / f"{id0:06}_{label}.txt"

        with open(out_file, "w", encoding="utf-8") as f:
            f.write("\n".join(clean_post_for_write(x) for x in summaries))

        metadata.append({
            "index": int(id0),
            "output_file": out_file.name,
            "source_file": user_file.name,
            "user_id": str(user_id),
            "label": int(label),
            "total_posts": len(user_posts),
            "selected_posts": len(summaries),
            "centroid_ids": [int(x) for x in centroid_ids]
        })

    meta_path = ROOT / f"kmeans{K}" / f"{split}_metadata.json"

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"{split} metadata saved to: {meta_path}")


# =========================================================
# 5. 主函数
# =========================================================
def main():
    print("Loading SentenceTransformer...")
    print(f"Model path: {MODEL_PATH}")

    sbert = SentenceTransformer(MODEL_PATH)

    for K in K_LIST:
        os.makedirs(ROOT / f"kmeans{K}", exist_ok=True)
        os.makedirs(ROOT / f"kmeans{K}" / "train", exist_ok=True)
        os.makedirs(ROOT / f"kmeans{K}" / "test", exist_ok=True)

        process_split("train", sbert, K)
        process_split("test", sbert, K)

    print("\n========== All Done ==========")
    for K in K_LIST:
        print(f"Output: {ROOT / f'kmeans{K}'}")


if __name__ == "__main__":
    main()