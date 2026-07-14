import re
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

# =========================================================
# 1. 路径配置
# =========================================================
TRAIN_DIR = Path(
    "/media/wrf/b2ba2d39-9360-4508-b9e8-a2698f12ae67/zym/project/depression_project/eRisk2017/processed_2018/origin/train"
)

MODEL_PATH = "/media/wrf/b2ba2d39-9360-4508-b9e8-a2698f12ae67/zym/project/depression_project/eRisk2017/paraphrase-MiniLM-L6-v2"

K = 16                  # 聚类数量
M = 16                  # 每个簇选取距离中心最近的帖子数量
BATCH_SIZE = 128
THRESHOLD = 0.45        # 症状模板匹配阈值

OUTPUT_FILE = f"label1_train_threshold{THRESHOLD}_cluster_top{K}_{M}posts.txt"
OUTPUT_META = f"label1_train_threshold{THRESHOLD}_cluster_top{K}_{M}posts_metadata.json"
FILTERED_FILE = f"label1_train_threshold{THRESHOLD}_filtered_posts.txt"
FILTERED_META = f"label1_train_threshold{THRESHOLD}_filtered_posts_metadata.json"

DEVICE = "cuda:0"

# =========================================================
# 2. 24条中文症状模板
# =========================================================
Psych_chinese = [
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

# =========================================================
# 3. 工具函数
# =========================================================
def clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_filename(file_path: Path):
    """
    文件名示例：
    train_subject96_1.txt
    train_subject25_0.txt

    返回：
    user_id, label
    """
    match = re.match(r"(.+)_([01])\.txt$", file_path.name)

    if not match:
        raise ValueError(f"文件名格式错误，应为 xxx_0.txt 或 xxx_1.txt：{file_path.name}")

    user_id = match.group(1)
    label = int(match.group(2))

    return user_id, label


def read_posts(file_path: Path):
    """
    每个用户文件中，每一行是一条 post。
    """
    posts = []

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            post = clean_text(line)
            if post:
                posts.append(post)

    return posts


def load_label1_posts(train_dir: Path):
    """
    读取 train 目录下所有 label=1 用户的全部帖子。
    """
    label1_posts = []
    label1_infos = []

    user_files = sorted(train_dir.glob("*.txt"))

    print(f"Train dir: {train_dir}")
    print(f"Total user files: {len(user_files)}")

    label1_user_count = 0

    for user_file in tqdm(user_files, desc="Loading label=1 users"):
        user_id, label = parse_filename(user_file)

        if label != 1:
            continue

        label1_user_count += 1
        posts = read_posts(user_file)

        for post_idx, post in enumerate(posts):
            label1_posts.append(post)
            label1_infos.append({
                "user_id": user_id,
                "source_file": user_file.name,
                "post_idx": int(post_idx),
                "label": int(label),
                "post": post
            })

    print(f"Label=1 users: {label1_user_count}")
    print(f"Label=1 posts: {len(label1_posts)}")

    return label1_posts, label1_infos


# =========================================================
# 4. BGE 编码
# =========================================================
def encode_texts(sbert, texts, desc="Encoding texts"):
    embs = sbert.encode(
        texts,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    return np.asarray(embs)


# =========================================================
# 5. 症状模板相似度筛选
# =========================================================
def filter_by_template_similarity(
    posts,
    infos,
    post_embs,
    template_embs,
    templates,
    threshold=0.45
):
    """
    对每条帖子计算其与24条症状模板的最大相似度。
    risk_score = max cosine(post, template)

    只保留 risk_score > threshold 的帖子。
    """
    print("\n========== Template Similarity Filtering ==========")

    # 因为 encode 时 normalize_embeddings=True，所以点积就是 cosine similarity
    sim_matrix = post_embs @ template_embs.T

    risk_scores = sim_matrix.max(axis=1)
    matched_template_ids = sim_matrix.argmax(axis=1)

    keep_indices = np.where(risk_scores > threshold)[0]

    filtered_posts = []
    filtered_infos = []
    filtered_embs = []

    for idx in keep_indices:
        template_id = int(matched_template_ids[idx])
        score = float(risk_scores[idx])

        info = dict(infos[idx])
        info["risk_score"] = score
        info["matched_template_id"] = template_id
        info["matched_template"] = templates[template_id]

        filtered_posts.append(posts[idx])
        filtered_infos.append(info)
        filtered_embs.append(post_embs[idx])

    filtered_embs = np.asarray(filtered_embs)

    print(f"Original posts : {len(posts)}")
    print(f"Filtered posts : {len(filtered_posts)}")
    print(f"Threshold      : {threshold}")

    if len(filtered_posts) > 0:
        print(f"Risk score min : {float(np.min(risk_scores[keep_indices])):.4f}")
        print(f"Risk score max : {float(np.max(risk_scores[keep_indices])):.4f}")
        print(f"Risk score mean: {float(np.mean(risk_scores[keep_indices])):.4f}")

    return filtered_posts, filtered_infos, filtered_embs


def save_filtered_posts(filtered_posts, filtered_infos):
    """
    保存阈值筛选后的全部帖子，方便人工检查。
    """
    with open(FILTERED_FILE, "w", encoding="utf-8") as f:
        for post, info in zip(filtered_posts, filtered_infos):
            f.write(clean_text(post) + "\n")
            f.write(
                f"[score={info['risk_score']:.4f}, "
                f"template_id={info['matched_template_id']}, "
                f"template={info['matched_template']}]\n"
            )
            f.write("-" * 50 + "\n")

    with open(FILTERED_META, "w", encoding="utf-8") as f:
        json.dump(filtered_infos, f, ensure_ascii=False, indent=2)

    print(f"已保存阈值筛选帖子到：{FILTERED_FILE}")
    print(f"已保存阈值筛选元信息到：{FILTERED_META}")


# =========================================================
# 6. KMeans 聚类并选择代表性帖子
# =========================================================
def cluster_and_select(posts, infos, embs, K=16, M=16):
    if len(posts) == 0:
        raise ValueError("阈值筛选后没有保留下来的帖子，请降低 THRESHOLD。")

    n_clusters = min(K, len(posts))

    print("\n========== KMeans Clustering ==========")
    print(f"KMeans n_clusters = {n_clusters}")
    print(f"Posts for clustering = {len(posts)}")

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    cluster_ids = kmeans.fit_predict(embs)

    cluster_selected_posts = []
    cluster_selected_infos = []

    for k in range(n_clusters):
        cluster_indices = np.where(cluster_ids == k)[0]

        if len(cluster_indices) == 0:
            cluster_selected_posts.append([])
            cluster_selected_infos.append([])
            continue

        cluster_embs = embs[cluster_indices]
        center = kmeans.cluster_centers_[k]

        distances = np.linalg.norm(cluster_embs - center, axis=1)
        sorted_idx = np.argsort(distances)

        top_m = sorted_idx[:M]

        selected_posts = []
        selected_infos = []

        for local_idx in top_m:
            global_idx = cluster_indices[local_idx]

            selected_posts.append(posts[global_idx])

            info = dict(infos[global_idx])
            info["cluster_id"] = int(k)
            info["distance_to_center"] = float(distances[local_idx])
            selected_infos.append(info)

        cluster_selected_posts.append(selected_posts)
        cluster_selected_infos.append(selected_infos)

        print(f"Cluster {k}: total={len(cluster_indices)}, selected={len(selected_posts)}")

    return cluster_selected_posts, cluster_selected_infos


# =========================================================
# 7. 保存聚类结果
# =========================================================
def save_cluster_results(cluster_selected_posts, cluster_selected_infos):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for cluster_id, posts in enumerate(cluster_selected_posts):
            clean_posts = [clean_text(p) for p in posts]

            line = "\n".join(clean_posts)

            f.write(line + "\n" + "*" * 50 + "\n")

    metadata = []

    for cluster_id, infos in enumerate(cluster_selected_infos):
        metadata.append({
            "cluster_id": int(cluster_id),
            "selected_count": len(infos),
            "posts": infos
        })

    with open(OUTPUT_META, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"已保存聚类帖子到：{OUTPUT_FILE}")
    print(f"已保存聚类元信息到：{OUTPUT_META}")


# =========================================================
# 8. 主函数
# =========================================================
def main():
    print("Loading BGE model...")
    print(f"Model path: {MODEL_PATH}")
    print(f"Device    : {DEVICE}")

    sbert = SentenceTransformer(MODEL_PATH, device=DEVICE)

    # 1. 读取 label=1 用户所有帖子
    label1_posts, label1_infos = load_label1_posts(TRAIN_DIR)

    # 2. 编码帖子
    print("\nEncoding label=1 posts...")
    label1_embs = encode_texts(sbert, label1_posts)

    print(f"Post embedding shape: {label1_embs.shape}")

    # 3. 编码24条中文症状模板
    print("\nEncoding symptom templates...")
    template_embs = encode_texts(sbert, Psych_chinese)

    print(f"Template embedding shape: {template_embs.shape}")

    # 4. 阈值筛选：只保留 risk_score > 0.45 的帖子
    filtered_posts, filtered_infos, filtered_embs = filter_by_template_similarity(
        posts=label1_posts,
        infos=label1_infos,
        post_embs=label1_embs,
        template_embs=template_embs,
        templates=Psych_chinese,
        threshold=THRESHOLD
    )

    # 5. 保存筛选后的全部帖子，方便查看
    save_filtered_posts(filtered_posts, filtered_infos)

    # 6. 对筛选后的帖子做 KMeans 聚类
    cluster_selected_posts, cluster_selected_infos = cluster_and_select(
        posts=filtered_posts,
        infos=filtered_infos,
        embs=filtered_embs,
        K=K,
        M=M
    )

    # 7. 保存聚类结果
    save_cluster_results(cluster_selected_posts, cluster_selected_infos)

    print("\n========== All Done ==========")


if __name__ == "__main__":
    main()