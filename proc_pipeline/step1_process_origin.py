import re
import xml.dom.minidom
from pathlib import Path


# =========================
# 1. 路径配置
# =========================
SRC_ROOT = Path(
    "/media/wrf/b2ba2d39-9360-4508-b9e8-a2698f12ae67/zym/project/depression_project/dataset/eRisk2018/by_user"
)
OUT_ROOT = Path(
    "/media/wrf/b2ba2d39-9360-4508-b9e8-a2698f12ae67/zym/project/depression_project/eRisk2017/processed_2018/origin"
)

SPLITS = ["train", "test"]

LABEL_MAP = {
    "positive": 1,
    "negative": 0,
}

# 每个 XML 文件内部是否反转帖子顺序
# 你说“每个 xml 文件的顺序是从下往上的”，所以这里设为 True
REVERSE_POSTS_IN_XML = True


# =========================
# 2. 文本清洗
# =========================
def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_node_text(node) -> str:
    """
    安全读取 XML 节点文本，避免 firstChild 为 None 报错。
    """
    if node is None:
        return ""

    if node.firstChild is None:
        return ""

    return node.firstChild.data or ""


# =========================
# 3. 解析单个 XML 文件
# =========================
def get_input_data(xml_path: Path):
    """
    参考你给出的逻辑：
    post = TITLE + ' ' + TEXT

    返回：
    posts: List[str]
    post_num: int
    """
    posts = []

    try:
        dom = xml.dom.minidom.parse(str(xml_path))
    except Exception as e:
        print(f"[错误] XML 解析失败：{xml_path}")
        print(f"       {e}")
        return [], 0

    collection = dom.documentElement
    title_nodes = collection.getElementsByTagName("TITLE")
    text_nodes = collection.getElementsByTagName("TEXT")

    pair_num = min(len(title_nodes), len(text_nodes))

    for i in range(pair_num):
        title = get_node_text(title_nodes[i])
        text = get_node_text(text_nodes[i])

        post = title + " " + text
        post = re.sub("\n", " ", post)
        post = clean_text(post)

        if len(post) > 0:
            posts.append(post)

    # 关键：每个 XML 内部从下往上，所以反转
    if REVERSE_POSTS_IN_XML:
        posts = posts[::-1]

    return posts, len(posts)


# =========================
# 4. 提取 chunk 编号
# =========================
def get_chunk_id(xml_path: Path) -> int:
    """
    例如：
    train_subject96_1.xml  -> 1
    train_subject96_10.xml -> 10
    """
    match = re.search(r"_(\d+)\.xml$", xml_path.name)

    if match:
        return int(match.group(1))

    return 999999


# =========================
# 5. 处理单个用户
# =========================
def process_one_user(user_dir: Path, out_file: Path):
    """
    一个用户文件夹下有多个 chunk XML。
    按 chunk1、chunk2、... 的顺序读取。
    每个 XML 内部按从下往上的顺序写入。
    """
    xml_files = sorted(
        user_dir.glob("*.xml"),
        key=get_chunk_id
    )

    all_posts = []
    total_post_num = 0

    for xml_file in xml_files:
        posts, post_num = get_input_data(xml_file)
        all_posts.extend(posts)
        total_post_num += post_num

    out_file.parent.mkdir(parents=True, exist_ok=True)

    with open(out_file, "w", encoding="utf-8") as f:
        for post in all_posts:
            f.write(post + "\n")

    return len(xml_files), total_post_num


# =========================
# 6. 主函数
# =========================
def build_origin_txt():
    total_users = 0
    total_xml_files = 0
    total_posts = 0

    for split in SPLITS:
        split_src_dir = SRC_ROOT / split
        split_out_dir = OUT_ROOT / split
        split_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n========== 处理 {split} ==========")
        print(f"输入目录：{split_src_dir}")
        print(f"输出目录：{split_out_dir}")

        if not split_src_dir.exists():
            print(f"[警告] 输入目录不存在，跳过：{split_src_dir}")
            continue

        for label_name, label_id in LABEL_MAP.items():
            label_src_dir = split_src_dir / label_name

            if not label_src_dir.exists():
                print(f"[警告] 标签目录不存在，跳过：{label_src_dir}")
                continue

            user_dirs = sorted(
                [p for p in label_src_dir.iterdir() if p.is_dir()],
                key=lambda x: x.name
            )

            print(f"\n处理类别：{label_name}，标签：{label_id}")
            print(f"用户数量：{len(user_dirs)}")

            for user_dir in user_dirs:
                user_id = user_dir.name

                # 输出文件名：train_subject96_1.txt / test_subject25_0.txt
                out_file = split_out_dir / f"{user_id}_{label_id}.txt"

                xml_count, post_count = process_one_user(user_dir, out_file)

                total_users += 1
                total_xml_files += xml_count
                total_posts += post_count

                print(
                    f"[完成] {split}/{label_name}/{user_id} "
                    f"xml={xml_count}, posts={post_count} -> {out_file.name}"
                )

    print("\n========== 全部处理完成 ==========")
    print(f"用户总数：{total_users}")
    print(f"XML 文件总数：{total_xml_files}")
    print(f"帖子总数：{total_posts}")
    print(f"最终输出目录：{OUT_ROOT}")


if __name__ == "__main__":
    build_origin_txt()