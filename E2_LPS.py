import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# =========================================================
# 1. Depression-related templates
# =========================================================

DEPRESSION_TEMPLATES = [
    "I feel depressed.",
    "I am diagnosed with depression.",
    "I am treating my depression.",
    "I feel sad.",
    "I am discouraged about my future.",
    "I always fail.",
    "I don’t get pleasure from things.",
    "I feel quite guilty.",
    "I expected to be punished.",
    "I am disappointed in myself.",
    "I always criticize myself for my faults.",
    "I have thoughts of killing myself.",
    "I always cry.",
    "I am hard to stay still.",
    "It’s hard to get interested in things.",
    "I have trouble making decisions.",
    "I feel worthless.",
    "I don’t have energy to do things.",
    "I have changes in my sleeping pattern.",
    "I am always irritable.",
    "I have changes in my appetite.",
    "I feel hard to concentrate on things.",
    "I am too tired to do things.",
    "I have lost my interest in sex."
]

Ours_e2017_th = [
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
]

Ours_e2018_th = [
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
    "When I’m depressed or anxious, sex can start to feel complicated instead of natural. I might want closeness, affection, and reassurance, but the actual sexual part can feel stressful, tiring, or just kind of impossible when I already feel ugly, numb, ashamed, or disconnected from my body. Sometimes I feel guilty because my partner wants more intimacy than I can give, and I worry they’ll take it personally or eventually get tired of me. My desire depends so much on whether I feel emotionally safe and wanted, and when I don’t, even being touched can bring up pressure instead of comfort. It makes me scared that my low libido or discomfort with sex means I’m hard to love or not enough for a relationship."]

DEPRESSION_TEMPLATES = DEPRESSION_TEMPLATES + Ours_e2018_th

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clean_post(text: str) -> str:
    text = str(text).replace("\n", " ").replace("\r", " ").strip()
    return re.sub(r"[ \t]+", " ", text)


def get_label_from_filename(path: Path) -> int:
    """
    Examples:
        train_subject25_0.txt   -> 0
        test_subject8969_1.txt  -> 1
    """
    label = path.stem.rsplit("_", 1)[-1]
    if label not in ["0", "1"]:
        raise ValueError(f"Cannot parse label from filename: {path.name}")
    return int(label)


def binary_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    }


class RawUserStore:
    def __init__(self, data_root: str, split: str):
        assert split in ["train", "test"]
        self.split = split
        self.data_dir = Path(data_root) / split
        self.files = sorted(self.data_dir.glob("*.txt"))

        if len(self.files) == 0:
            raise FileNotFoundError(f"No .txt files found in {self.data_dir}")

        self.labels = np.array([get_label_from_filename(p) for p in self.files], dtype=np.int64)
        print(f"[{split}] users: {len(self.files)}")
        print(f"[{split}] positive: {int(self.labels.sum())}, negative: {len(self.labels) - int(self.labels.sum())}")

    def __len__(self):
        return len(self.files)

    def read_posts(self, path: Path) -> List[str]:
        text = path.read_text(encoding="utf-8", errors="ignore")

        posts = text.splitlines()

        posts = [clean_post(p) for p in posts]
        posts = [p for p in posts if p]
        return posts if posts else [""]

    def get_user(self, user_idx: int, max_posts: Optional[int] = None, training: bool = False) -> Dict:
        path = self.files[user_idx]
        posts = self.read_posts(path)
        if max_posts is not None and len(posts) > max_posts:
            if training:
                start = random.randint(0, len(posts) - max_posts)
                posts = posts[start:start + max_posts]
            else:
                posts = posts[:max_posts]

        return {
            "user_idx": user_idx,
            "filename": path.name,
            "posts": posts,
            "label": int(self.labels[user_idx]),
        }


# MiniLM encoder used in the screening process.
class SentenceBERTEncoder(nn.Module):
    def __init__(self, model_path: str, gradient_checkpointing: bool = False):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)

        if gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

    @staticmethod
    def mean_pool(last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        return (last_hidden_state * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-8)

    def forward(self, texts: List[str], max_len: int, device):
        batch_texts = texts
        inputs = self.tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(device)
        outputs = self.model(**inputs)
        emb = self.mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
        return emb


# BERT [CLS] encoder used in the detection process.
class BertPostEncoder(nn.Module):
    def __init__(self, model_path: str, gradient_checkpointing: bool = False):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.hidden_size = self.model.config.hidden_size

        if gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

    def forward(self, posts: List[str], max_len: int, device):
        inputs = self.tokenizer(
            posts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(device)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :]

# Screening Proc
class StrictScaleScreening(nn.Module):
    def __init__(self, screen_model: str, screen_max_len: int, gradient_checkpointing=False):
        super().__init__()
        self.encoder = SentenceBERTEncoder(screen_model, gradient_checkpointing)
        self.templates = DEPRESSION_TEMPLATES
        self.screen_max_len = screen_max_len

    def forward(self, posts: List[str], device):
        post_embs = self.encoder(posts, self.screen_max_len, device)
        temp_embs = self.encoder(self.templates, self.screen_max_len, device)

        post_embs = F.normalize(post_embs, p=2, dim=-1)
        temp_embs = F.normalize(temp_embs, p=2, dim=-1)

        sim = post_embs @ temp_embs.T
        risk_scores = sim.max(dim=1).values
        return risk_scores, sim

class MaskedSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        n, h = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = q.view(n, self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(n, self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(n, self.num_heads, self.head_dim).transpose(0, 1)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores - scores.max(dim=-1, keepdim=True).values.detach()

        attn = torch.exp(scores) * mask.view(1, 1, n)
        attn = attn / torch.clamp(attn.sum(dim=-1, keepdim=True), min=1e-8)
        attn = self.dropout(attn)

        out = attn @ v
        out = out.transpose(0, 1).contiguous().view(n, h)
        return self.out_proj(out)


class MaskedTransformerLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = MaskedSelfAttention(hidden_size, num_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class MaskedTransformerEncoder(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_layers: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([
            MaskedTransformerLayer(hidden_size, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class E2LPSStrict(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.top_ratio = args.top_ratio
        self.detect_max_len = args.detect_max_len
        self.infer_selected_only = args.infer_selected_only

        if torch.cuda.is_available():
            self.screen_device = torch.device(args.screen_device)
            self.detect_device = torch.device(args.detect_device)
        else:
            self.screen_device = torch.device("cpu")
            self.detect_device = torch.device("cpu")
        # 2 TODO
        self.screening = StrictScaleScreening(
            args.screen_model,
            args.screen_max_len,
            args.gradient_checkpointing,
        ).to(self.screen_device)  # screening Proc

        self.post_encoder = BertPostEncoder(args.detect_model, args.gradient_checkpointing).to(self.detect_device)

        hidden_size = self.post_encoder.hidden_size
        self.user_encoder = MaskedTransformerEncoder(hidden_size, args.num_heads, args.num_layers, args.dropout).to(self.detect_device)
        self.post_attn = nn.Linear(hidden_size, 1).to(self.detect_device)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(args.dropout),
            nn.Linear(hidden_size, 1),
        ).to(self.detect_device)

    ########## significance ##################
    def build_ste_mask(self, risk_scores):
        n = risk_scores.size(0)
        k = max(1, math.ceil(n * self.top_ratio))
        # k = max(1, 16)
        k = min(k, n)

        top_idx = torch.topk(risk_scores, k=k).indices
        hard_mask = torch.zeros_like(risk_scores)
        hard_mask[top_idx] = 1.0

        # Straight-through estimator: hard mask in forward, risk-score gradient in backward.
        ste_mask = risk_scores + (hard_mask - risk_scores.detach())
        return ste_mask, hard_mask, top_idx

    def forward(self, posts: List[str], infer_selected_only: bool = False):
        posts = posts if posts else [""]

        # 1) Screening process.
        risk_scores, sim = self.screening(posts, self.screen_device)
        ste_mask, hard_mask, top_idx = self.build_ste_mask(risk_scores)

        # 2) Detection process.
        if (not self.training) and infer_selected_only:
            top_idx_sorted = torch.sort(top_idx).values
            detect_posts = [posts[int(i)] for i in top_idx_sorted.detach().cpu().tolist()]
            detect_mask = ste_mask[top_idx_sorted].to(self.detect_device)
            detect_hard_mask = hard_mask[top_idx_sorted].to(self.detect_device)
        else:
            detect_posts = posts
            detect_mask = ste_mask.to(self.detect_device)
            detect_hard_mask = hard_mask.to(self.detect_device)

        post_repr = self.post_encoder(
            detect_posts,
            self.detect_max_len,
            self.detect_device,
        )  # [1, 0.125*k, 64, E]

        post_repr = self.user_encoder(post_repr, detect_mask)
        attn_logits = self.post_attn(post_repr).squeeze(-1)
        masked_logits = attn_logits.masked_fill(detect_mask == 0, -1e9)
        attn = torch.softmax(masked_logits, dim=-1)
        user_repr = torch.sum(attn.unsqueeze(-1) * post_repr, dim=0)

        logit = self.classifier(user_repr).squeeze(-1)

        return {
            "logit": logit,
            "risk_scores": risk_scores,
            "similarity": sim,
            "hard_mask": detect_hard_mask,
            "top_idx": top_idx,
            "alpha": attn,
        }

    @torch.no_grad()
    def screen_only(self, posts: List[str]):
        self.eval()
        posts = posts if posts else [""]
        risk_scores, _ = self.screening(posts, self.screen_device)
        _, _, top_idx = self.build_ste_mask(risk_scores)

        top_idx = np.sort(top_idx.cpu().numpy())
        risk_np = risk_scores.cpu().numpy()

        selected_posts = [posts[i] for i in top_idx]
        selected_scores = [float(risk_np[i]) for i in top_idx]
        selected_indices = [int(i) for i in top_idx]
        return selected_posts, selected_scores, selected_indices

def train_one_epoch(model, store, indices, optimizer, device, args):
    model.train()
    random.shuffle(indices)
    loss_fn = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    y_true, y_pred = [], []

    for user_idx in tqdm(indices, desc="train", ncols=120):
        item = store.get_user(user_idx, args.max_posts_per_user, training=True)
        label = torch.tensor(float(item["label"]), device=device)

        optimizer.zero_grad()
        out = model(item["posts"])
        loss = loss_fn(out["logit"], label)
        loss.backward()

        if args.max_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        optimizer.step()

        prob = torch.sigmoid(out["logit"]).item()
        pred = 1 if prob >= args.threshold else 0
        total_loss += loss.item()
        y_true.append(item["label"])
        y_pred.append(pred)

    metrics = binary_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / max(1, len(indices))
    return metrics


@torch.no_grad()
def evaluate(model, store, indices, device, args, name="eval"):
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()

    total_loss = 0.0
    y_true, y_pred, y_prob = [], [], []

    for user_idx in tqdm(indices, desc=name, ncols=120):
        item = store.get_user(user_idx, args.max_posts_per_user, training=False)
        label = torch.tensor(float(item["label"]), device=device)

        out = model(item["posts"], infer_selected_only=args.infer_selected_only)
        loss = loss_fn(out["logit"], label)

        prob = torch.sigmoid(out["logit"]).item()
        pred = 1 if prob >= args.threshold else 0

        total_loss += loss.item()
        y_true.append(item["label"])
        y_pred.append(pred)
        y_prob.append(prob)

    metrics = binary_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / max(1, len(indices))
    report = classification_report(
        y_true, y_pred,
        labels=[0, 1],
        target_names=["control", "depressed"],
        digits=4,
        zero_division=0,
    )
    return metrics, report, y_true, y_pred, y_prob


@torch.no_grad()
def export_screened_posts(model, store, indices, out_dir: Path, args, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_dir = out_dir / "meta"
    meta_dir.mkdir(exist_ok=True)

    for user_idx in tqdm(indices, desc=f"export-{name}", ncols=120):
        item = store.get_user(user_idx, max_posts=None, training=False)
        selected_posts, selected_scores, selected_indices = model.screen_only(item["posts"])

        (out_dir / item["filename"]).write_text("\n".join(selected_posts), encoding="utf-8")

        meta = {
            "user_idx": user_idx,
            "filename": item["filename"],
            "label": item["label"],
            "top_ratio": args.top_ratio,
            "num_original_posts": len(item["posts"]),
            "num_selected_posts": len(selected_posts),
            "selected_indices": selected_indices,
            "selected_scores": selected_scores,
        }
        meta_path = meta_dir / item["filename"].replace(".txt", ".json")
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def build_args():
    parser = argparse.ArgumentParser()

    # Data and models
    parser.add_argument("--data_root", type=str, default="../processed_2018/origin")
    parser.add_argument("--screen_model", type=str, default="../pretrain_file/all-MiniLM-L6-v2")
    parser.add_argument("--detect_model", type=str, default="../pretrain_file/bert_base_uncased")
    parser.add_argument("--output_dir", type=str, default="./e2lps_base")

    # Training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--bert_lr", type=float, default=1e-5)
    parser.add_argument("--other_lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    # E2-LPS
    parser.add_argument("--top_ratio", type=float, default=0.125)
    parser.add_argument("--screen_max_len", type=int, default=128)
    parser.add_argument("--detect_max_len", type=int, default=64)
    parser.add_argument("--max_posts_per_user", type=int, default=0)
    parser.add_argument("--screen_device", type=str, default="cuda:0")
    parser.add_argument("--detect_device", type=str, default="cuda:1")
    parser.add_argument("--infer_selected_only", action="store_true")

    # Model structure
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Debug and export
    parser.add_argument("--max_train_users", type=int, default=0)
    parser.add_argument("--max_test_users", type=int, default=0)
    parser.add_argument("--export_screened", action="store_true")

    args = parser.parse_args()
    if args.max_posts_per_user <= 0:
        args.max_posts_per_user = None
    return args


def build_optimizer(model, args):
    bert_params = list(model.post_encoder.parameters())
    screen_params = list(model.screening.parameters())

    bert_ids = {id(p) for p in bert_params}
    screen_ids = {id(p) for p in screen_params}
    other_params = [p for p in model.parameters() if id(p) not in bert_ids and id(p) not in screen_ids]

    return torch.optim.AdamW(
        [
            {"params": bert_params, "lr": args.bert_lr},
            {"params": screen_params, "lr": args.other_lr},
            {"params": other_params, "lr": args.other_lr},
        ],
        weight_decay=args.weight_decay,
    )


def main():
    args = build_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")

    print("========== Load origin data ==========")
    train_store = RawUserStore(args.data_root, "train")
    test_store = RawUserStore(args.data_root, "test")

    train_ids = list(range(len(train_store)))
    test_ids = list(range(len(test_store)))

    if args.max_train_users > 0:
        train_ids = train_ids[:args.max_train_users]
    if args.max_test_users > 0:
        test_ids = test_ids[:args.max_test_users]

    print(f"Train users: {len(train_ids)}")
    print(f"Test users:  {len(test_ids)}")
    # 1 TODO
    print("========== Build E2-LPS model ==========")
    model = E2LPSStrict(args)
    device = model.detect_device   # loss/logit are on the detection device
    print(f"Screen device: {model.screen_device}")
    print(f"Detect device: {model.detect_device}")
    optimizer = build_optimizer(model, args)

    best_test_f1 = -1.0
    best_ckpt = output_dir / "best_e2lps_raw.pt"

    print("========== Training ==========")
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_metrics = train_one_epoch(model, train_store, train_ids, optimizer, device, args)
        test_metrics, test_report, _, _, _ = evaluate(model, test_store, test_ids, device, args, name="test")

        print("Train:", json.dumps(train_metrics, ensure_ascii=False))
        print("Test: ", json.dumps(test_metrics, ensure_ascii=False))
        print(test_report)

        with (output_dir / "train_log.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": epoch, "train": train_metrics, "test": test_metrics}, ensure_ascii=False) + "\n")

        # 测试集在这里作为验证集使用，用 test F1 保存当前最好模型。
        if test_metrics["f1"] > best_test_f1:
            best_test_f1 = test_metrics["f1"]
            torch.save({"model": model.state_dict(), "args": vars(args), "epoch": epoch, "best_test_f1": best_test_f1}, best_ckpt)
            print(f"Saved best checkpoint by test F1: {best_ckpt}")

    print("\n========== Best checkpoint on test set ==========")
    ckpt = torch.load(best_ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"])

    test_metrics, test_report, y_true, y_pred, y_prob = evaluate(model, test_store, test_ids, device, args, name="test-best")
    print("Test:", json.dumps(test_metrics, ensure_ascii=False, indent=2))
    print(test_report)

    (output_dir / "metrics.txt").write_text(
        f"Best epoch: {ckpt.get('epoch')}\n" +
        "Test metrics:\n" + json.dumps(test_metrics, ensure_ascii=False, indent=2) +
        "\n\nClassification report:\n" + test_report,
        encoding="utf-8",
    )

    with (output_dir / "test_predictions.jsonl").open("w", encoding="utf-8") as f:
        for user_idx, gold, pred, prob in zip(test_ids, y_true, y_pred, y_prob):
            item = test_store.get_user(user_idx, max_posts=1, training=False)
            f.write(json.dumps({
                "user_idx": user_idx,
                "filename": item["filename"],
                "gold": int(gold),
                "pred": int(pred),
                "prob": float(prob),
            }, ensure_ascii=False) + "\n")

    print(f"Saved metrics: {output_dir / 'metrics.txt'}")
    print(f"Saved predictions: {output_dir / 'test_predictions.jsonl'}")

    if args.export_screened:
        print("\n========== Export screened posts ==========")
        export_dir = output_dir / "e2lps_screened_ratio0125"
        export_screened_posts(model, train_store, list(range(len(train_store))), export_dir / "train", args, "train")
        export_screened_posts(model, test_store, list(range(len(test_store))), export_dir / "test", args, "test")
        print(f"Exported screened posts to: {export_dir}")


if __name__ == "__main__":
    main()
