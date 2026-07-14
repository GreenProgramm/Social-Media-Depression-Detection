import os
import re
import json
import pickle
import numpy as np
import pandas as pd
import torch
from collections import defaultdict, Counter
from sentence_transformers import SentenceTransformer
import xml.dom.minidom
import string
from tqdm import tqdm
from sklearn.cluster import KMeans, MiniBatchKMeans, Birch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

#%%
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
    "I am treating my depression."
]
# questionaire_single = [
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
#     "I have lost my interest in sex."
# ]  # 21
#
# #  %%
#
# depression_texts = [
#     "I feel depressed.",
#     "I am diagnosed with depression.",
#     "I am treating my depression."
# ]  # 3

#%%

with open("processed/miniLM_L6_embs.pkl", "rb") as f:
    data = pickle.load(f)

train_posts = data["train_posts"]
train_mappings = data["train_mappings"]
train_tags = data["train_labels"]
train_embs = data["train_embs"]
test_posts = data["test_posts"]
test_mappings = data["test_mappings"]
test_tags = data["test_labels"]
test_embs = data["test_embs"]

sbert = SentenceTransformer('paraphrase-MiniLM-L6-v2')

questionaire_single_embs = sbert.encode(questionaire_single)
depression_embs = sbert.encode(depression_texts)

# take care, require ~100G RAM
train_posts = np.array(train_posts)
test_posts = np.array(test_posts)

depression_pair_sim = cosine_similarity(train_embs, depression_embs)
print(depression_pair_sim.shape)  # (295023, 3)

depression_pair_sim_test = cosine_similarity(test_embs, depression_embs)
print(depression_pair_sim_test.shape)  # (236371, 3)


#%%
# topK = 16
# os.makedirs(f"processed/depress_sim{topK}", exist_ok=True)
# os.makedirs(f"processed/depress_sim{topK}/train", exist_ok=True)
# os.makedirs(f"processed/depress_sim{topK}/test", exist_ok=True)
# for i, (mapping, label) in enumerate(zip(train_mappings, train_tags)):
#     posts = train_posts[mapping]
#     sim_scores = depression_pair_sim[mapping, 0]
#     top_ids = sim_scores.argsort()[-topK:]
#     top_ids = np.sort(top_ids)  # sort in time order
#     sel_posts = posts[top_ids]
#     with open(f"processed/depress_sim{topK}/train/{i:06}_{label}.txt", "w") as f:
#         f.write("\n".join(x.replace("\n", " ") for x in sel_posts))
#
# for i, (mapping, label) in enumerate(zip(test_mappings, test_tags)):
#     posts = test_posts[mapping]
#     sim_scores = depression_pair_sim_test[mapping, 0]
#     top_ids = sim_scores.argsort()[-topK:]
#     top_ids = np.sort(top_ids)  # sort in time order
#     sel_posts = posts[top_ids]
#     with open(f"processed/depress_sim{topK}/test/{i:06}_{label}.txt", "w") as f:
#         f.write("\n".join(x.replace("\n", " ") for x in sel_posts))

#%%

dimension_sim_single = cosine_similarity(train_embs, questionaire_single_embs)
print(dimension_sim_single.shape)  # (295023, 21)

#%%

dimension_sim_single_test = cosine_similarity(test_embs, questionaire_single_embs)
print(dimension_sim_single_test.shape)  # (236371, 21)

#%%

combined_sim = np.concatenate([depression_pair_sim, dimension_sim_single], axis=1)  # (295023, 24)
combined_sim_test = np.concatenate([depression_pair_sim_test, dimension_sim_single_test], axis=1)  # (236371, 24)
print(combined_sim.shape, combined_sim_test.shape)

#%%

topK = 16
os.makedirs(f"processed/combined_maxsim{topK}", exist_ok=True)
os.makedirs(f"processed/combined_maxsim{topK}/train", exist_ok=True)
os.makedirs(f"processed/combined_maxsim{topK}/test", exist_ok=True)
for i, (mapping, label) in enumerate(zip(train_mappings, train_tags)):
    posts = train_posts[mapping]
    sim_scores = combined_sim[mapping].max(1)
    top_ids = sim_scores.argsort()[-topK:]
    top_ids = np.sort(top_ids)  # sort in time order
    sel_posts = posts[top_ids]
    with open(f"processed/combined_maxsim{topK}/train/{i:06}_{label}.txt", "w") as f:
        f.write("\n".join(x.replace("\n", " ") for x in sel_posts))

for i, (mapping, label) in enumerate(zip(test_mappings, test_tags)):
    posts = test_posts[mapping]
    sim_scores = combined_sim_test[mapping].max(1)
    top_ids = sim_scores.argsort()[-topK:]
    top_ids = np.sort(top_ids)  # sort in time order
    sel_posts = posts[top_ids]
    with open(f"processed/combined_maxsim{topK}/test/{i:06}_{label}.txt", "w") as f:
        f.write("\n".join(x.replace("\n", " ") for x in sel_posts))

#%%

# topK = 16
# os.makedirs(f"processed/questionaire_maxsim{topK}", exist_ok=True)
# os.makedirs(f"processed/questionaire_maxsim{topK}/train", exist_ok=True)
# os.makedirs(f"processed/questionaire_maxsim{topK}/test", exist_ok=True)
# for i, (mapping, label) in enumerate(zip(train_mappings, train_tags)):
#     posts = train_posts[mapping]
#     sim_scores = dimension_sim_single[mapping].max(1)
#     top_ids = sim_scores.argsort()[-topK:]
#     top_ids = np.sort(top_ids)  # sort in time order
#     sel_posts = posts[top_ids]
#     with open(f"processed/questionaire_maxsim{topK}/train/{i:06}_{label}.txt", "w") as f:
#         f.write("\n".join(x.replace("\n", " ") for x in sel_posts))
#
# for i, (mapping, label) in enumerate(zip(test_mappings, test_tags)):
#     posts = test_posts[mapping]
#     sim_scores = dimension_sim_single_test[mapping].max(1)
#     top_ids = sim_scores.argsort()[-topK:]
#     top_ids = np.sort(top_ids)  # sort in time order
#     sel_posts = posts[top_ids]
#     with open(f"processed/questionaire_maxsim{topK}/test/{i:06}_{label}.txt", "w") as f:
#         f.write("\n".join(x.replace("\n", " ") for x in sel_posts))

#%%

# topK = 16
# os.makedirs(f"processed/last{topK}", exist_ok=True)
# os.makedirs(f"processed/last{topK}/train", exist_ok=True)
# os.makedirs(f"processed/last{topK}/test", exist_ok=True)
# for i, (mapping, label) in enumerate(zip(train_mappings, train_tags)):
#     posts = train_posts[mapping]
#     sel_posts = posts[-topK:]
#     with open(f"processed/last{topK}/train/{i:06}_{label}.txt", "w") as f:
#         f.write("\n".join(x.replace("\n", " ") for x in sel_posts))
#
# for i, (mapping, label) in enumerate(zip(test_mappings, test_tags)):
#     posts = test_posts[mapping]
#     sel_posts = posts[-topK:]
#     with open(f"processed/last{topK}/test/{i:06}_{label}.txt", "w") as f:
#         f.write("\n".join(x.replace("\n", " ") for x in sel_posts))

#%%
