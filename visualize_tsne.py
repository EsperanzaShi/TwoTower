import numpy as np
import torch
import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from model import TwoTowerModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

print("🔹 Loading tokenized triplets...")
# ---- Load tokenized triplets (small subset is enough) ----
with open("data/BERTtokenized_triplets.pkl", "rb") as f:
    triplets = pickle.load(f)[:200]  # visualize only 200 for clarity

print("🔹 Preparing dataset and dataloader...")
class TripletDataset(Dataset):
    def __init__(self, triplets):
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return self.triplets[idx]

dataset = TripletDataset(triplets)
dataloader = DataLoader(dataset, batch_size=16)

print("🔹 Loading trained model...")
# ---- Load trained model ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TwoTowerModel(freeze_bert=False).to(device)
model.eval()

print("🔹 Collecting embeddings...")
# ---- Collect embeddings ----
query_embeds = []
pos_embeds = []
neg_embeds = []

with torch.no_grad():
    for batch in tqdm(dataloader, desc="Collecting embeddings"):
        q_input_ids = batch["query_input_ids"].to(device)
        q_mask = batch["query_attention_mask"].to(device)
        p_input_ids = batch["positive_input_ids"].to(device)
        p_mask = batch["positive_attention_mask"].to(device)
        n_input_ids = batch["negative_input_ids"].to(device)
        n_mask = batch["negative_attention_mask"].to(device)

        q_embed, pos_embed = model(q_input_ids, q_mask, p_input_ids, p_mask)
        _, neg_embed = model(q_input_ids, q_mask, n_input_ids, n_mask)

        query_embeds.append(q_embed.cpu())
        pos_embeds.append(pos_embed.cpu())
        neg_embeds.append(neg_embed.cpu())

print("🔹 Preparing embeddings for t-SNE...")
# ---- Stack and prepare for t-SNE ----
query_embeds = torch.cat(query_embeds).numpy()
pos_embeds = torch.cat(pos_embeds).numpy()
neg_embeds = torch.cat(neg_embeds).numpy()

X = np.vstack([query_embeds, pos_embeds, neg_embeds])
y = ["query"] * len(query_embeds) + ["positive"] * len(pos_embeds) + ["negative"] * len(neg_embeds)

# ---- t-SNE ----
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_2d = tsne.fit_transform(X)

print("🔹 Plotting t-SNE visualization...")
# ---- Plot ----
colors = {"query": "blue", "positive": "green", "negative": "red"}
plt.figure(figsize=(10, 8))
for label in ["query", "positive", "negative"]:
    idxs = [i for i, tag in enumerate(y) if tag == label]
    plt.scatter(X_2d[idxs, 0], X_2d[idxs, 1], c=colors[label], label=label, alpha=0.6)

plt.legend()
plt.title("t-SNE of Query, Positive and Negative Embeddings")
plt.show()