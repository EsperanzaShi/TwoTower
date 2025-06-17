import torch
import pickle
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from model import TwoTowerModel
from sklearn.manifold import TSNE
import numpy as np

# ----- Load tokenized triplets -----
with open(".data/BERTtokenized_triplets.pkl", "rb") as f:
    triplets = pickle.load(f)[:200]  # use 200 examples for clarity

# ----- Custom Dataset -----
class TripletDataset(Dataset):
    def __init__(self, triplets):
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return self.triplets[idx]

dataset = TripletDataset(triplets)
dataloader = DataLoader(dataset, batch_size=16)

# ----- Load trained model -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TwoTowerModel(freeze_bert=False)
model.query_encoder.load_state_dict(torch.load("saved_models/query_encoder.pt"))
model.doc_encoder.load_state_dict(torch.load("saved_models/doc_encoder.pt"))
model.to(device)
model.eval()

# ----- Collect embeddings -----
query_embeds, pos_embeds, neg_embeds = [], [], []

with torch.no_grad():
    for batch in dataloader:
        q_input_ids = batch["query_input_ids"].to(device)
        q_mask = batch["query_attention_mask"].to(device)
        p_input_ids = batch["positive_input_ids"].to(device)
        p_mask = batch["positive_attention_mask"].to(device)
        n_input_ids = batch["negative_input_ids"].to(device)
        n_mask = batch["negative_attention_mask"].to(device)

        q_embed, p_embed = model(q_input_ids, q_mask, p_input_ids, p_mask)
        _, n_embed = model(q_input_ids, q_mask, n_input_ids, n_mask)

        query_embeds.append(q_embed.cpu())
        pos_embeds.append(p_embed.cpu())
        neg_embeds.append(n_embed.cpu())

# ----- Prepare for t-SNE -----
q_arr = torch.cat(query_embeds).numpy()
p_arr = torch.cat(pos_embeds).numpy()
n_arr = torch.cat(neg_embeds).numpy()

X = np.vstack([q_arr, p_arr, n_arr])
y = ["query"] * len(q_arr) + ["positive"] * len(p_arr) + ["negative"] * len(n_arr)

# ----- t-SNE and Plot -----
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_2d = tsne.fit_transform(X)

plt.figure(figsize=(10, 6))
colors = {"query": "blue", "positive": "green", "negative": "red"}
for label in colors:
    idxs = [i for i, lbl in enumerate(y) if lbl == label]
    plt.scatter(X_2d[idxs, 0], X_2d[idxs, 1], c=colors[label], label=label, alpha=0.6)

plt.legend()
plt.title("t-SNE of Query, Positive, Negative Embeddings (after training)")
plt.savefig("tsne_after_training1epoch.png")
plt.show()
