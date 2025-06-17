import torch
import pickle
from model import TwoTowerModel
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

# ----- Load a few tokenized triplets -----
with open(".data/BERTtokenized_triplets.pkl", "rb") as f:
    triplets = pickle.load(f)[:3]  # Just 3 triplets to visualize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TwoTowerModel(freeze_bert=False).to(device)
model.eval()

embeddings = []
labels = []

with torch.no_grad():
    for i, triplet in enumerate(triplets):
        q_ids = triplet["query_input_ids"].unsqueeze(0).to(device)
        q_mask = triplet["query_attention_mask"].unsqueeze(0).to(device)
        p_ids = triplet["positive_input_ids"].unsqueeze(0).to(device)
        p_mask = triplet["positive_attention_mask"].unsqueeze(0).to(device)
        n_ids = triplet["negative_input_ids"].unsqueeze(0).to(device)
        n_mask = triplet["negative_attention_mask"].unsqueeze(0).to(device)

        q_embed, p_embed = model(q_ids, q_mask, p_ids, p_mask)
        _, n_embed = model(q_ids, q_mask, n_ids, n_mask)

        embeddings.extend([q_embed.cpu().numpy()[0], p_embed.cpu().numpy()[0], n_embed.cpu().numpy()[0]])
        labels.extend([f"query_{i}", f"positive_{i}", f"negative_{i}"])

# ---- Dimensionality reduction ----
tsne = TSNE(n_components=2, perplexity=5, random_state=42)
X_2d = tsne.fit_transform(np.array(embeddings))

# ---- Plot ----
plt.figure(figsize=(8, 6))
colors = {"query": "blue", "positive": "green", "negative": "red"}

for i, label in enumerate(labels):
    kind = label.split("_")[0]
    plt.scatter(X_2d[i, 0], X_2d[i, 1], c=colors[kind], label=label)

# Unique legend entries only
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.title("Individual Triplet Embedding Visualization")
plt.show()