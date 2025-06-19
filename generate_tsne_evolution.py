import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from two_tower.model import TwoTowerModel, triplet_loss
from transformers import get_scheduler
from sklearn.manifold import TSNE
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import numpy as np
import wandb


# ----- Model, Optimizer, Scheduler -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TwoTowerModel(freeze_bert=False).to(device)

optimizer = torch.optim.AdamW([
    {"params": model.query_encoder.parameters(), "lr": 1e-5},
    {"params": model.doc_encoder.parameters(), "lr": 1e-5}
])

epochs = 20

# ----- Capture embeddings every 5 epochs for 3D t-SNE -----
capture_epochs = [0, 5, 10, 15, 20]
margins = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4]

# ----- Load tokenized triplets -----
wandb.init(
    project="TwoTower",
    entity="week2_two_tower_neural_network",
    name="tsne_evolution"
)
 
artifact = wandb.use_artifact("TwoTower/msmarco-triplets:v0", type="dataset")
artifact_dir = artifact.download()
print("📦 Loading triplets from pickle...")
with open(os.path.join(artifact_dir, "BERTtokenized_triplets.pkl"), "rb") as f:
    train_triplets = pickle.load(f)[:1000]  # small batch for dev/test runs
print(f"✅ Triplets loaded: {len(train_triplets)} samples")

def extract_embeddings(train_loader, model):
    embeddings, labels = [], []
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            if i > 6: break  # ~200 samples
            q_input_ids = batch["query_input_ids"].to(device)
            q_mask = batch["query_attention_mask"].to(device)
            p_input_ids = batch["positive_input_ids"].to(device)
            p_mask = batch["positive_attention_mask"].to(device)
            n_input_ids = batch["negative_input_ids"].to(device)
            n_mask = batch["negative_attention_mask"].to(device)

            q_embed, p_embed = model(q_input_ids, q_mask, p_input_ids, p_mask)
            _, n_embed = model(q_input_ids, q_mask, n_input_ids, n_mask)

            embeddings.extend(q_embed.cpu().numpy())
            labels.extend(["query"] * q_embed.shape[0])
            embeddings.extend(p_embed.cpu().numpy())
            labels.extend(["positive"] * p_embed.shape[0])
            embeddings.extend(n_embed.cpu().numpy())
            labels.extend(["negative"] * n_embed.shape[0])
    return np.array(embeddings), labels

# ----- Dataset and Dataloader -----
class TripletDataset(Dataset):
    def __init__(self, triplets):
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return {
            "query_input_ids": torch.tensor(self.triplets[idx]["query_input_ids"]),
            "query_attention_mask": torch.tensor(self.triplets[idx]["query_attention_mask"]),
            "positive_input_ids": torch.tensor(self.triplets[idx]["positive_input_ids"]),
            "positive_attention_mask": torch.tensor(self.triplets[idx]["positive_attention_mask"]),
            "negative_input_ids": torch.tensor(self.triplets[idx]["negative_input_ids"]),
            "negative_attention_mask": torch.tensor(self.triplets[idx]["negative_attention_mask"]),
        }

train_dataset = TripletDataset(train_triplets)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)


all_tsne_embeddings = {}

for margin in margins:
    all_tsne_embeddings[margin] = {}
    tower_model = TwoTowerModel(freeze_bert=False).to(device)
    optimizer = torch.optim.AdamW([
        {"params": tower_model.query_encoder.parameters(), "lr": 1e-5},
        {"params": tower_model.doc_encoder.parameters(), "lr": 1e-5}
    ])
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=len(train_loader) * epochs
    )

    for epoch in range(epochs):
        tower_model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            q_input_ids = batch["query_input_ids"].to(device)
            q_mask = batch["query_attention_mask"].to(device)
            p_input_ids = batch["positive_input_ids"].to(device)
            p_mask = batch["positive_attention_mask"].to(device)
            n_input_ids = batch["negative_input_ids"].to(device)
            n_mask = batch["negative_attention_mask"].to(device)
            q_embed, pos_embed = tower_model(q_input_ids, q_mask, p_input_ids, p_mask)
            _, neg_embed = tower_model(q_input_ids, q_mask, n_input_ids, n_mask)
            loss = triplet_loss(q_embed, pos_embed, neg_embed, margin=margin)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

        if epoch in capture_epochs:
            tower_model.eval()
            embeds, labs = extract_embeddings(train_loader, tower_model)
            tsne = TSNE(n_components=3, perplexity=30, random_state=42)
            X_3d = tsne.fit_transform(embeds)
            all_tsne_embeddings[margin][epoch] = (X_3d, labs)

    # ----- Interactive 3D t-SNE Plot with Plotly per margin -----
    fig = go.Figure()

    colors = {"query": "blue", "positive": "green", "negative": "red"}

    for epoch, (points, labels) in all_tsne_embeddings[margin].items():
        for label in colors:
            idx = [i for i, l in enumerate(labels) if l == label]
            fig.add_trace(go.Scatter3d(
                x=points[idx, 0], y=points[idx, 1], z=points[idx, 2],
                mode='markers',
                marker=dict(size=3, color=colors[label]),
                name=f"{label} (Epoch {epoch})",
                visible=(epoch == 0)
            ))

    steps = []
    for i, epoch in enumerate(capture_epochs):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)}],
            label=f"Epoch {epoch}"
        )
        for j in range(i * 3, (i + 1) * 3):
            step["args"][0]["visible"][j] = True
        steps.append(step)

    sliders = [dict(active=0, currentvalue={"prefix": "Epoch: "}, steps=steps)]
    fig.update_layout(sliders=sliders, title=f"3D t-SNE of Embeddings Across Epochs (Margin {margin})", margin=dict(l=0, r=0, b=0, t=40))
    plot_path = f"tsne_3d_evolution_margin_{margin}.html"
    fig.write_html(plot_path)
    wandb.save(plot_path)