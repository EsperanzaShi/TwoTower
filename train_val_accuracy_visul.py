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

# ----- Weights & Biases Init -----
training_config = {
    "model": "TwoTowerModel",
    "encoder": "bert-base-uncased",
    "epochs": epochs,
    "batch_size": 16,
    "triplet_margin": 1.0,  # default margin, overwritten by sweep config
    "optimizer": "AdamW",
    "learning_rate": 1e-5,
    "warmup_steps": 100,
    "device": str(device)
}

wandb.login(key="95ab75842f9b83eb3d3827739cdcb91239e81de7")
wandb.init(
    project="TwoTower",
    entity="week2_two_tower_neural_network",
    name="bert-finetune",
    config=training_config
)

# update margin from sweep config
training_config["triplet_margin"] = wandb.config.triplet_margin

# ----- Load tokenized triplets -----
artifact = wandb.use_artifact("TwoTower/msmarco-triplets:v0", type="dataset")
artifact_dir = artifact.download()
print("📦 Loading triplets from pickle...")
with open(os.path.join(artifact_dir, "BERTtokenized_triplets.pkl"), "rb") as f:
    train_triplets = pickle.load(f)[:1000]  # small batch for dev/test runs
print(f"✅ Triplets loaded: {len(train_triplets)} samples")


# ----- Load validation triplets -----
artifact_val = wandb.use_artifact("TwoTower/msmarco-triplets:v1", type="dataset")
artifact_val_dir = artifact_val.download()
val_path = os.path.join(artifact_val_dir, "BERTtokenized_triplets_val.pkl")
if not os.path.exists(val_path):
    raise FileNotFoundError(f"Validation file not found at {val_path}")
with open(val_path, "rb") as f:
    val_triplets = pickle.load(f)[:200]  # use first 1000 validation triplets

# ----- Dataset and Dataloader -----
class TripletDataset(Dataset):
    def __init__(self, triplets):
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return self.triplets[idx]

train_dataset = TripletDataset(train_triplets)
val_dataset = TripletDataset(val_triplets)

train_loader = DataLoader(train_dataset, batch_size=training_config["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=training_config["batch_size"])

lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=100,
    num_training_steps=len(train_loader) * epochs
)

training_config["num_training_steps"] = len(train_loader) * epochs

# ----- Training Loop -----
for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs} starting...")

    # ----- Training Phase -----
    model.train()
    total_train_loss = 0
    for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1} [Training]")):
        optimizer.zero_grad()
        q_input_ids = batch["query_input_ids"].to(device)
        q_mask = batch["query_attention_mask"].to(device)
        p_input_ids = batch["positive_input_ids"].to(device)
        p_mask = batch["positive_attention_mask"].to(device)
        n_input_ids = batch["negative_input_ids"].to(device)
        n_mask = batch["negative_attention_mask"].to(device)

        q_embed, pos_embed = model(q_input_ids, q_mask, p_input_ids, p_mask)
        _, neg_embed = model(q_input_ids, q_mask, n_input_ids, n_mask)

        loss = triplet_loss(q_embed, pos_embed, neg_embed, margin=training_config["triplet_margin"])
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    # ----- Validation Phase -----
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Validation]"):
            q_input_ids = batch["query_input_ids"].to(device)
            q_mask = batch["query_attention_mask"].to(device)
            p_input_ids = batch["positive_input_ids"].to(device)
            p_mask = batch["positive_attention_mask"].to(device)
            n_input_ids = batch["negative_input_ids"].to(device)
            n_mask = batch["negative_attention_mask"].to(device)

            q_embed, pos_embed = model(q_input_ids, q_mask, p_input_ids, p_mask)
            _, neg_embed = model(q_input_ids, q_mask, n_input_ids, n_mask)

            loss = triplet_loss(q_embed, pos_embed, neg_embed, margin=training_config["triplet_margin"])
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)

    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "margin": training_config["triplet_margin"]
    })

# ----- Save Models -----
os.makedirs("saved_models", exist_ok=True)
torch.save(model.query_encoder.state_dict(), "saved_models/query_encoder.pt")
torch.save(model.doc_encoder.state_dict(), "saved_models/doc_encoder.pt")
torch.save(optimizer.state_dict(), "saved_models/optimizer.pt")
print("✅ Saved model weights to ./saved_models/")

artifact = wandb.Artifact("bert-encoders", type="model")
artifact.add_file("saved_models/query_encoder.pt")
artifact.add_file("saved_models/doc_encoder.pt")
artifact.add_file("saved_models/optimizer.pt")
wandb.log_artifact(artifact)

# ----- Capture embeddings every 5 epochs for 3D t-SNE -----
all_tsne_embeddings = {}
capture_epochs = [0, 5, 10, 15, 20]

def extract_embeddings(loader):
    embeddings, labels = [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
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

loader = DataLoader(train_dataset, batch_size=training_config["batch_size"])

# Re-train and track embeddings at each capture epoch
model = TwoTowerModel(freeze_bert=False).to(device)
optimizer = torch.optim.AdamW([
    {"params": model.query_encoder.parameters(), "lr": 1e-5},
    {"params": model.doc_encoder.parameters(), "lr": 1e-5}
])
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=100,
    num_training_steps=len(train_loader) * epochs
)

for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        q_input_ids = batch["query_input_ids"].to(device)
        q_mask = batch["query_attention_mask"].to(device)
        p_input_ids = batch["positive_input_ids"].to(device)
        p_mask = batch["positive_attention_mask"].to(device)
        n_input_ids = batch["negative_input_ids"].to(device)
        n_mask = batch["negative_attention_mask"].to(device)
        q_embed, pos_embed = model(q_input_ids, q_mask, p_input_ids, p_mask)
        _, neg_embed = model(q_input_ids, q_mask, n_input_ids, n_mask)
        loss = triplet_loss(q_embed, pos_embed, neg_embed, margin=training_config["triplet_margin"])
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

    if epoch in capture_epochs:
        model.eval()
        embeds, labs = extract_embeddings(loader)
        tsne = TSNE(n_components=3, perplexity=30, random_state=42)
        X_3d = tsne.fit_transform(embeds)
        all_tsne_embeddings[epoch] = (X_3d, labs)

# ----- Interactive 3D t-SNE Plot with Plotly -----
fig = go.Figure()

colors = {"query": "blue", "positive": "green", "negative": "red"}

for epoch, (points, labels) in all_tsne_embeddings.items():
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
fig.update_layout(sliders=sliders, title="3D t-SNE of Embeddings Across Epochs", margin=dict(l=0, r=0, b=0, t=40))
plot_path = f"tsne_3d_evolution_margin_{training_config['triplet_margin']}.html"
fig.write_html(plot_path)
wandb.save(plot_path)