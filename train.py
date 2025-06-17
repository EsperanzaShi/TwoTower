import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model import TwoTowerModel, triplet_loss
from transformers import get_scheduler
from sklearn.manifold import TSNE
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

training_config = {
    "model": "TwoTowerModel",
    "encoder": "bert-base-uncased",
    "epochs": epochs,
    "batch_size": 64,
    "triplet_margin": 1.0,
    "optimizer": "AdamW",
    "learning_rate": 1e-5,
    "warmup_steps": 100,
    "device": str(device)
}

# ----- Weights & Biases Init -----
wandb.login(key="95ab75842f9b83eb3d3827739cdcb91239e81de7")
wandb.init(
    project="TwoTower",
    entity="week2_two_tower_neural_network",
    name="bert-finetune",
    config=training_config
)

# ----- Load tokenized triplets -----
artifact = wandb.use_artifact("week2_two_tower_neural_network/msmarco-triplets:latest", type="dataset")
artifact_dir = artifact.download()
with open(os.path.join(artifact_dir, "BERTtokenized_triplets.pkl"), "rb") as f:
    triplets = pickle.load(f)[:1000]  # small batch for dev/test runs

# ----- Dataset and Dataloader -----
class TripletDataset(Dataset):
    def __init__(self, triplets):
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return self.triplets[idx]

dataset = TripletDataset(triplets)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=100,
    num_training_steps=len(dataloader) * epochs
)

training_config["num_training_steps"] = len(dataloader) * epochs

# ----- Training Loop -----
for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs} starting...")
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
        optimizer.zero_grad()

        q_input_ids = batch["query_input_ids"].to(device)
        q_mask = batch["query_attention_mask"].to(device)
        p_input_ids = batch["positive_input_ids"].to(device)
        p_mask = batch["positive_attention_mask"].to(device)
        n_input_ids = batch["negative_input_ids"].to(device)
        n_mask = batch["negative_attention_mask"].to(device)

        q_embed, pos_embed = model(q_input_ids, q_mask, p_input_ids, p_mask)
        _, neg_embed = model(q_input_ids, q_mask, n_input_ids, n_mask)

        loss = triplet_loss(q_embed, pos_embed, neg_embed)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
    wandb.log({"epoch": epoch + 1, "loss": avg_loss})

# ----- Save Models -----
os.makedirs("saved_models", exist_ok=True)
torch.save(model.query_encoder.state_dict(), "saved_models/query_encoder.pt")
torch.save(model.doc_encoder.state_dict(), "saved_models/doc_encoder.pt")
torch.save(optimizer.state_dict(), "saved_models/optimizer.pt")
print("✅ Saved model weights to ./saved_models/")

wandb.save("saved_models/query_encoder.pt")
wandb.save("saved_models/doc_encoder.pt")
wandb.save("saved_models/optimizer.pt")

# ----- Final t-SNE Visualisation -----
print("🔍 Generating t-SNE for final embeddings...")
loader = DataLoader(dataset, batch_size=64)
model.eval()
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

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_2d = tsne.fit_transform(np.array(embeddings))

plt.figure(figsize=(10, 6))
colors = {"query": "blue", "positive": "green", "negative": "red"}
for label in colors:
    idx = [i for i, l in enumerate(labels) if l == label]
    plt.scatter(X_2d[idx, 0], X_2d[idx, 1], c=colors[label], label=label, alpha=0.6)
plt.legend()
plt.title("t-SNE of final embeddings")
plt.tight_layout()
plt.savefig("tsne_plot.png")
wandb.log({"t-SNE": wandb.Image("tsne_plot.png")})