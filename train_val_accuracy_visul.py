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
import matplotlib.pyplot as plt
import numpy as np
import wandb

# ----- Model, Optimizer, Scheduler -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

training_config = {
    "model": "TwoTowerModel",
    "encoder": "bert-base-uncased",
    "epochs": 20,
    "batch_size": 16,
    "triplet_margin": 1.0,  # default margin, overwritten by sweep config
    "optimizer": "AdamW",
    "learning_rate": 1e-5,
    "warmup_steps": 100,
    "device": str(device)
}

model = TwoTowerModel(freeze_bert=False).to(device)

optimizer = torch.optim.AdamW([
    {"params": model.query_encoder.parameters(), "lr": training_config["learning_rate"]},
    {"params": model.doc_encoder.parameters(), "lr": training_config["learning_rate"]}
])
epochs = training_config["epochs"]

# ----- Weights & Biases Init -----
wandb.login(key="95ab75842f9b83eb3d3827739cdcb91239e81de7")
wandb.init(
    project="TwoTower",
    entity="week2_two_tower_neural_network",
    name="bert-finetune",
    config=training_config
)

# update margin from sweep config (ensure robustness for local runs)
training_config["triplet_margin"] = wandb.config.get("triplet_margin", training_config["triplet_margin"])
training_config["batch_size"] = wandb.config.get("batch_size", training_config["batch_size"])
training_config["learning_rate"] = wandb.config.get("learning_rate", training_config["learning_rate"])
training_config["epochs"] = wandb.config.get("epochs", training_config["epochs"])

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
    num_warmup_steps=training_config["warmup_steps"],
    num_training_steps=len(train_loader) * training_config["epochs"]
)

training_config["num_training_steps"] = len(train_loader) * training_config["epochs"]

# ----- Training Loop -----
for epoch in range(training_config["epochs"]):
    print(f"\nEpoch {epoch+1}/{training_config['epochs']} starting...")

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

    print(f"Epoch {epoch+1}/{training_config['epochs']} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
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

artifact = wandb.Artifact(f"bert-encoders-margin-{training_config['triplet_margin']}", type="model")
artifact.add_file("saved_models/query_encoder.pt")
artifact.add_file("saved_models/doc_encoder.pt")
artifact.add_file("saved_models/optimizer.pt")
wandb.log_artifact(artifact)