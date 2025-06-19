import wandb
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import matplotlib.pyplot as plt
import os


import wandb

parser = argparse.ArgumentParser()
parser.add_argument("--model_version", type=str, default="latest", help="Model artifact version to use (e.g., v1, v2...)")
args = parser.parse_args()

wandb.init(
    project="TwoTower",
    name=f"validate-RNN-{args.model_version}",
    job_type="validation"
)
# ----- Load tokenized validation triplets from W&B -----
artifact = wandb.use_artifact("TwoTower/msmarco-triplets:v1", type="dataset")
artifact_dir = artifact.download()
val_path = os.path.join(artifact_dir, "BERTtokenized_triplets_val.pkl")
if not os.path.exists(val_path):
    raise FileNotFoundError(f"Validation file not found at {val_path}")

print("📦 Loading validation triplets from pickle...")
with open(val_path, "rb") as f:
    val_triplets = pickle.load(f)
print(f"✅ Validation triplets loaded: {len(val_triplets)} samples")

# ----- Dataset -----
class TripletDataset(Dataset):
    def __init__(self, triplets):
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return self.triplets[idx]

# ----- Load RNN Model from W&B -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load embedding layer
embedding_dim = 128
vocab_size = 30522  # Match training config
embedding = torch.nn.Embedding(vocab_size, embedding_dim).to(device)

# Define towers (same as training)
class QryTower(torch.nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.rnn = torch.nn.RNN(embed_dim, embed_dim, batch_first=True)
        self.fc = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        _, h = self.rnn(x)
        return self.fc(h.squeeze(0))

class DocTower(torch.nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.rnn = torch.nn.RNN(embed_dim, embed_dim, batch_first=True)
        self.fc = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        _, h = self.rnn(x)
        return self.fc(h.squeeze(0))

qry_tower = QryTower(embedding_dim).to(device)
doc_tower = DocTower(embedding_dim).to(device)

# Load state_dict
model_artifact = wandb.use_artifact(f"TwoTower/rnn-encoders:{args.model_version}", type="model")
artifact_dir = model_artifact.download()
qry_tower.load_state_dict(torch.load(os.path.join(artifact_dir, "query_encoder.pt"), map_location=device))
doc_tower.load_state_dict(torch.load(os.path.join(artifact_dir, "doc_encoder.pt"), map_location=device))

qry_tower.eval()
doc_tower.eval()

# ----- Dataloader -----
val_loader = DataLoader(TripletDataset(val_triplets), batch_size=16)

# ----- Validation Loop -----
correct = 0
total = 0

with torch.no_grad():
    for batch in val_loader:
        q_input_ids = batch["query_input_ids"].to(device)
        q_mask = batch["query_attention_mask"].to(device)
        p_input_ids = batch["positive_input_ids"].to(device)
        p_mask = batch["positive_attention_mask"].to(device)
        n_input_ids = batch["negative_input_ids"].to(device)
        n_mask = batch["negative_attention_mask"].to(device)

        q_input = embedding(q_input_ids)
        p_input = embedding(p_input_ids)
        q_embed = qry_tower(q_input)
        pos_embed = doc_tower(p_input)

        n_input = embedding(n_input_ids)
        neg_embed = doc_tower(n_input)

        pos_sim = torch.cosine_similarity(q_embed, pos_embed, dim=1) 
        neg_sim = torch.cosine_similarity(q_embed, neg_embed, dim=1)

        correct += (pos_sim > neg_sim).sum().item()
        total += q_input_ids.size(0)

val_acc = correct / total
wandb.log({"val_accuracy": val_acc})
print(f"✅ Validation Accuracy: {val_acc:.4f}")

plt.hist(pos_sim.cpu().numpy(), bins=30, alpha=0.6, label="positive")
plt.hist(neg_sim.cpu().numpy(), bins=30, alpha=0.6, label="negative")
plt.legend()
plt.title("Cosine similarity distributions")
wandb.log({"val_similarity_histogram": wandb.Image(plt)})
plt.close()
