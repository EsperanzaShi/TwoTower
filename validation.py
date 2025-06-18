import wandb
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from transformers import BertModel, BertTokenizer
from src.two_tower.model import TwoTowerModel
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import os


import wandb

wandb.init(
    project="TwoTower",
    name="validate-10000-triplets",  # or another descriptive name
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

# ----- Load Model from W&B -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TwoTowerModel(freeze_bert=False).to(device)

model_artifact = wandb.use_artifact("TwoTower/bert-encoders:latest", type="model")
artifact_dir = model_artifact.download()
query_encoder_path = os.path.join(artifact_dir, "query_encoder.pt")
doc_encoder_path = os.path.join(artifact_dir, "doc_encoder.pt")

model.query_encoder.load_state_dict(torch.load(query_encoder_path, map_location=device))
model.doc_encoder.load_state_dict(torch.load(doc_encoder_path, map_location=device))
model.eval()

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

        #Encodes query and positive doc
        q_embed, pos_embed = model(q_input_ids, q_mask, p_input_ids, p_mask)
        #Encodes the negative doc with the same query again
        _, neg_embed = model(q_input_ids, q_mask, n_input_ids, n_mask)

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