# precompute_doc_embeddings.py

import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from src.two_tower.model import TwoTowerModel
import numpy as np
import wandb

# ----- Config -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EMBEDDINGS_SAVE_PATH = "precomputed_doc_embeddings.npy"

# ----- Load Tokenizer and Model -----
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = TwoTowerModel(freeze_bert=True).to(device)

# Load model from wandb artifact
wandb.login()
artifact = wandb.use_artifact("week2_two_tower_neural_network/TwoTower/bert-encoders:v3", type="model")
artifact_dir = artifact.download()
model.doc_encoder.load_state_dict(torch.load(os.path.join(artifact_dir, "doc_encoder.pt"), map_location=device))
model.eval()

# ----- Load Corpus -----
with open(".data/BERTtokenized_triplets_test.pkl", "rb") as f:
    triplets = pickle.load(f)

corpus_docs = list({tuple(doc["positive_input_ids"]) for doc in triplets})
print(f"Total unique documents in corpus: {len(corpus_docs)}")

# ----- Dataset and Dataloader -----
class DocDataset(Dataset):
    def __init__(self, input_ids):
        self.input_ids = input_ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        ids = torch.tensor(self.input_ids[idx])
        mask = (ids != 0).long()
        return ids, mask

doc_dataset = DocDataset(corpus_docs)
doc_loader = DataLoader(doc_dataset, batch_size=BATCH_SIZE)

# ----- Encode Documents -----
all_embeddings = []

with torch.no_grad():
    for input_ids, attn_mask in doc_loader:
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        _, doc_embed = model(input_ids, attn_mask, input_ids, attn_mask)
        all_embeddings.append(doc_embed.cpu())

all_embeddings = torch.cat(all_embeddings, dim=0).numpy()

# ----- Save Embeddings -----
np.save(EMBEDDINGS_SAVE_PATH, all_embeddings)
print(f"✅ Saved {all_embeddings.shape[0]} doc embeddings to {EMBEDDINGS_SAVE_PATH}")