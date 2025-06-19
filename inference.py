import os
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer
from src.two_tower.model import TwoTowerModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ----- Config -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_dim = 768
top_k = 5

# ----- Load Tokenizer and Model -----
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = TwoTowerModel(freeze_bert=True).to(device)
import wandb

# Initialize W&B and load model artifact
wandb.login()
artifact = wandb.use_artifact("week2_two_tower_neural_network/TwoTower/bert-encoders:v3", type="model")
artifact_dir = artifact.download()

# Load encoder states
model.query_encoder.load_state_dict(torch.load(os.path.join(artifact_dir, "query_encoder.pt"), map_location=device))
model.doc_encoder.load_state_dict(torch.load(os.path.join(artifact_dir, "doc_encoder.pt"), map_location=device))

# Optional: Load optimizer if needed
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
optimizer.load_state_dict(torch.load(os.path.join(artifact_dir, "optimizer.pt"), map_location=device))
model.eval()

# ----- Load Corpus -----
with open(".data/BERTtokenized_triplets_test.pkl", "rb") as f:
    triplets = pickle.load(f)

corpus_docs = list({tuple(doc["positive_input_ids"]) for doc in triplets})
print(f"Total unique documents in corpus: {len(corpus_docs)}")

# ----- Encode Corpus Documents -----
class DummyDataset(Dataset):
    def __init__(self, input_ids):
        self.input_ids = input_ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return torch.tensor(self.input_ids[idx])

doc_dataset = DummyDataset(corpus_docs)
doc_loader = DataLoader(doc_dataset, batch_size=64)

doc_embeddings = []

with torch.no_grad():
    for batch in doc_loader:
        batch = batch.to(device)
        attn_mask = (batch != 0).long()
        _, embeddings = model(batch, attn_mask, batch, attn_mask)
        doc_embeddings.append(embeddings.cpu())

doc_embeddings = np.load("precomputed_doc_embeddings.npy")
print("✅ Document embeddings precomputed")

# ----- Query Inference -----
while True:
    query = input("\nEnter a query (or 'exit' to quit): ")
    if query.lower() == "exit":
        break

    tokens = tokenizer(query, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    with torch.no_grad():
        q_embed, _ = model(input_ids, attention_mask, input_ids, attention_mask)
        q_embed = q_embed.cpu().numpy()

    similarities = cosine_similarity(q_embed, doc_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]

    print(f"Top {top_k} most similar documents:")
    for i, idx in enumerate(top_indices):
        print(f"{i+1}. Similarity: {similarities[idx]:.4f}, Doc ID: {idx}")
