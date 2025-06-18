import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import pickle
import wandb
import os

# ----- Config -----
wandb.init(project="TwoTower", job_type="sweep_training")
config = wandb.config

# ----- Load dataset from artifact -----
artifact = wandb.use_artifact("TwoTower/msmarco-triplets:v0", type="dataset")
artifact_dir = artifact.download()
with open(os.path.join(artifact_dir, "BERTtokenized_triplets.pkl"), "rb") as f:
    triplets = pickle.load(f)[:1000]  # subset for dev/testing

# ----- Dataset -----
class TripletDataset(Dataset):
    def __init__(self, triplets):
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return self.triplets[idx]

train_loader = DataLoader(TripletDataset(triplets), batch_size=config.batch_size, shuffle=True)

# ----- Query Tower (Simple Linear RNN) -----
class QryTower(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x, mask):
        # Average pooling over sequence dimension (batch, seq_len, embed_dim)
        mask = mask.unsqueeze(-1).expand_as(x)
        x = (x * mask).sum(1) / mask.sum(1)
        return self.mlp(x)

# ----- Document Tower (Simple Linear RNN) -----
class DocTower(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x, mask):
        mask = mask.unsqueeze(-1).expand_as(x)
        x = (x * mask).sum(1) / mask.sum(1)
        return self.mlp(x)

# ----- Initialize tokenizer and embedding layer -----
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
vocab_size = tokenizer.vocab_size
embedding_dim = 128
embedding = nn.Embedding(vocab_size, embedding_dim)

# ----- Model setup -----
query_encoder = QryTower(embed_dim=embedding_dim).to("cuda" if torch.cuda.is_available() else "cpu")
doc_encoder = DocTower(embed_dim=embedding_dim).to("cuda" if torch.cuda.is_available() else "cpu")
optimizer = torch.optim.Adam(list(embedding.parameters()) + list(query_encoder.parameters()) + list(doc_encoder.parameters()), lr=config.learning_rate)
device = next(query_encoder.parameters()).device

# ----- Triplet Loss -----
def triplet_loss(query, pos, neg, margin):
    pos_dist = F.pairwise_distance(query, pos)
    neg_dist = F.pairwise_distance(query, neg)
    return torch.relu(pos_dist - neg_dist + margin).mean()

# ----- Training Loop -----
for epoch in range(config.epochs):
    total_loss = 0
    for batch in train_loader:
        q_input = embedding(torch.tensor(batch["query_input_ids"]).to(device))
        q_mask = torch.tensor(batch["query_attention_mask"]).to(device)
        p_input = embedding(torch.tensor(batch["positive_input_ids"]).to(device))
        p_mask = torch.tensor(batch["positive_attention_mask"]).to(device)
        n_input = embedding(torch.tensor(batch["negative_input_ids"]).to(device))
        n_mask = torch.tensor(batch["negative_attention_mask"]).to(device)

        q_embed = query_encoder(q_input, q_mask)
        p_embed = doc_encoder(p_input, p_mask)
        n_embed = doc_encoder(n_input, n_mask)

        loss = triplet_loss(q_embed, p_embed, n_embed, config.triplet_margin)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    wandb.log({"loss": total_loss / len(train_loader), "epoch": epoch})
    print(f"Epoch {epoch+1}: Loss = {total_loss / len(train_loader):.4f}")