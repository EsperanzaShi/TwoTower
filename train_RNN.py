import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle
import wandb
import os

# ----- Config -----
wandb.init(project="TwoTower", job_type="sweep_training")
config = wandb.config

#🔢 Step 1: Token IDs from the pickle
# ----- Load dataset from artifact -----
artifact = wandb.use_artifact("TwoTower/msmarco-triplets:v0", type="dataset")
artifact_dir = artifact.download()
with open(os.path.join(artifact_dir, "BERTtokenized_triplets.pkl"), "rb") as f:
    triplets = pickle.load(f)[:10000]  # subset for dev/testing

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
        self.rnn = nn.RNN(embed_dim, embed_dim, batch_first=True)
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask):
        output, _ = self.rnn(x)  # (batch, seq_len, hidden)
        mask = mask.unsqueeze(-1).expand_as(output)
        x = (output * mask).sum(1) / mask.sum(1)
        return self.fc(x)

# ----- Document Tower (Simple Linear RNN) -----
class DocTower(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.rnn = nn.RNN(embed_dim, embed_dim, batch_first=True)
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask):
        output, _ = self.rnn(x)
        mask = mask.unsqueeze(-1).expand_as(output)
        x = (output * mask).sum(1) / mask.sum(1)
        return self.fc(x)

#🧠 Step 2: Embedding Layer
# ----- Initialize embedding layer -----
vocab_size = 30522  # Standard BERT vocab size
embedding_dim = 256
embedding = nn.Embedding(vocab_size, embedding_dim).to("cuda" if torch.cuda.is_available() else "cpu")

#🏗️ Step 3: Towers
# ----- Model setup -----
query_encoder = QryTower(embed_dim=embedding_dim).to("cuda" if torch.cuda.is_available() else "cpu")
doc_encoder = DocTower(embed_dim=embedding_dim).to("cuda" if torch.cuda.is_available() else "cpu")
optimizer = torch.optim.Adam(
    list(embedding.parameters()) + list(query_encoder.parameters()) + list(doc_encoder.parameters()),
    lr=float(config.learning_rate)
)
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
        q_input_ids = batch["query_input_ids"].to(device)
        q_mask = batch["query_attention_mask"].to(device)
        p_input_ids = batch["positive_input_ids"].to(device)
        p_mask = batch["positive_attention_mask"].to(device)
        n_input_ids = batch["negative_input_ids"].to(device)
        n_mask = batch["negative_attention_mask"].to(device)

        q_input = embedding(q_input_ids)
        p_input = embedding(p_input_ids)
        n_input = embedding(n_input_ids)

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

# ----- Save Models -----
os.makedirs("saved_models", exist_ok=True)
torch.save(query_encoder.state_dict(), "saved_models/query_encoder.pt")
torch.save(doc_encoder.state_dict(), "saved_models/doc_encoder.pt")
torch.save(optimizer.state_dict(), "saved_models/optimizer.pt")
print("✅ Saved model weights to ./saved_models/")

artifact = wandb.Artifact("rnn-encoders", type="model")
artifact.add_file("saved_models/query_encoder.pt")
artifact.add_file("saved_models/doc_encoder.pt")
artifact.add_file("saved_models/optimizer.pt")
wandb.log_artifact(artifact)

# ----- Final t-SNE Visualisation -----
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

print("🔍 Generating t-SNE for final embeddings...")
embeddings, labels = [], []
query_encoder.eval()
doc_encoder.eval()

loader = DataLoader(TripletDataset(triplets), batch_size=config.batch_size)
with torch.no_grad():
    for i, batch in enumerate(loader):
        if i > 6: break  # ~200 samples
        q_input_ids = batch["query_input_ids"].to(device)
        q_mask = batch["query_attention_mask"].to(device)
        p_input_ids = batch["positive_input_ids"].to(device)
        p_mask = batch["positive_attention_mask"].to(device)
        n_input_ids = batch["negative_input_ids"].to(device)
        n_mask = batch["negative_attention_mask"].to(device)

        q_input = embedding(q_input_ids)
        p_input = embedding(p_input_ids)
        n_input = embedding(n_input_ids)

        q_embed = query_encoder(q_input, q_mask)
        p_embed = doc_encoder(p_input, p_mask)
        n_embed = doc_encoder(n_input, n_mask)

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
plt.title(f"t-SNE of final embeddings (margin={config.triplet_margin})")
plt.tight_layout()
plt.savefig("tsne_plot.png")
wandb.log({"t-SNE": wandb.Image("tsne_plot.png")})

# ----- Cosine Similarity Histogram -----
from torch.nn.functional import cosine_similarity

print("📊 Generating cosine similarity histogram...")
pos_sims, neg_sims = [], []

with torch.no_grad():
    for i, batch in enumerate(loader):
        if i > 6: break
        q_input_ids = batch["query_input_ids"].to(device)
        q_mask = batch["query_attention_mask"].to(device)
        p_input_ids = batch["positive_input_ids"].to(device)
        p_mask = batch["positive_attention_mask"].to(device)
        n_input_ids = batch["negative_input_ids"].to(device)
        n_mask = batch["negative_attention_mask"].to(device)

        q_input = embedding(q_input_ids)
        p_input = embedding(p_input_ids)
        n_input = embedding(n_input_ids)

        q_embed = query_encoder(q_input, q_mask)
        pos_embed = doc_encoder(p_input, p_mask)
        neg_embed = doc_encoder(n_input, n_mask)

        pos_sim = cosine_similarity(q_embed, pos_embed, dim=1).cpu().numpy()
        neg_sim = cosine_similarity(q_embed, neg_embed, dim=1).cpu().numpy()
        pos_sims.extend(pos_sim)
        neg_sims.extend(neg_sim)

plt.figure()
plt.hist(pos_sims, bins=20, alpha=0.7, label="positive")
plt.hist(neg_sims, bins=20, alpha=0.7, label="negative")
plt.legend()
plt.title(f"Cosine similarity distributions (margin={config.triplet_margin})")
plt.tight_layout()
hist_path = f"cosine_hist_margin_{config.triplet_margin}.png"
plt.savefig(hist_path)
wandb.log({f"cosine_hist_margin_{config.triplet_margin}": wandb.Image(hist_path)})