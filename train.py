import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
from tqdm import tqdm
from model import TwoTowerModel, triplet_loss

# ----- Load tokenized triplets -----
with open("data/BERTtokenized_triplets.pkl", "rb") as f:
    triplets = pickle.load(f)[:1000]  # use only the first 10,000 for faster training

# ----- Custom Dataset -----
class TripletDataset(Dataset):
    def __init__(self, triplets):
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        data = self.triplets[idx]
        return {k: v for k, v in data.items()}

# ----- Dataloader -----
dataset = TripletDataset(triplets)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# ----- Model -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TwoTowerModel(freeze_bert=False).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

# ----- Training Loop -----
epochs = 1
for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs} starting...")
    model.train()
    total_loss = 0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
        print(f"Epoch {epoch+1} - Batch {batch_idx+1}/{len(dataloader)}")
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
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    #model.safetensors is a secure and faster alternative to traditional .bin model weight files used by Hugging Face Transformers.