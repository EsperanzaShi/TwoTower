from datasets import load_dataset
import random
from tqdm import tqdm
import pickle
import os

# Load MS MARCO dataset
print("Loading MS MARCO dataset...")
dataset = load_dataset("microsoft/ms_marco", "v1.1", split="train")

# Step 1: Collect all unique passages from the entire dataset
print("Collecting all unique passages for negative sampling...")
all_docs = set()
for entry in tqdm(dataset, desc="Scanning all passages"):
    all_docs.update(entry["passages"]["passage_text"])
all_docs = list(all_docs)

# Step 2: Generate triplets
print("Generating triplets...")
triplets = []

for entry in tqdm(dataset, desc="Generating"):
    query = entry["query"]
    positive_passages = list(set(entry["passages"]["passage_text"]))

    # Skip if there are no passages
    if not positive_passages:
        continue

    # Sample the same number of negatives (up to 10) from all_docs excluding the positives
    negatives_pool = [doc for doc in all_docs if doc not in positive_passages]
    num_negatives = min(len(positive_passages), 10)
    negative_passages = random.sample(negatives_pool, k=num_negatives)

    # For each pair of positive and negative, create a triplet
    for pos_doc, neg_doc in zip(positive_passages, negative_passages):
        triplets.append((query, pos_doc, neg_doc))

print(f"Generated {len(triplets):,} triplets.")

# Display an example triplet before saving
if triplets:
    print("\nExample triplet:")
    print("Query:", triplets[0][0])
    print("Positive doc:", triplets[0][1])
    print("Negative doc:", triplets[0][2])

# Step 3: Save triplets to a pickle file in the current directory
output_path = "msmarco_triplets.pkl"

with open(output_path, "wb") as f:
    pickle.dump(triplets, f)

print(f"Triplets saved to {output_path}")
