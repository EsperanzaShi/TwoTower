

import pickle
from transformers import AutoTokenizer
from tqdm import tqdm
import os

# Load triplets from pickle file
triplets_path = "msmarco_triplets_test.pkl"
with open(triplets_path, "rb") as f:
    triplets = pickle.load(f)

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Tokenization settings
max_length = 128
tokenized_data = []

# Tokenize all triplets
print("Tokenizing triplets...")
for query, pos_doc, neg_doc in tqdm(triplets):
    encoded = {}

    # Tokenize query
    query_enc = tokenizer(query, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
    encoded["query_input_ids"] = query_enc["input_ids"].squeeze(0)
    encoded["query_attention_mask"] = query_enc["attention_mask"].squeeze(0)

    # Tokenize positive doc
    pos_enc = tokenizer(pos_doc, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
    encoded["positive_input_ids"] = pos_enc["input_ids"].squeeze(0)
    encoded["positive_attention_mask"] = pos_enc["attention_mask"].squeeze(0)

    # Tokenize negative doc
    neg_enc = tokenizer(neg_doc, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
    encoded["negative_input_ids"] = neg_enc["input_ids"].squeeze(0)
    encoded["negative_attention_mask"] = neg_enc["attention_mask"].squeeze(0)

    tokenized_data.append(encoded)

# Save tokenized data
output_path = "BERTtokenized_triplets_test.pkl"
with open(output_path, "wb") as f:
    pickle.dump(tokenized_data, f)

print(f"Tokenized triplets saved to {output_path}")