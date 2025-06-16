import pickle
from transformers import AutoTokenizer

with open("data/msmarco_triplets.pkl", "rb") as f:
    triplets = pickle.load(f)

print(f"Total triplets: {len(triplets)}")

# Show 3 examples
for i in range(3):
    query, pos, neg = triplets[i]
    print(f"\nTriplet {i+1}")
    print("Query:", query)
    print("Positive Doc:", pos)
    print("Negative Doc:", neg)



with open("data/BERTtokenized_triplets.pkl", "rb") as f:
    tokenized = pickle.load(f)

print(f"Total tokenized triplets: {len(tokenized)}")

# View structure of one entry
example = tokenized[0]
for key, value in example.items():
    print(f"{key}: {value.shape} → {value.tolist()[:10]}...")



tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

decoded_query = tokenizer.decode(tokenized[0]["query_input_ids"])
print("Decoded query text:", decoded_query)