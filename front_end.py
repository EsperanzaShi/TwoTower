# front_end.py

import streamlit as st
import torch
import numpy as np
import pickle
import os
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import wandb

wandb.init(project="TwoTower", job_type="inference", anonymous="allow")

# ----- Config -----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TOP_K = 5
PICKLE_PATH = ".data/BERTtokenized_triplets_test.pkl"
DOC_EMBEDDINGS_PATH = "precomputed_doc_embeddings.npy"

# ----- Load Tokenizer -----
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# ----- Load query encoder from wandb -----
wandb.login()
artifact = wandb.use_artifact("week2_two_tower_neural_network/TwoTower/bert-encoders:v3", type="model")
artifact_dir = artifact.download()
query_encoder = BertModel.from_pretrained("bert-base-uncased")
query_encoder.load_state_dict(torch.load(os.path.join(artifact_dir, "query_encoder.pt"), map_location=DEVICE))
query_encoder.to(DEVICE).eval()

# ----- Load doc embeddings and text corpus -----
doc_embeddings = np.load(DOC_EMBEDDINGS_PATH)

with open(PICKLE_PATH, "rb") as f:
    triplets = pickle.load(f)

# Keep only unique positive documents
unique_docs = list({tuple(doc["positive_input_ids"]): doc for doc in triplets}.values())
doc_token_ids = [doc["positive_input_ids"] for doc in unique_docs]

# ----- Streamlit UI -----
st.set_page_config(page_title="Search", layout="centered")
st.markdown("<h1 style='text-align: center;'>Search Engine</h1>", unsafe_allow_html=True)

query = st.text_input("Enter your query:")
if st.button("Search") and query:
    encoded = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = encoded["input_ids"].to(DEVICE)
    attention_mask = encoded["attention_mask"].to(DEVICE)

    with torch.no_grad():
        query_embed = query_encoder(input_ids=input_ids, attention_mask=attention_mask).pooler_output

    similarities = cosine_similarity(query_embed.cpu().numpy(), doc_embeddings)[0]
    top_k_indices = similarities.argsort()[::-1][:TOP_K]

    st.subheader("Top 5 Results:")
    for i in top_k_indices:
        text = tokenizer.decode(doc_token_ids[i], skip_special_tokens=True)
        st.write(f"- {text}")