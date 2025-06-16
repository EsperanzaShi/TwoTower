import torch
import torch.nn as nn
from transformers import AutoModel

class TwoTowerModel(nn.Module):
    def __init__(self, model_name="bert-base-uncased", freeze_bert=True, use_cls=True):
        super(TwoTowerModel, self).__init__()
        self.query_encoder = AutoModel.from_pretrained(model_name)
        self.doc_encoder = AutoModel.from_pretrained(model_name)

        self.use_cls = use_cls

        if freeze_bert:
            for param in self.query_encoder.parameters():
                param.requires_grad = False
            for param in self.doc_encoder.parameters():
                param.requires_grad = False

    def forward(self, query_input_ids, query_attention_mask,
                      doc_input_ids, doc_attention_mask):
        # Encode query
        query_output = self.query_encoder(input_ids=query_input_ids,
                                          attention_mask=query_attention_mask)
        # Encode document
        doc_output = self.doc_encoder(input_ids=doc_input_ids,
                                      attention_mask=doc_attention_mask)

        # Get sentence embeddings
        if self.use_cls:
            query_embed = query_output.last_hidden_state[:, 0]  # CLS token
            doc_embed = doc_output.last_hidden_state[:, 0]
        else:
            query_embed = (query_output.last_hidden_state * query_attention_mask.unsqueeze(-1)).sum(1)
            query_embed = query_embed / query_attention_mask.sum(1, keepdim=True)
            doc_embed = (doc_output.last_hidden_state * doc_attention_mask.unsqueeze(-1)).sum(1)
            doc_embed = doc_embed / doc_attention_mask.sum(1, keepdim=True)

        # Normalize embeddings
        query_embed = nn.functional.normalize(query_embed, p=2, dim=1)
        doc_embed = nn.functional.normalize(doc_embed, p=2, dim=1)

        return query_embed, doc_embed

def cosine_distance(a, b):
    return 1 - torch.sum(a * b, dim=1)

def triplet_loss(query_embed, pos_embed, neg_embed, margin=1.0):
    pos_dist = cosine_distance(query_embed, pos_embed)
    neg_dist = cosine_distance(query_embed, neg_embed)
    losses = torch.relu(pos_dist - neg_dist + margin)
    return losses.mean()