import torch
import torch.nn as nn
import torch.nn.functional as F

# You need input a list of embeddings obtained from interaction-based ranker such as https://huggingface.co/cross-encoder
# Each embedding in this list is the [CLS] token with query-document as the input.
# The embedding in this list should be ranked as the relevance score from ranker.
# For example, the input list should be looked like this:
# [CLS_1, CLS_2, CLS_3,..., CLS_n]

class GetTextFeatures(nn.Module):
    def __init__(self, config, model_args):
        super().__init__()
        self.config = config
        self.model_args = model_args
        self.emb_layer = nn.Linear(config.emb_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

    def forward(self,inputs):
        inputs = torch.stack(inputs, dim=0)
        input_shape = inputs.size()
        seq_length = input_shape[1]
        inputs = self.emb_layer(inputs)
        position_ids = self.position_ids[:, : seq_length]
        position_emb = self.position_embeddings(position_ids)
        embs = inputs + position_emb
        embs = self.LayerNorm(embs)
        embs = self.dropout(embs)

        return embs

