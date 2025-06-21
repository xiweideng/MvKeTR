import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.efficient_kan import KANLinear


class CrossModalKnowledgeEnhancer(nn.Module):
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_model ** -0.5
        self.visual_query = nn.Linear(d_model, d_model)
        self.top_k_report_key = nn.Linear(d_model, d_model)
        self.top_k_report_value = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = KANLinear(d_model, d_model)

    def forward(self, visual_tokens, top_k_report_tokens):
        """
        Args:
            visual_tokens: CT visual tokens, shape [batch_size, visual_len, d_model]
            top_k_report_tokens: CT top-k report tokens, shape [batch_size, top_k, d_model]
        """
        batch_size = visual_tokens.size(0)

        # Project visual tokens to query
        query = self.visual_query(visual_tokens)
        res_query = query.view(batch_size, -1, self.d_model)  # Reshaped query
        query = query.view(batch_size, -1, self.num_heads, self.d_model // self.num_heads)
        query = query.permute(0, 2, 1, 3)  # Permute the dimensions of the query

        # Project top-k report tokens to key
        key = self.top_k_report_key(top_k_report_tokens)
        key = key.view(batch_size, -1, self.num_heads, self.d_model // self.num_heads)
        key = key.permute(0, 2, 3, 1)  # Permute the dimensions of the key

        # Project top-k report tokens to value
        value = self.top_k_report_value(top_k_report_tokens)
        value = value.view(batch_size, -1, self.num_heads, self.d_model // self.num_heads)
        value = value.permute(0, 2, 1, 3)  # Permute the dimensions of the value

        # Compute cross attention
        attn = torch.matmul(query, key)  # Attention scores
        attn = attn * self.scale  # Scale the attention scores
        attn = F.softmax(attn, dim=-1)  # Apply softmax to the attention scores
        attn = self.dropout(attn)  # Apply dropout to the attention scores

        out = torch.matmul(attn, value)  # Output after applying attention to the values
        out = out.permute(0, 2, 1, 3).contiguous()  # Permute the dimensions of the output
        out = out.view(batch_size, -1, self.d_model)  # Reshape the output
        out = self.norm(res_query + out)  # Apply layer normalization to the sum of res_query and output
        out = self.fc1(out)  # Apply the fully connected layer to the output
        return out
