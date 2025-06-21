import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.efficient_kan import KANLinear


class MultiviewPerceptionAggregator(nn.Module):
    def __init__(self, hidden_dim):
        super(MultiviewPerceptionAggregator, self).__init__()
        self.hidden_dim = hidden_dim

        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

        self.view_embeddings = nn.Embedding(3, hidden_dim)
        self.fusion_kan = KANLinear(3 * hidden_dim, hidden_dim)

        self.axial_norm = nn.LayerNorm(hidden_dim)
        self.sagittal_norm = nn.LayerNorm(hidden_dim)
        self.coronal_norm = nn.LayerNorm(hidden_dim)

    def forward(self, axial_tokens, sagittal_tokens, coronal_tokens):
        batch_size, seq_len, _ = axial_tokens.size()

        # Project tokens to query, key, value for each view
        axial_query = self.query_proj(axial_tokens)
        axial_key = self.key_proj(axial_tokens)
        axial_value = self.value_proj(axial_tokens)

        sagittal_query = self.query_proj(sagittal_tokens)
        sagittal_key = self.key_proj(sagittal_tokens)
        sagittal_value = self.value_proj(sagittal_tokens)

        coronal_query = self.query_proj(coronal_tokens)
        coronal_key = self.key_proj(coronal_tokens)
        coronal_value = self.value_proj(coronal_tokens)

        # Get view embeddings
        view_emb = self.view_embeddings(torch.arange(3).to(axial_tokens.device))
        view_emb = view_emb.view(3, 1, -1).expand(-1, seq_len, -1)

        # Compute attention scores for each view
        axial_scores = torch.matmul(axial_query, axial_key.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.hidden_dim, device=axial_tokens.device))
        sagittal_scores = torch.matmul(sagittal_query, sagittal_key.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.hidden_dim, device=axial_tokens.device))
        coronal_scores = torch.matmul(coronal_query, coronal_key.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.hidden_dim, device=axial_tokens.device))

        # Apply view-aware attention
        axial_scores = axial_scores + torch.matmul(axial_query, view_emb[0].transpose(-2, -1))
        sagittal_scores = sagittal_scores + torch.matmul(sagittal_query, view_emb[1].transpose(-2, -1))
        coronal_scores = coronal_scores + torch.matmul(coronal_query, view_emb[2].transpose(-2, -1))

        # Apply softmax to get attention weights for each view
        axial_attn_weights = F.softmax(axial_scores, dim=-1)
        sagittal_attn_weights = F.softmax(sagittal_scores, dim=-1)
        coronal_attn_weights = F.softmax(coronal_scores, dim=-1)

        # Compute attended values for each view
        axial_attended = torch.matmul(axial_attn_weights, axial_value)
        sagittal_attended = torch.matmul(sagittal_attn_weights, sagittal_value)
        coronal_attended = torch.matmul(coronal_attn_weights, coronal_value)

        # Add & Norm for each view before concatenation
        axial_out = self.axial_norm(axial_attended + axial_tokens)
        sagittal_out = self.sagittal_norm(sagittal_attended + sagittal_tokens)
        coronal_out = self.coronal_norm(coronal_attended + coronal_tokens)

        # Concatenate attended values from all views
        attended_concat = torch.cat((axial_out, sagittal_out, coronal_out), dim=-1)

        # Fuse attended values using a linear projection
        fused_tokens = self.fusion_kan(attended_concat)

        return fused_tokens
