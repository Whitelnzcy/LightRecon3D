import torch
import torch.nn as nn
import torch.nn.functional as F


class PlaneMaskHead(nn.Module):
    """Query-based bounded plane mask predictor."""

    def __init__(
        self,
        feature_dim=768,
        hidden_dim=256,
        num_queries=8,
        num_decoder_layers=3,
        num_heads=8,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.feature_proj = nn.Conv2d(feature_dim, hidden_dim, kernel_size=1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
            norm=nn.LayerNorm(hidden_dim),
        )
        self.mask_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.existence_head = nn.Linear(hidden_dim, 1)
        self.background_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, feature_map):
        pixel_features = self.feature_proj(feature_map)
        batch_size, channels, height, width = pixel_features.shape
        memory = pixel_features.flatten(2).transpose(1, 2)
        queries = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        decoded = self.decoder(queries, memory)
        mask_embeddings = self.mask_embed(decoded)
        mask_logits = torch.einsum(
            "bqc,bchw->bqhw",
            mask_embeddings,
            F.normalize(pixel_features, dim=1),
        ) / (channels ** 0.5)
        existence_logits = self.existence_head(decoded).squeeze(-1)
        background_logits = self.background_head(pixel_features)
        return {
            "mask_logits": mask_logits,
            "existence_logits": existence_logits,
            "background_logits": background_logits,
        }
