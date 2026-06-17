import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        padding = kernel_size // 2
        groups = 8 if out_channels % 8 == 0 else 1
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )


class ResidualRefineBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(channels, channels, 3),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8 if channels % 8 == 0 else 1, channels),
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.block(x))


class MultiScalePlaneMaskHead(nn.Module):
    """Query-based plane mask head with multi-layer DUSt3R fusion.

    DUSt3R decoder stages share the same token grid, but encode different
    semantic depths. This head first fuses encoder/shallow/middle/deep maps at
    32x32, decodes plane queries on that fused memory, and then predicts masks
    on learned 64x64 and 128x128 pixel features. A low-level RGB skip can be
    used only for boundary refinement; plane identity still comes from DUSt3R.
    """

    def __init__(
        self,
        input_dims=(1024, 768, 768, 768),
        hidden_dim=256,
        num_queries=8,
        num_decoder_layers=3,
        num_heads=8,
        output_size=128,
        use_rgb_skip=True,
    ):
        super().__init__()
        if output_size not in (32, 64, 128):
            raise ValueError("output_size must be one of 32, 64, 128")

        self.input_dims = tuple(int(value) for value in input_dims)
        self.hidden_dim = int(hidden_dim)
        self.num_queries = int(num_queries)
        self.output_size = int(output_size)
        self.use_rgb_skip = bool(use_rgb_skip)

        self.stage_names = ("encoder", "shallow", "middle", "deep")
        self.lateral = nn.ModuleDict(
            {
                name: nn.Conv2d(in_dim, hidden_dim, kernel_size=1)
                for name, in_dim in zip(self.stage_names, self.input_dims)
            }
        )
        self.stage_logits = nn.Parameter(torch.zeros(len(self.stage_names)))
        self.fuse = nn.Sequential(
            ConvNormAct(hidden_dim * len(self.stage_names), hidden_dim, 3),
            ResidualRefineBlock(hidden_dim),
        )

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

        self.refine32 = ResidualRefineBlock(hidden_dim)
        self.up64 = nn.Sequential(
            ConvNormAct(hidden_dim, hidden_dim, 3),
            ResidualRefineBlock(hidden_dim),
        )
        self.up128 = nn.Sequential(
            ConvNormAct(hidden_dim, hidden_dim, 3),
            ResidualRefineBlock(hidden_dim),
        )

        if self.use_rgb_skip:
            rgb_channels = max(hidden_dim // 4, 32)
            self.rgb_proj = nn.Sequential(
                ConvNormAct(3, rgb_channels, 3),
                ConvNormAct(rgb_channels, rgb_channels, 3),
            )
            self.rgb_fuse = nn.Sequential(
                ConvNormAct(hidden_dim + rgb_channels, hidden_dim, 3),
                ResidualRefineBlock(hidden_dim),
            )

        self.background32 = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        self.background64 = nn.Conv2d(hidden_dim, 1, kernel_size=1)
        self.background128 = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def _fuse_features(self, features):
        projected = []
        expected_hw = None
        for name in self.stage_names:
            if name not in features:
                raise KeyError(f"Missing multi-layer feature: {name}")
            value = self.lateral[name](features[name])
            if expected_hw is None:
                expected_hw = value.shape[-2:]
            elif value.shape[-2:] != expected_hw:
                value = F.interpolate(
                    value,
                    size=expected_hw,
                    mode="bilinear",
                    align_corners=False,
                )
            projected.append(value)

        weights = self.stage_logits.softmax(dim=0)
        weighted = [feature * weights[index] for index, feature in enumerate(projected)]
        fused = self.fuse(torch.cat(weighted, dim=1))
        return fused, weights

    def _mask_logits(self, mask_embeddings, pixel_features):
        normalized_pixels = F.normalize(pixel_features, dim=1)
        return torch.einsum(
            "bqc,bchw->bqhw",
            mask_embeddings,
            normalized_pixels,
        ) / math.sqrt(pixel_features.shape[1])

    def forward(self, features, rgb=None):
        fused32, stage_weights = self._fuse_features(features)
        batch_size = fused32.shape[0]
        memory = fused32.flatten(2).transpose(1, 2)
        queries = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        decoded = self.decoder(queries, memory)
        mask_embeddings = self.mask_embed(decoded)

        pixel32 = self.refine32(fused32)
        outputs = []
        outputs.append(
            {
                "mask_logits": self._mask_logits(mask_embeddings, pixel32),
                "background_logits": self.background32(pixel32),
            }
        )

        pixel64 = F.interpolate(
            pixel32,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )
        pixel64 = self.up64(pixel64)
        outputs.append(
            {
                "mask_logits": self._mask_logits(mask_embeddings, pixel64),
                "background_logits": self.background64(pixel64),
            }
        )

        pixel128 = F.interpolate(
            pixel64,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )
        pixel128 = self.up128(pixel128)
        if self.use_rgb_skip:
            if rgb is None:
                raise ValueError("rgb is required when use_rgb_skip=True")
            rgb = F.interpolate(
                rgb,
                size=pixel128.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            pixel128 = self.rgb_fuse(
                torch.cat((pixel128, self.rgb_proj(rgb)), dim=1)
            )
        outputs.append(
            {
                "mask_logits": self._mask_logits(mask_embeddings, pixel128),
                "background_logits": self.background128(pixel128),
            }
        )

        level_index = {32: 0, 64: 1, 128: 2}[self.output_size]
        selected = dict(outputs[level_index])
        selected["existence_logits"] = self.existence_head(decoded).squeeze(-1)
        selected["stage_weights"] = stage_weights

        auxiliary = []
        for index, output in enumerate(outputs[:level_index]):
            auxiliary.append(
                {
                    **output,
                    "existence_logits": selected["existence_logits"],
                    "resolution": (32, 64, 128)[index],
                }
            )
        selected["aux_outputs"] = auxiliary
        return selected

    def load_clean_checkpoint(self, checkpoint_state):
        """Warm-start query components from the previous clean 32x32 head."""
        own_state = self.state_dict()
        copied = []
        skipped = []
        transferable_prefixes = (
            "query_embed.",
            "decoder.",
            "mask_embed.",
            "existence_head.",
        )
        for key, value in checkpoint_state.items():
            if not key.startswith(transferable_prefixes):
                continue
            if key in own_state and own_state[key].shape == value.shape:
                own_state[key] = value
                copied.append(key)
            else:
                skipped.append(key)
        self.load_state_dict(own_state, strict=True)
        return copied, skipped
