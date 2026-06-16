import torch
import torch.nn as nn
import torch.nn.functional as F


def _pixel_shuffle_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1),
        nn.GroupNorm(8, out_channels * 4),
        nn.GELU(),
        nn.PixelShuffle(2),
    )


class PlaneMaskHead(nn.Module):
    """Query plane masks with residual 32->64->128 refinement."""

    def __init__(
        self,
        feature_dim=768,
        encoder_feature_dim=1024,
        shallow_feature_dim=768,
        middle_feature_dim=768,
        hidden_dim=256,
        num_queries=8,
        num_decoder_layers=3,
        num_heads=8,
        use_rgb_edge=True,
        refinement_margin=0.55,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.use_rgb_edge = use_rgb_edge
        self.refinement_margin = refinement_margin

        # These names intentionally match the original coarse head.
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
        self.boundary_head32 = nn.Conv2d(hidden_dim, 1, kernel_size=1)

        self.middle_proj = nn.Conv2d(middle_feature_dim, hidden_dim, kernel_size=1)
        self.encoder_proj = nn.Conv2d(encoder_feature_dim, hidden_dim, kernel_size=1)
        self.shallow_proj = nn.Conv2d(shallow_feature_dim, hidden_dim, kernel_size=1)
        self.middle_up = _pixel_shuffle_block(
            hidden_dim * 2 + num_queries + 1,
            hidden_dim,
        )
        self.middle_delta = nn.Conv2d(hidden_dim, num_queries + 1, kernel_size=1)
        self.boundary_head64 = nn.Conv2d(hidden_dim, 1, kernel_size=1)

        self.fine_semantic_up = _pixel_shuffle_block(
            hidden_dim * 3 + num_queries + 1,
            hidden_dim,
        )
        edge_channels = 32
        self.rgb_edge_stem = nn.Sequential(
            nn.Conv2d(3, edge_channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, edge_channels),
            nn.GELU(),
            nn.Conv2d(edge_channels, edge_channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, edge_channels),
            nn.GELU(),
        )
        fine_channels = hidden_dim + num_queries + 1
        if use_rgb_edge:
            fine_channels += edge_channels
        self.fine_fuse = nn.Sequential(
            nn.Conv2d(fine_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        )
        self.fine_delta = nn.Conv2d(hidden_dim, num_queries + 1, kernel_size=1)
        self.boundary_head128 = nn.Conv2d(hidden_dim, 1, kernel_size=1)

        # Zero gates preserve the old coarse result at initialization.
        self.alpha64 = nn.Parameter(torch.zeros(()))
        self.alpha128 = nn.Parameter(torch.zeros(()))

    def coarse_modules(self):
        return (
            self.feature_proj,
            self.query_embed,
            self.decoder,
            self.mask_embed,
            self.existence_head,
            self.background_head,
        )

    def coarse_parameters(self):
        for module in self.coarse_modules():
            yield from module.parameters()

    def refinement_parameters(self):
        coarse_ids = {id(parameter) for parameter in self.coarse_parameters()}
        for parameter in self.parameters():
            if id(parameter) not in coarse_ids:
                yield parameter

    def load_coarse_state_dict(self, state_dict):
        incompatible = self.load_state_dict(state_dict, strict=False)
        missing_coarse = [
            key
            for key in incompatible.missing_keys
            if not key.startswith(
                (
                    "middle_",
                    "encoder_",
                    "shallow_",
                    "fine_",
                    "rgb_edge_",
                    "boundary_",
                    "alpha64",
                    "alpha128",
                )
            )
        ]
        if incompatible.unexpected_keys or missing_coarse:
            raise RuntimeError(
                "Incompatible coarse checkpoint: "
                f"missing={missing_coarse}, unexpected={incompatible.unexpected_keys}"
            )
        return incompatible

    def _coarse_forward(self, feature_map):
        pixel_features = self.feature_proj(feature_map)
        batch_size, channels, _, _ = pixel_features.shape
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
        return mask_logits, existence_logits, background_logits, pixel_features

    @staticmethod
    def _uncertainty(class_logits):
        probabilities = class_logits.softmax(dim=1)
        top2 = probabilities.topk(k=2, dim=1).values
        return (1.0 - (top2[:, :1] - top2[:, 1:2])).clamp(0.0, 1.0)

    def _refinement_gate(self, class_logits):
        probabilities = class_logits.softmax(dim=1)
        top2 = probabilities.topk(k=2, dim=1).values
        margin = top2[:, :1] - top2[:, 1:2]
        return ((self.refinement_margin - margin) / self.refinement_margin).clamp(
            0.0,
            1.0,
        )

    @staticmethod
    def _assert_spatial(feature, expected_hw, name):
        if feature.shape[-2:] != expected_hw:
            raise ValueError(
                f"{name} has spatial shape {tuple(feature.shape[-2:])}, "
                f"expected {expected_hw}"
            )

    def forward(
        self,
        feature_map,
        middle_feature=None,
        shallow_feature=None,
        encoder_feature=None,
        image=None,
    ):
        middle_feature = feature_map if middle_feature is None else middle_feature
        shallow_feature = feature_map if shallow_feature is None else shallow_feature
        mask32, existence, background32, coarse_pixels = self._coarse_forward(feature_map)
        coarse_hw = mask32.shape[-2:]
        self._assert_spatial(middle_feature, coarse_hw, "middle_feature")
        self._assert_spatial(shallow_feature, coarse_hw, "shallow_feature")
        if encoder_feature is not None:
            self._assert_spatial(encoder_feature, coarse_hw, "encoder_feature")

        class32 = torch.cat((mask32, background32), dim=1)
        middle_pixels = self.middle_up(
            torch.cat(
                (
                    coarse_pixels,
                    self.middle_proj(middle_feature),
                    class32,
                ),
                dim=1,
            )
        )
        class64_base = F.interpolate(
            class32,
            scale_factor=2.0,
            mode="bilinear",
            align_corners=False,
        )
        gate64 = F.interpolate(
            self._refinement_gate(class32).detach(),
            size=class64_base.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        class64 = (
            class64_base
            + self.alpha64 * gate64 * self.middle_delta(middle_pixels)
        )

        shallow64 = F.interpolate(
            self.shallow_proj(shallow_feature),
            size=middle_pixels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        if encoder_feature is None:
            encoder64 = torch.zeros_like(shallow64)
        else:
            encoder64 = F.interpolate(
                self.encoder_proj(encoder_feature),
                size=middle_pixels.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        fine_semantic = self.fine_semantic_up(
            torch.cat((middle_pixels, shallow64, encoder64, class64), dim=1)
        )
        class128_base = F.interpolate(
            class64,
            scale_factor=2.0,
            mode="bilinear",
            align_corners=False,
        )
        fine_inputs = [fine_semantic]
        if self.use_rgb_edge:
            if image is None:
                edge_features = fine_semantic.new_zeros(
                    fine_semantic.shape[0],
                    32,
                    *fine_semantic.shape[-2:],
                )
            else:
                edge_features = self.rgb_edge_stem(image)
                edge_features = F.interpolate(
                    edge_features,
                    size=fine_semantic.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                uncertainty = F.interpolate(
                    self._refinement_gate(class64).detach(),
                    size=fine_semantic.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                edge_features = edge_features * uncertainty
            fine_inputs.append(edge_features)
        fine_inputs.append(class128_base)
        fine_pixels = self.fine_fuse(torch.cat(fine_inputs, dim=1))
        gate128 = F.interpolate(
            self._refinement_gate(class64).detach(),
            size=class128_base.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        class128 = (
            class128_base
            + self.alpha128 * gate128 * self.fine_delta(fine_pixels)
        )

        expected64 = (coarse_hw[0] * 2, coarse_hw[1] * 2)
        expected128 = (coarse_hw[0] * 4, coarse_hw[1] * 4)
        self._assert_spatial(class64, expected64, "class64")
        self._assert_spatial(class128, expected128, "class128")

        return {
            "mask_logits_32": mask32,
            "mask_logits_64": class64[:, : self.num_queries],
            "mask_logits_128": class128[:, : self.num_queries],
            "mask_logits": class128[:, : self.num_queries],
            "existence_logits": existence,
            "background_logits_32": background32,
            "background_logits_64": class64[:, self.num_queries :],
            "background_logits_128": class128[:, self.num_queries :],
            "background_logits": class128[:, self.num_queries :],
            "boundary_logits_32": self.boundary_head32(coarse_pixels),
            "boundary_logits_64": self.boundary_head64(middle_pixels),
            "boundary_logits_128": self.boundary_head128(fine_pixels),
            "boundary_logits": self.boundary_head128(fine_pixels),
            "refinement_alpha64": self.alpha64,
            "refinement_alpha128": self.alpha128,
            "refinement_gate64": gate64,
            "refinement_gate128": gate128,
        }
