import torch
import torch.nn as nn
import torch.nn.functional as F


class LightReconModel(nn.Module):
    """
    LightRecon3D model.

    Current version:
    - DUSt3R backbone
    - line prediction head
    - plane embedding head

    pred_line:
        [B, 1, H, W], line logits

    pred_plane:
        [B, plane_embed_dim, H, W], plane embedding map

    Important:
        pred_plane is no longer:
        - 20-class plane instance logits
        - 1-channel plane boundary logits

        It is now a per-pixel plane embedding.
    """

    def __init__(
        self,
        dust3r_backbone,
        hidden_dim=768,
        patch_size=16,
        line_hidden_dim=256,
        plane_hidden_dim=256,
        plane_embed_dim=16,
        num_planes=None,  # kept for compatibility; no longer used
    ):
        super().__init__()

        self.backbone = dust3r_backbone
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size
        self.plane_embed_dim = plane_embed_dim

        # -------------------------
        # Line head
        # output: [B, 1, H, W]
        # -------------------------
        self.line_head = nn.Sequential(
            nn.Conv2d(hidden_dim, line_hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=line_hidden_dim),
            nn.GELU(),
            nn.Conv2d(line_hidden_dim, line_hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=line_hidden_dim),
            nn.GELU(),
            nn.Conv2d(line_hidden_dim, 1, kernel_size=1),
        )

        # -------------------------
        # Plane embedding head
        # output: [B, plane_embed_dim, H, W]
        # -------------------------
        self.plane_head = nn.Sequential(
            nn.Conv2d(hidden_dim, plane_hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=plane_hidden_dim),
            nn.GELU(),
            nn.Conv2d(plane_hidden_dim, plane_hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=plane_hidden_dim),
            nn.GELU(),
            nn.Conv2d(plane_hidden_dim, plane_embed_dim, kernel_size=1),
        )

    def _get_dec_features(self, res):
        """
        Extract decoder features from DUSt3R output.

        Expected:
            res["dec_features"]: [B, S, D]

        If it is a list / tuple, use the last feature.
        """
        if "dec_features" not in res:
            raise KeyError(
                "DUSt3R output does not contain 'dec_features'. "
                "Please check whether dust3r/model.py returns decoder features."
            )

        features = res["dec_features"]

        if isinstance(features, (list, tuple)):
            features = features[-1]

        if features.ndim != 3:
            raise ValueError(
                f"Expected dec_features shape [B, S, D], got {features.shape}"
            )

        return features

    def _tokens_to_feature_map(self, features, img):
        """
        Convert transformer token sequence to CNN feature map.

        features:
            [B, S, D]

        img:
            [B, 3, H_img, W_img]

        return:
            [B, D, H_feat, W_feat]
        """
        B, S, D = features.shape
        H_img, W_img = img.shape[-2:]

        H_feat = H_img // self.patch_size
        W_feat = W_img // self.patch_size

        expected_tokens = H_feat * W_feat

        if S != expected_tokens:
            raise AssertionError(
                f"Token number mismatch: S={S}, "
                f"H_feat*W_feat={H_feat}*{W_feat}={expected_tokens}. "
                f"Please check image_size={H_img}x{W_img} and patch_size={self.patch_size}."
            )

        feat_cnn = (
            features.transpose(1, 2)
            .contiguous()
            .view(B, D, H_feat, W_feat)
        )

        return feat_cnn

    def _predict_heads(self, res, view):
        """
        Attach line logits and plane embedding to DUSt3R output.
        """
        img = view["img"]
        H_img, W_img = img.shape[-2:]

        features = self._get_dec_features(res)
        feat_cnn = self._tokens_to_feature_map(features, img)

        pred_line_lowres = self.line_head(feat_cnn)
        pred_plane_lowres = self.plane_head(feat_cnn)

        pred_line = F.interpolate(
            pred_line_lowres,
            size=(H_img, W_img),
            mode="bilinear",
            align_corners=False,
        )

        pred_plane = F.interpolate(
            pred_plane_lowres,
            size=(H_img, W_img),
            mode="bilinear",
            align_corners=False,
        )

        res = dict(res)

        res["pred_line"] = pred_line

        # pred_plane now means plane embedding.
        res["pred_plane"] = pred_plane
        res["pred_plane_embedding"] = pred_plane

        return res

    def forward(self, view1, view2):
        """
        DUSt3R expects two views.

        Current baseline can still use pseudo-pair:
        view1 and view2 may be the same image.

        return:
            res1, res2
        """
        res1, res2 = self.backbone(view1, view2)

        res1 = self._predict_heads(res1, view1)
        res2 = self._predict_heads(res2, view2)

        return res1, res2