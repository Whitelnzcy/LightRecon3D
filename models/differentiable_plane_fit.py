import torch
import torch.nn.functional as F


def differentiable_weighted_plane_fit(
    points,
    weights,
    valid_mask=None,
    eps=1e-6,
    cov_jitter=1e-5,
):
    """Fit K planes from soft point support with differentiable weighted PCA.

    Args:
        points: [N, 3] points in one shared coordinate frame.
        weights: [N, K] non-negative support weights.
        valid_mask: optional [N] finite/valid point mask.
    """
    if points.ndim != 2 or points.shape[-1] != 3:
        raise ValueError(f"points must be [N,3], got {tuple(points.shape)}")
    if weights.ndim != 2 or weights.shape[0] != points.shape[0]:
        raise ValueError(
            f"weights must be [N,K] with matching N, got {tuple(weights.shape)}"
        )

    finite = torch.isfinite(points).all(dim=-1) & (points.abs().amax(dim=-1) < 1e4)
    if valid_mask is not None:
        finite = finite & valid_mask.bool()
    safe_points = torch.where(finite[:, None], points, torch.zeros_like(points))
    safe_weights = weights.clamp_min(0.0) * finite[:, None].to(weights.dtype)

    mass = safe_weights.sum(dim=0)
    safe_mass = mass.clamp_min(eps)
    centers = (safe_weights.transpose(0, 1) @ safe_points) / safe_mass[:, None]
    centered = safe_points[:, None, :] - centers[None, :, :]
    covariance = torch.einsum(
        "nk,nki,nkj->kij",
        safe_weights,
        centered,
        centered,
    ) / safe_mass[:, None, None]
    eye = torch.eye(3, device=points.device, dtype=points.dtype).unsqueeze(0)
    covariance = covariance + float(cov_jitter) * eye
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    normals = F.normalize(eigenvectors[:, :, 0], dim=-1, eps=eps)
    offsets = -(normals * centers).sum(dim=-1)
    return normals, offsets, centers, eigenvalues, mass


def point_to_plane_distance(points, normals, offsets):
    """Return unsigned distances for points [N,3] and planes [K,3]/[K]."""
    return torch.abs(points @ normals.transpose(0, 1) + offsets[None])
