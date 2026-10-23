"""GT-free patch geometry for fixed-point, bounded-plane ownership refinement.

Pixel coordinates are integer-center (x, y) in the cropped DUSt3R pointmap.
RGB sampling uses an explicit half-pixel inverse resize, never shape guessing.
No layout labels, camera poses or ground-truth-selected ROIs enter this module.
"""
import cv2
import numpy as np


TRANSFORM_KEYS = (
    'alignment_view_index', 'image_path', 'original_hw', 'resized_hw',
    'crop_xyxy', 'pointmap_hw', 'image_size', 'patch_size', 'square_ok',
)


def prediction_transform(transform):
    result = {k: transform[k] for k in TRANSFORM_KEYS}
    oh, ow = result['original_hw']
    rh, rw = result['resized_hw']
    x0, y0, x1, y1 = result['crop_xyxy']
    h, w = result['pointmap_hw']
    if min(oh, ow, rh, rw, h, w) <= 0:
        raise ValueError('Non-positive image dimensions')
    if not (0 <= x0 < x1 <= rw and 0 <= y0 < y1 <= rh):
        raise ValueError('Crop outside resized image')
    if [y1-y0, x1-x0] != [h, w]:
        raise ValueError('Crop and pointmap dimensions disagree')
    result['pixel_coordinate_order'] = 'xy'
    result['pixel_coordinate_space'] = 'dust3r_aligned_pointmap'
    result['pixel_centers'] = 'integer; inverse resize uses half-pixel convention'
    return result


def pointmap_to_original(xy, transform):
    oh, ow = transform['original_hw']
    rh, rw = transform['resized_hw']
    x0, y0, _, _ = transform['crop_xyxy']
    return (np.asarray(xy, float) + [x0, y0] + .5) * [ow/rw, oh/rh] - .5


def original_to_pointmap(xy, transform):
    oh, ow = transform['original_hw']
    rh, rw = transform['resized_hw']
    x0, y0, _, _ = transform['crop_xyxy']
    return (np.asarray(xy, float) + .5) * [rw/ow, rh/oh] - .5 - [x0, y0]


def rgb_in_pointmap(rgb, transform):
    if list(rgb.shape[:2]) != transform['original_hw']:
        raise ValueError('RGB dimensions disagree with explicit transform')
    h, w = transform['pointmap_hw']
    yy, xx = np.indices((h, w))
    uv = pointmap_to_original(np.stack([xx, yy], -1), transform).astype(np.float32)
    return cv2.remap(rgb, uv[..., 0], uv[..., 1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def point_index_grid(xy, source_indices, hw):
    xy = np.asarray(xy)
    source_indices = np.asarray(source_indices)
    h, w = hw
    if xy.shape != (len(source_indices), 2) or not np.issubdtype(xy.dtype, np.integer):
        raise ValueError('Expected integer N x 2 xy registry')
    if np.any(xy < 0) or np.any(xy[:, 0] >= w) or np.any(xy[:, 1] >= h):
        raise ValueError('Point registry is outside its declared pointmap')
    flat = xy[:, 1]*w + xy[:, 0]
    if len(np.unique(flat)) != len(flat) or len(np.unique(source_indices)) != len(flat):
        raise ValueError('Duplicate pixel or source index')
    grid = np.full((h, w), -1, np.int64)
    grid[xy[:, 1], xy[:, 0]] = source_indices
    return grid


def valid_transitions(labels):
    valid = labels >= 0
    boundary = np.zeros(labels.shape, bool)
    horizontal = (labels[:, 1:] != labels[:, :-1]) & valid[:, 1:] & valid[:, :-1]
    vertical = (labels[1:] != labels[:-1]) & valid[1:] & valid[:-1]
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal
    boundary[1:] |= vertical
    boundary[:-1] |= vertical
    return boundary


def propose_patches(labels, size=64, boundary_count=6, uniform_count=2, seed=42):
    """Same axis-aligned proposals for all arms; neither GT nor lines accepted."""
    h, w = labels.shape
    if size <= 0 or min(h, w) < size or min(boundary_count, uniform_count) < 0:
        raise ValueError('Invalid patch size/count')
    rng = np.random.default_rng(seed)
    boundary = np.argwhere(valid_transitions(labels))
    valid = np.argwhere(labels >= 0)
    result, seen = [], set()
    for role, pool, count in [('predicted_boundary', boundary, boundary_count),
                              ('ordinary_context', valid, uniform_count)]:
        made = 0
        for j in rng.permutation(len(pool)):
            y, x = pool[j]
            x0 = int(np.clip(x-size//2, 0, w-size))
            y0 = int(np.clip(y-size//2, 0, h-size))
            box = (x0, y0, x0+size, y0+size)
            if box in seen:
                continue
            if made >= count:
                break
            seen.add(box)
            result.append(dict(box_xyxy=list(box), source=role))
            made += 1
    return result


def canonical_line(line):
    line = np.asarray(line, float)
    if line.shape != (3,) or not np.isfinite(line).all() or np.linalg.norm(line[:2]) < 1e-10:
        raise ValueError('Invalid projected line')
    line = line / np.linalg.norm(line[:2])
    pivot = int(np.argmax(np.abs(line[:2])))
    return line if line[pivot] >= 0 else -line


def line_coordinates(xy, line):
    """Canonical along/across coordinates; invariant to homogeneous line sign."""
    line = canonical_line(line)
    xy = np.asarray(xy, float)
    return np.stack([xy @ np.array([-line[1], line[0]]),
                     xy @ line[:2] + line[2]], -1)


def fuse_patch_logits(original_ids, candidate_ids, contributions):
    """Average overlapping candidate logits once, preserve uncovered/unassigned.

    Each contribution is (global source indices, N x K logits). Candidate axis
    follows candidate_ids explicitly, not implicit numerical label order.
    """
    ids = np.asarray(candidate_ids)
    if ids.ndim != 1 or len(np.unique(ids)) != len(ids) or np.any(ids < 0):
        raise ValueError('Candidate IDs must be distinct nonnegative IDs')
    original = np.asarray(original_ids)
    if not np.isin(original[original >= 0], ids).all():
        raise ValueError('Original labels missing from candidate pool')
    sums = np.zeros((len(original), len(ids)), float)
    counts = np.zeros(len(original), int)
    for index, logits in contributions:
        index, logits = np.asarray(index), np.asarray(logits)
        if not np.issubdtype(index.dtype, np.integer) or index.ndim != 1:
            raise ValueError('Invalid source indices')
        if np.any(index < 0) or np.any(index >= len(original)):
            raise ValueError('Source index outside original cloud')
        if logits.shape != (len(index), len(ids)) or not np.isfinite(logits).all():
            raise ValueError('Invalid candidate logits')
        np.add.at(sums, index, logits)
        np.add.at(counts, index, 1)
    result = original.copy()
    active = (counts > 0) & (original >= 0)
    # Stable tie-breaking by candidate ID, not candidate column order.
    order = np.argsort(ids)
    if len(ids):
        result[active] = ids[order[np.argmax(sums[active][:, order]/counts[active, None], axis=1)]]
    return result
