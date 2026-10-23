"""Prediction-only view tensors and separately attached layout supervision."""
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image
from audit_line_support_refinement import only_npz, sha
from build_structured3d_point_aligned_gt import (
    dust3r_resize_crop_transform, render_layout_plane_ids, preprocess_plane_mask,
)
from models.line_support_patches import (
    prediction_transform, point_index_grid, rgb_in_pointmap, propose_patches, valid_transitions,
)
from models.ownership_supervision import supervision_only
from recover_cache_projection import recover_checked


def tile_boxes(hw, size=64, stride=48):
    h, w = hw
    if min(h, w) < size or not 0 < stride <= size:
        raise ValueError('Invalid tiling size/stride')
    def starts(length):
        return sorted(set(range(0, length-size+1, stride)) | {length-size})
    return [[x,y,x+size,y+size] for y in starts(h) for x in starts(w)]


def load_prediction_scene(root):
    """Layout files/labels and GT cameras are not read here.

    Historical GT container is accessed for preprocessing metadata only, then
    immediately allowlisted; dimensions are independently regenerated from RGB.
    """
    root = Path(root)
    source, metadata = only_npz(root/'guided_ransac'), only_npz(root/'gt')
    with np.load(source) as z:
        assert z['pixel_coordinate_order'].item() == 'xy'
        assert z['pixel_coordinate_space'].item() == 'dust3r_aligned_pointmap'
        old = {k:z[k] for k in ['points','point_plane_ids','plane_ids','plane_normals',
                                'plane_offsets','source_views','pixel_xy']}
    with np.load(metadata) as z:
        transforms = [prediction_transform(t) for t in json.loads(z['view_transforms_json'].item())]
    ids = old['plane_ids']
    if len(ids) == 0 or len(np.unique(ids)) != len(ids) or not np.isin(old['point_plane_ids'][old['point_plane_ids']>=0],ids).all():
        raise ValueError('Invalid candidate pool')
    scale = max(float(np.linalg.norm(np.ptp(old['points'],axis=0))),1e-6)
    views, seen = [], np.zeros(len(old['points']),bool)
    for t in transforms:
        rgb = np.asarray(Image.open(t['image_path']).convert('RGB'))
        assert list(rgb.shape[:2]) == t['original_hw']
        expected = dust3r_resize_crop_transform(rgb.shape[:2],t['image_size'],t['patch_size'],t['square_ok'])
        assert all(t[k] == v for k,v in expected.items())
        ix = np.flatnonzero(old['source_views'] == t['alignment_view_index'])
        assert len(ix) and not seen[ix].any()
        seen[ix] = True
        xy = old['pixel_xy'][ix]
        registry = point_index_grid(xy,ix,t['pointmap_hw'])
        observed = registry >= 0
        h,w = registry.shape
        labels = np.full((h,w),-1,int)
        labels[observed] = old['point_plane_ids'][registry[observed]]
        rgb = rgb_in_pointmap(rgb,t)
        common = np.concatenate([rgb.transpose(2,0,1).astype(np.float32)/127.5-1,
                                 observed[None].astype(np.float32)])
        candidate = np.zeros((len(ids),2,h,w),np.float32)
        # Geometric residual is independent of GT; unknown samples remain zero.
        residual = np.log1p(100*np.abs(old['points'][ix]@old['plane_normals'].T+old['plane_offsets'])/scale)
        for k,pid in enumerate(ids):
            candidate[k,0] = labels == pid
            candidate[k,1,xy[:,1],xy[:,0]] = np.minimum(residual[:,k],8)
        columns = np.full((h,w),-1,np.int64)
        for k,pid in enumerate(ids):
            columns[labels == pid] = k
        patches = propose_patches(labels,boundary_count=64,uniform_count=32,
                                  seed=42+int(t['alignment_view_index']))
        projection, quality = recover_checked(old['points'][ix],xy)
        views.append(dict(transform=t, registry=registry, common=common, candidates=candidate,
                          original_columns=columns, patches=patches, projection=projection,
                          projection_quality=quality))
    assert seen.all()
    return dict(name=root.name, root=str(root), source=str(source), source_sha256=sha(source),
                metadata=str(metadata), old=old, views=views)


def attach_supervision(scene):
    """Must be called after prediction tensors and patch proposals are finalized."""
    old = scene['old']
    dense = np.full(len(old['points']),-1,np.int32)
    distance = np.full(len(dense),np.inf,np.float32)
    with np.load(scene['metadata']) as z:
        assert np.array_equal(z['points'],old['points'])
        assert np.array_equal(z['pixel_xy'],old['pixel_xy'])
        assert np.array_equal(z['view_indices'],old['source_views'])
        transforms = json.loads(z['view_transforms_json'].item())
    lookup = {int(t['alignment_view_index']):t for t in transforms}
    for view in scene['views']:
        t = lookup[int(view['transform']['alignment_view_index'])]
        layout = json.loads(Path(t['layout_path']).read_text())
        gt = preprocess_plane_mask(render_layout_plane_ids(layout,t['original_hw']),t,0)
        b = valid_transitions(gt)
        dist = cv2.distanceTransform((~b).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        registry = view['registry'];valid = registry>=0
        dense[registry[valid]] = gt[valid]
        distance[registry[valid]] = dist[valid]
    sup = supervision_only(old['point_plane_ids'],dense,old['plane_ids'])
    columns = np.full(len(dense),-100,np.int64)
    for k,pid in enumerate(old['plane_ids']):
        columns[sup['target_plane_id'] == pid] = k
    # Inference never edits originally unassigned points: do not train to do so.
    columns[old['point_plane_ids']<0] = -100
    coverage = np.zeros(len(dense),bool)
    for view in scene['views']:
        registry = view['registry'];valid = registry>=0
        target = np.full(registry.shape,-100,np.int64)
        target[valid] = columns[registry[valid]]
        view['target'] = target
        for patch in view['patches']:
            x0,y0,x1,y1 = patch['box_xyxy'];indices=registry[y0:y1,x0:x1]
            coverage[indices[indices>=0]] = True
    scene.update(truth=dense, distance=distance, mapping=sup['mapping'],
                 target_columns=columns, missing_candidate=int(sup['missing_candidate'].sum()))
    valid = dense>=0
    before = np.full(len(dense),-2,int)
    for p,g in sup['mapping'].items(): before[old['point_plane_ids']==p]=g
    wrong = valid&(before!=dense)
    scene['coverage'] = dict(points=int(len(dense)), valid=int(valid.sum()), supervised=int((columns>=0).sum()),
        proposed_covered=int(coverage.sum()), wrong=int(wrong.sum()), wrong_covered=int((wrong&coverage).sum()),
        boundary_wrong=int((wrong&(distance<=5)).sum()), boundary_wrong_covered=int((wrong&(distance<=5)&coverage).sum()))


def patch_batch(view, boxes, device, with_targets=True):
    import torch
    common = np.stack([view['common'][:,y0:y1,x0:x1] for x0,y0,x1,y1 in boxes])
    candidates = np.stack([view['candidates'][:,:,y0:y1,x0:x1] for x0,y0,x1,y1 in boxes])
    result = [torch.from_numpy(common).to(device),torch.from_numpy(candidates).to(device)]
    if with_targets:
        for field in ['target','original_columns']:
            value = np.stack([view[field][y0:y1,x0:x1] for x0,y0,x1,y1 in boxes])
            result.append(torch.from_numpy(value).to(device))
    return result
