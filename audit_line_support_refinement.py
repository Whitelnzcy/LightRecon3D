"""P0/P1 read-only cache audit. Never trains, edits old results or uses GT ROIs.

Recomputes saved DEV predictions (NOT fresh inference), inventories provenance,
and writes new prediction-only patch indices plus separate evaluation reports.
"""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from threadpoolctl import threadpool_limits

from build_structured3d_point_aligned_gt import (
    dust3r_resize_crop_transform, render_layout_plane_ids, preprocess_plane_mask,
)
from models.line_support_patches import (
    prediction_transform, pointmap_to_original, original_to_pointmap,
    rgb_in_pointmap, point_index_grid, propose_patches, valid_transitions,
)
from recover_cache_projection import recover_checked


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def only_npz(directory):
    paths = sorted(Path(directory).glob('*.npz'))
    if len(paths) != 1:
        raise ValueError(f'Expected exactly one NPZ: {directory}, got {len(paths)}')
    return paths[0]


def convert(labels, mapping):
    result = np.full(labels.shape, -2, np.int32)
    for source, target in mapping.items():
        result[labels == int(source)] = int(target)
    return result


def metrics(before, after, truth, distance):
    valid = truth >= 0
    b, a, boundary = before == truth, after == truth, distance <= 5
    return dict(valid=int(valid.sum()), before_errors=int((valid & ~b).sum()),
                after_errors=int((valid & ~a).sum()), corrected=int((valid & ~b & a).sum()),
                damaged=int((valid & b & ~a).sum()), changed=int((before != after).sum()),
                boundary_before=int((valid & boundary & ~b).sum()),
                boundary_after=int((valid & boundary & ~a).sum()))


def error_partition(raw_labels, mapped, truth, distance, represented):
    wrong = (truth >= 0) & (mapped != truth)
    missing = wrong & ~np.isin(truth, list(represented))
    unassigned = wrong & ~missing & (raw_labels < 0)
    actionable = wrong & ~missing & ~unassigned
    result = dict(total=int(wrong.sum()), fixed_identity_unrepresented=int(missing.sum()),
                  represented_unassigned=int(unassigned.sum()),
                  represented_boundary_5px=int((actionable & (distance <= 5)).sum()),
                  represented_interior_gt_5px=int((actionable & (distance > 5)).sum()))
    assert result['total'] == sum(v for k, v in result.items() if k != 'total')
    result['represented_distance_histogram'] = {
        '0_to_5': int((actionable & (distance <= 5)).sum()),
        '5_to_20': int((actionable & (distance > 5) & (distance <= 20)).sum()),
        '20_to_32': int((actionable & (distance > 20) & (distance <= 32)).sum()),
        'over_32': int((actionable & (distance > 32)).sum()),
    }
    return result


def inner_band(mask, valid, width=5):
    # Unknown pixels must not generate artificial sparse-cloud boundaries.
    filled = (mask | ~valid).astype(np.uint8)
    distance = cv2.distanceTransform(np.pad(filled, 1), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    return mask & valid & (distance <= width)


def boundary_iou(before, after, truth, valid):
    """Fixed-identity observed-domain variant, not official rematched Boundary IoU."""
    rows = []
    for label in sorted(set(truth[valid].tolist()) | set(before[valid].tolist()) | set(after[valid].tolist())):
        gt = inner_band(truth == label, valid)
        row = dict(label=int(label))
        for name, pred in [('before', before), ('after', after)]:
            band = inner_band(pred == label, valid)
            row[name] = [int((gt & band).sum()), int((gt | band).sum())]
        rows.append(row)
    return rows


def palette(labels):
    out = np.full((*labels.shape, 3), 240, np.uint8)
    for label in np.unique(labels):
        if label >= 0:
            out[labels == label] = np.random.default_rng(int(label)+130).integers(40, 225, 3, dtype=np.uint8)
        elif label == -2:
            out[labels == label] = [80, 80, 80]
    return out


def save_figure(path, rgb, index, original, refined, truth, patches, lines, quality):
    h, w = index.shape
    observed = index >= 0
    before, after, gt = [np.full((h, w), -1, int) for _ in range(3)]
    for grid, values in [(before, original), (after, refined), (gt, truth)]:
        grid[observed] = values[index[observed]]
    error = np.full((h, w, 3), 238, np.uint8)
    valid = observed & (gt >= 0)
    error[valid & (before != gt) & (after == gt)] = [0, 150, 95]
    error[valid & (before == gt) & (after != gt)] = [220, 45, 55]
    error[valid & (before != gt) & (after != gt)] = [235, 165, 35]
    rgb_lines = rgb.copy()
    yy, xx = np.indices((h, w))
    for row in lines:
        line = np.array(row['after'], float)
        d = np.abs(xx*line[0]+yy*line[1]+line[2])/max(np.linalg.norm(line[:2]), 1e-9)
        rgb_lines[d <= .6] = [35, 90, 240]
    panels = [('RGB + predicted pair lines', rgb_lines), ('Original plane IDs', palette(before)),
              ('Saved finite model', palette(after)), ('Layout GT (evaluation only)', palette(gt)),
              ('Green: fixed / red: damaged', error)]
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 18)
    sheet = Image.new('RGB', (5*w, 2*h+120), 'white')
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 6), f'{path.stem} | Same registered points; no new inference or XYZ updates', fill='black', font=font)
    box = patches[0]['box_xyxy'] if patches else [0, 0, min(h,w), min(h,w)]
    for col, (title, arr) in enumerate(panels):
        img = Image.fromarray(arr)
        sheet.paste(img, (col*w, 58))
        draw.text((col*w+8, 33), title, fill='black', font=font)
        draw.rectangle([col*w+box[0], 58+box[1], col*w+box[2], 58+box[3]], outline='black', width=2)
        crop = img.crop(box).resize((h, h), Image.Resampling.NEAREST)
        sheet.paste(crop, (col*w+(w-h)//2, h+83))
    draw.text((12, 2*h+88), f'Zoom selected from predicted boundary only. Orange: still wrong. DLT reprojection P95: {quality["p95"]:.3f} px.', fill='black', font=font)
    sheet.save(path)


def inventory():
    rgb_protocol = Path('local_outputs/rgb_structure_boundary_20260908_full3997_e50_v1/protocol.json')
    rgb = read_json(rgb_protocol)
    groups = {k: set(rgb[k]) for k in ['train_scenes', 'dev_scenes', 'test_scenes']}
    rows = []
    for path in sorted(Path('local_outputs').glob('*/items/*/guided_ransac/*.npz')):
        match = re.search(r'scene_\d{5}', path.parent.parent.name)
        if not match:
            continue
        scene = match.group()
        rows.append(dict(scene=scene, item=path.parent.parent.name, source=str(path),
                         gt_metadata_present=len(list((path.parent.parent/'gt').glob('*.npz'))) == 1,
                         boundary_model_exposure=[k for k, values in groups.items() if scene in values],
                         stage1_training_exposure='not_yet_audited', independent_test_eligible=False))
    available = {r['scene'] for r in rows}
    return dict(schema_version=1, scope='Available guided_ransac item caches only, not all raw RGB scenes',
                rgb_protocol_sha256=sha(rgb_protocol), boundary_counts=rgb['counts'],
                unique_cached_scenes=len(available), cache_entries=len(rows),
                boundary_overlap_counts={k: len(available & values) for k, values in groups.items()},
                boundary_unexposed_scenes=sorted(available-set.union(*groups.values())),
                items=rows, caveat='Historical boundary test exposure is not fresh blind evaluation; upstream scene/building split still requires audit.')


def main():
    threadpool_limits(1)
    cv2.setNumThreads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out/'figures').mkdir()
    (out/'prediction_inputs').mkdir()
    manifest = inventory()
    write_json(out/'data_manifest.json', manifest)
    frozen = Path('local_outputs/intersection_frozen_dev2_v1')
    targets = Path('local_outputs/dense_boundary_competition_20260918_v1')
    saved = read_json(frozen/'report.json')
    target_protocol = read_json(targets/'report.json')
    protocol = dict(schema_version=1, git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    phase='P0 and P1 alignment pilot, no training; saved-prediction recount, not fresh inference',
                    geometry='Fixed global DUSt3R XYZ, row order and candidate parameters',
                    gt='layout.json visible_mask; this audited subset uses perspective/empty RGB, not furniture-visible surfaces',
                    boundary='Euclidean distance <=5 in original cropped pointmap resolution; both valid GT sides',
                    secondary_metric='Fixed-identity observed-domain inner-band IoU, 5px Euclidean; sparse holes do not create boundaries; image perimeter included. Not official benchmark Boundary IoU.',
                    patch='64x64 axis-aligned; 6 predicted-edge and 2 ordinary-context samples per view; seed42+view; all scene candidates',
                    rgb_sampling='Original RGB bilinear half-pixel sampling; not byte-identical to DUSt3R Lanczos preprocessing',
                    prediction_inputs='Allowlisted resize/crop metadata exported separately; no GT labels, layout/camera paths or GT camera matrices',
                    pending=['Stage1/full upstream split audit', 'untouched scene acquisition or split redesign',
                             'nearest-work comparison', 'spatial refiner and M1/M2/M3 training'])
    write_json(out/'protocol.json', protocol)
    report = {}
    for item in saved:
        root = Path('local_outputs/b12r2/items')/item
        source, gt_path = only_npz(root/'guided_ransac'), only_npz(root/'gt')
        provenance = read_json(frozen/f'{item}_provenance.json')
        assert sha(source) == provenance['source_sha256'], 'Source changed since saved inference'
        with np.load(source) as z:
            old = {k:z[k] for k in ['points','pixel_xy','source_views','point_plane_ids','plane_ids','plane_normals','plane_offsets']}
            assert z['pixel_coordinate_order'].item() == 'xy'
            assert z['pixel_coordinate_space'].item() == 'dust3r_aligned_pointmap'
        with np.load(gt_path) as z:
            ts = json.loads(z['view_transforms_json'].item())
            assert np.array_equal(old['points'], z['points'])
            assert np.array_equal(old['pixel_xy'], z['pixel_xy'])
            assert np.array_equal(old['source_views'], z['view_indices'])
        prediction_path = frozen/f'{item}_predictions.npz'
        with np.load(prediction_path) as z:
            assert np.array_equal(z['source_indices'], np.arange(len(old['points'])))
            assert np.array_equal(z['original_ids'], old['point_plane_ids'])
            preds = {k:z[k] for k in ['finite_control','finite_support']}
        with np.load(targets/f'{item}_eval_only.npz') as z:
            assert np.array_equal(z['pixel_xy'], old['pixel_xy'])
            assert np.array_equal(z['view_indices'], old['source_views'])
            truth, distance = z['raw_dense_gt'], z['boundary_distance']
        mapping = target_protocol[item]['fixed_identity_map']
        before = convert(old['point_plane_ids'], mapping)
        mapped = {k:convert(v,mapping) for k,v in preds.items()}
        recalculated = {k:metrics(before,v,truth,distance) for k,v in mapped.items()}
        for arm in recalculated:
            # Historical changed counts refer to raw IDs, not GT-mapped IDs.
            recalculated[arm]['changed'] = int((preds[arm] != old['point_plane_ids']).sum())
            assert recalculated[arm] == saved[item]['metrics'][arm], (item, arm, recalculated[arm])
        partitions = {k:error_partition(raw, labels, truth, distance, mapping.values())
                      for k,raw,labels in [('original',old['point_plane_ids'],before),
                                           ('finite_support',preds['finite_support'],mapped['finite_support'])]}
        views, inputs, boundaries = [], [], []
        represented = set(mapping.values())
        overlap = []
        for gid in np.unique(truth[truth >= 0]):
            support = truth == gid
            counts = [(int(p), int((support & (old['point_plane_ids'] == p)).sum())) for p in old['plane_ids']]
            overlap.append(dict(gt_id=int(gid), points=int(support.sum()), fixed_identity_represented=int(gid) in represented,
                                predicted_support_counts=counts))
        seen = np.zeros(len(truth), bool)
        for t in ts:
            vi = int(t['alignment_view_index'])
            transform = prediction_transform(t)
            reconstructed = dust3r_resize_crop_transform(t['original_hw'], t['image_size'], t['patch_size'], t['square_ok'])
            for key, value in reconstructed.items():
                assert t[key] == value, (item,vi,key)
            ix = np.flatnonzero(old['source_views'] == vi)
            assert len(ix) and not seen[ix].any()
            seen[ix] = True
            xy = old['pixel_xy'][ix]
            registry = point_index_grid(xy, ix, t['pointmap_hw'])
            roundtrip = float(np.abs(original_to_pointmap(pointmap_to_original(xy,t),t)-xy).max())
            assert roundtrip < 1e-8
            # Camera is recovered only from reconstructed points and registered pixels.
            projection, quality = recover_checked(old['points'][ix], xy)
            rgb = np.asarray(Image.open(t['image_path']).convert('RGB'))
            rgb = rgb_in_pointmap(rgb, transform)
            grid = np.full(registry.shape, -1, int)
            grid[registry >= 0] = old['point_plane_ids'][registry[registry >= 0]]
            patches = propose_patches(grid, seed=42+vi)
            line_rows = next(r['lines'] for r in provenance['views'] if r['view'] == vi)
            inputs.append(dict(transform=transform, source=str(source), candidate_ids=old['plane_ids'].tolist(),
                               projection_from_predicted_xyz=projection.tolist(), lines=line_rows, patches=patches))
            # Evaluation labels are opened only after GT-free proposals are fixed.
            dense = preprocess_plane_mask(render_layout_plane_ids(read_json(t['layout_path']),t['original_hw']),t,0)
            assert np.array_equal(dense[xy[:,1],xy[:,0]],truth[ix])
            b = valid_transitions(dense)
            recomputed_distance = cv2.distanceTransform((~b).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
            assert np.allclose(recomputed_distance[xy[:,1],xy[:,0]],distance[ix],atol=1e-5)
            grids = []
            for labels in [before,mapped['finite_support'],truth]:
                g = np.full(registry.shape,-1,int)
                g[registry>=0] = labels[registry[registry>=0]]
                grids.append(g)
            boundaries.extend(boundary_iou(*grids,(registry>=0)&(grids[2]>=0)))
            save_figure(out/'figures'/f'{item}_view{vi}.png', rgb, registry, before,
                        mapped['finite_support'], truth, patches, line_rows, quality)
            views.append(dict(view=vi, points=len(ix), patches=len(patches), roundtrip_max_px=roundtrip,
                              projection=quality, image_sha256=sha(t['image_path']),
                              source_rgb=str(t['image_path']), gt_dense_and_boundary_reproduced=True))
        assert seen.all()
        totals = {arm: {} for arm in ['before','after']}
        for row in boundaries:
            for arm in totals:
                count = totals[arm].setdefault(row['label'],[0,0])
                count[0] += row[arm][0]
                count[1] += row[arm][1]
        scores = {arm:float(np.mean([i/u for i,u in data.values() if u])) for arm,data in totals.items()}
        write_json(out/'prediction_inputs'/f'{item}.json',dict(schema_version=1, views=inputs))
        report[item] = dict(saved_metrics_exact_match=True, metrics=recalculated, error_partition=partitions,
                            fixed_identity_overlap=overlap, observed_boundary_iou_macro=scores, views=views,
                            input_sha256=sha(source), prediction_sha256=sha(prediction_path),
                            gt_metadata_sha256=sha(gt_path), caveat='Unrepresented identity is not proof of physically missing plane.')
        write_json(out/'baseline_diagnosis.json',report)
        print(item, partitions, flush=True)
    write_json(out/'completion.json',dict(status='audit_completed', scenes=len(report), views=sum(len(r['views']) for r in report.values()),
               figures=len(list((out/'figures').glob('*.png'))), training_started=False))
    print('COMPLETE',str(out),flush=True)


if __name__ == '__main__':
    main()
