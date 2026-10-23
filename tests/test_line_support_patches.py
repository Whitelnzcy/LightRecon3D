import unittest
import numpy as np
from models.line_support_patches import (
    prediction_transform, pointmap_to_original, original_to_pointmap,
    point_index_grid, propose_patches, line_coordinates, fuse_patch_logits,
    rgb_in_pointmap,
)
from audit_line_support_refinement import error_partition, inner_band


class PatchGeometryTests(unittest.TestCase):
    def transform(self):
        return dict(alignment_view_index=2,image_path='rgb.png', original_hw=[900,1200],
                    resized_hw=[384,512],crop_xyxy=[0,16,512,368],pointmap_hw=[352,512],
                    image_size=512,patch_size=16,square_ok=False,
                    layout_path='MUST_NOT_EXPORT',camera_to_world_m='MUST_NOT_EXPORT')

    def test_explicit_roundtrip_and_no_gt(self):
        t = prediction_transform(self.transform())
        self.assertNotIn('layout_path',t)
        self.assertNotIn('camera_to_world_m',t)
        xy = np.array([[0,0],[511,351],[16,32]])
        np.testing.assert_allclose(original_to_pointmap(pointmap_to_original(xy,t),t),xy,atol=1e-10)

    def test_invalid_transform_rejected(self):
        t = self.transform();t['pointmap_hw'] = [384,512]
        with self.assertRaises(ValueError): prediction_transform(t)

    def test_unique_point_registry(self):
        xy = np.array([[1,0],[3,2]])
        grid = point_index_grid(xy,np.array([9,2]),[3,4])
        self.assertEqual(grid[0,1],9)
        self.assertEqual(grid[2,3],2)
        with self.assertRaises(ValueError): point_index_grid(np.array([[1,0],[1,0]]),np.array([9,2]),[3,4])
        with self.assertRaises(ValueError): point_index_grid(np.array([[4,0]]),np.array([0]),[3,4])

    def test_no_gt_or_line_patch_selection(self):
        labels = np.zeros((96,128),int);labels[:,64:] = 1
        a = propose_patches(labels)
        self.assertEqual(a,propose_patches(labels))
        self.assertEqual(len(a),8)
        for p in a:
            x0,y0,x1,y1 = p['box_xyxy']
            self.assertEqual((x1-x0,y1-y0),(64,64))
            self.assertTrue(0<=x0<x1<=128 and 0<=y0<y1<=96)
        self.assertEqual(propose_patches(np.full((64,64),-1)),[])

    def test_line_sign_and_scale_invariance(self):
        xy = np.array([[3,4],[5,6]])
        np.testing.assert_allclose(line_coordinates(xy,[1,2,3]),line_coordinates(xy,[-2,-4,-6]))
        with self.assertRaises(ValueError): line_coordinates(xy,[0,0,1])

    def test_logit_fusion_order_and_candidate_permutation(self):
        original = np.array([10,20,10,-1,20])
        ids = np.array([10,20])
        contributions = [(np.array([0,1,3]),np.array([[0.,2.],[3.,0.],[3.,0.]])),
                         (np.array([0,2]),np.array([[3.,0.],[0.,4.]]))]
        out = fuse_patch_logits(original,ids,contributions)
        np.testing.assert_array_equal(out,[10,10,20,-1,20])
        np.testing.assert_array_equal(out,fuse_patch_logits(original,ids,contributions[::-1]))
        permuted = [(i,x[:,::-1]) for i,x in contributions]
        np.testing.assert_array_equal(out,fuse_patch_logits(original,ids[::-1],permuted))
        np.testing.assert_array_equal(original,[10,20,10,-1,20])
        np.testing.assert_array_equal(original,fuse_patch_logits(original,ids,[]))

    def test_tie_break_is_candidate_order_independent(self):
        for ids in [np.array([20,10]),np.array([10,20])]:
            result = fuse_patch_logits(np.array([20]),ids,[(np.array([0]),np.zeros((1,2)))])
            self.assertEqual(result[0],10)

    def test_rgb_identity_sampling(self):
        t = self.transform();t.update(original_hw=[384,512])
        rgb = np.arange(384*512*3,dtype=np.uint8).reshape(384,512,3)
        np.testing.assert_array_equal(rgb_in_pointmap(rgb,t),rgb[16:368])

    def test_disjoint_error_categories(self):
        raw=np.array([1,1,-1,1,1]);mapped=np.array([0,0,-2,0,0])
        result=error_partition(raw,mapped,np.array([0,1,1,2,1]),np.array([0,2,2,2,30]),{0,1})
        self.assertEqual(result['total'],4)
        self.assertEqual(result['fixed_identity_unrepresented'],1)
        self.assertEqual(result['represented_unassigned'],1)
        self.assertEqual(result['represented_boundary_5px'],1)
        self.assertEqual(result['represented_interior_gt_5px'],1)

    def test_sparse_hole_is_not_an_artificial_boundary(self):
        mask=np.ones((40,40),bool);valid=mask.copy();valid[20,20]=False;mask[20,20]=False
        self.assertFalse(inner_band(mask,valid)[20,21])


if __name__ == '__main__': unittest.main()
