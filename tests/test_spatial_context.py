import unittest
import numpy as np
import torch
from spatial_context_data import extract_context,center_crop,context_batch
from spatial_support_data import patch_batch,tile_boxes
from spatial_line_features import line_batch


class ContextTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(4)
        self.v=dict(common=rng.normal(size=(4,96,160)).astype(np.float32),
            candidates=rng.normal(size=(3,2,96,160)).astype(np.float32),
            lines=rng.normal(size=(3,6,96,160)).astype(np.float32),
            target=rng.integers(0,3,(96,160)),original_columns=rng.integers(0,3,(96,160)))
        self.boxes=[[0,0,64,64],[96,32,160,96]]

    def test_zero_margin_matches_existing_path(self):
        old=patch_batch(self.v,self.boxes,'cpu');old=old[:2]+[line_batch(self.v,self.boxes,'cpu')]+old[2:]
        for a,b in zip(old,context_batch(self.v,self.boxes,'cpu',0)):torch.testing.assert_close(a,b)

    def test_center_and_targets_identical_including_image_edges(self):
        small=context_batch(self.v,self.boxes,'cpu',0);large=context_batch(self.v,self.boxes,'cpu',32)
        for a,b in zip(small[:3],large[:3]):torch.testing.assert_close(a,center_crop(b,32))
        for a,b in zip(small[3:],large[3:]):torch.testing.assert_close(a,b)
        self.assertFalse(large[0][0,:,:32].any());self.assertFalse(large[0][1,:,-32:].any())
        self.assertFalse(large[2][1,:,:,:,-32:].any())
        torch.testing.assert_close(large[0][0,:,32:96,96:],torch.from_numpy(self.v['common'][:,:64,64:96]))

    def test_inference_does_not_read_supervision(self):
        v={k:self.v[k] for k in ['common','candidates','lines']}
        self.assertEqual(len(context_batch(v,self.boxes,'cpu',32,False)),3)

    def test_output_crop_gives_no_direct_halo_supervision(self):
        x=torch.randn(1,3,128,128,requires_grad=True);center_crop(x,32).sum().backward()
        self.assertFalse(x.grad[...,:32,:].any());self.assertTrue((x.grad[...,32:96,32:96]==1).all())

    def test_all_center_pixels_keep_exact_registry(self):
        grid=np.arange(96*160).reshape(96,160);boxes=tile_boxes(grid.shape)
        large=extract_context(grid,boxes,32);small=extract_context(grid,boxes,0)
        np.testing.assert_array_equal(center_crop(large,32),small)
        with self.assertRaises(ValueError):extract_context(grid,boxes,-1)
        with self.assertRaises(ValueError):extract_context(grid,[[-1,0,63,64]],32)


if __name__=='__main__':unittest.main()
