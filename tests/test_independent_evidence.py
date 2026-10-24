import unittest
import numpy as np
import torch
from independent_boundary_evidence import align_probability
from models.independent_evidence_refiner import IndependentEvidenceRefiner


class EvidenceTest(unittest.TestCase):
    def test_alignment_identity_and_crop(self):
        p=np.arange(256*256,dtype=np.float32).reshape(256,256)
        t=dict(pointmap_hw=[64,64],resized_hw=[256,256],crop_xyxy=[32,48,96,112])
        np.testing.assert_array_equal(align_probability(p,t),p[48:112,32:96])

    def test_isolated_inputs(self):
        c=torch.randn(1,4,8,8); k=torch.randn(1,3,2,8,8); e=torch.randn(1,3,6,8,8)
        for arm in ['N','B','G','BG']:
            m=IndependentEvidenceRefiner(arm); original=m.evidence_channels(c,k,e)
            changed=e.clone()
            if 'G' not in arm:
                changed[:,:,:5]+=100
                self.assertTrue(torch.equal(original,m.evidence_channels(c,k,changed)))
            changed=e.clone()
            if 'B' not in arm:
                changed[:,:,5]+=100
                self.assertTrue(torch.equal(original,m.evidence_channels(c,k,changed)))

    def test_candidate_permutation(self):
        torch.set_num_threads(1); m=IndependentEvidenceRefiner('BG').eval()
        torch.nn.init.normal_(m.line_score.weight)
        c=torch.randn(1,4,8,8); c[:,3]=1
        k=torch.randn(1,3,2,8,8); e=torch.randn(1,3,6,8,8); order=[2,0,1]
        torch.testing.assert_close(m(c,k,e)[:,order],m(c,k[:,order],e[:,order]))


if __name__=='__main__': unittest.main()
