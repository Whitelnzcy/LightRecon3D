import unittest
import torch
from spatial_direction_data import direction_loss

class UnitDirectionLossTests(unittest.TestCase):
    def test_scale_invariant_and_cosine_equivalence(self):
        p=torch.tensor([1.,1.,0.]).reshape(1,1,3,1,1)
        t=torch.tensor([1.,0.,0.]).reshape(1,3,1,1);y=torch.zeros(1,1,1,dtype=torch.long)
        a=direction_loss(p,t,y,normalize=True)
        torch.testing.assert_close(a,direction_loss(p*.1,t,y,normalize=True))
        self.assertAlmostEqual(a.item(),2*(1-2**-.5)/3,places=6)

    def test_zero_target_keeps_raw_penalty(self):
        p=torch.ones(1,1,3,1,1)*.2;t=torch.zeros(1,3,1,1);y=torch.zeros(1,1,1,dtype=torch.long)
        torch.testing.assert_close(direction_loss(p,t,y,normalize=True),direction_loss(p,t,y))

    def test_zero_prediction_finite_gradient_and_ignore(self):
        p=torch.zeros(1,2,3,1,2,requires_grad=True);t=torch.zeros(1,3,1,2);t[:,0]=1
        y=torch.tensor([[[1,-100]]]);loss=direction_loss(p,t,y,normalize=True);loss.backward()
        self.assertTrue(torch.isfinite(p.grad).all());self.assertEqual(p.grad[:,:,:,:,1].abs().sum().item(),0)
        self.assertEqual(p.grad[:,0].abs().sum().item(),0)

if __name__=='__main__':unittest.main()
