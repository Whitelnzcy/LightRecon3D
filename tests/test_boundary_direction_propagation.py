import unittest
import torch
from models.boundary_direction_propagation import direction_propagation


class PropagationTests(unittest.TestCase):
    def test_interior_preference(self):
        f=torch.tensor([[[1.],[0.]]]);o=torch.tensor([[[1.,0,0],[-1.,0,0]]])
        y,w=direction_propagation(f,o,torch.tensor([[1.,0,0]]),torch.ones(1),1.)
        self.assertGreater(w[0,0],w[0,1]);torch.testing.assert_close(w.sum(-1),torch.ones(1))
    def test_zero_direction(self):
        f=torch.tensor([[[1.],[0.]]]);o=torch.tensor([[[1.,0,0],[-1.,0,0]]])
        y,w=direction_propagation(f,o,torch.zeros(1,3),torch.ones(1),1.)
        torch.testing.assert_close(y,torch.tensor([[.5]]))
    def test_gradient(self):
        f=torch.randn(2,4,5,requires_grad=True);o=torch.randn(2,4,3)
        d=torch.randn(2,3,requires_grad=True);p=torch.full((2,),.5,requires_grad=True)
        y,w=direction_propagation(f,o,d,p,1.);y.square().sum().backward()
        for x in [f,d,p]:self.assertTrue(torch.isfinite(x.grad).all())


if __name__=='__main__':unittest.main()
