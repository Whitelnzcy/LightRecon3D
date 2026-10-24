import unittest
import torch
from models.line_permission_loss import line_permission_loss


class PermissionTests(unittest.TestCase):
    def test_gradient_preserves_correct_and_supports_correct_target(self):
        g=torch.full((1,3,1,3),.5,requires_grad=True);a=torch.ones_like(g)
        original=torch.tensor([[[0,0,0]]]);target=torch.tensor([[[0,1,-100]]])
        line_permission_loss(g,a,target,original).backward()
        self.assertGreater(float(g.grad[0,1,0,0]),0)  # correct original: suppress alternative
        self.assertLess(float(g.grad[0,1,0,1]),0)  # wrong original: enable true alternative
        self.assertGreater(float(g.grad[0,2,0,1]),0)  # suppress wrong alternative
        self.assertFalse(g.grad[:,0].any());self.assertFalse(g.grad[:,:,:,2].any())

    def test_unavailable_and_unassigned_have_zero_loss(self):
        g=torch.full((1,2,2,2),.5,requires_grad=True);t=torch.zeros(1,2,2,dtype=torch.long)
        loss=line_permission_loss(g,torch.zeros_like(g),t,t)
        self.assertEqual(float(loss.detach()),0);loss.backward();self.assertFalse(g.grad.any())
        self.assertEqual(float(line_permission_loss(g,torch.ones_like(g),t,t-1).detach()),0)

    def test_candidate_permutation_and_positive_negative_balance(self):
        g=torch.tensor([[[[.2,.3]],[[.7,.8]],[[.4,.6]]]])
        a=torch.ones_like(g);t=torch.tensor([[[1,0]]]);o=torch.zeros_like(t)
        p=torch.tensor([2,0,1]);inverse=torch.argsort(p)
        torch.testing.assert_close(line_permission_loss(g,a,t,o),line_permission_loss(g[:,p],a[:,p],inverse[t],inverse[o]))


if __name__=='__main__':unittest.main()
