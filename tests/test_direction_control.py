import unittest
import torch
from models.spatial_direction_refiner import SpatialDirectionRefiner
from train_direction_control import NoPropagation


class ControlTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7);torch.set_num_threads(1)
        self.m=SpatialDirectionRefiner('DS').eval()
        torch.nn.init.normal_(self.m.score.weight)
        self.c=torch.randn(1,4,8,8);self.k=torch.randn(1,2,2,8,8)
        self.e=torch.randn(1,2,6,8,8);self.e[:,:,0]=1;self.e[:,:,4]=.5

    def test_override_matches_hook(self):
        t=torch.randn(1,2,3,8,8)
        explicit=self.m(self.c,self.k,self.e,direction_override=t)
        hook=self.m.line_direction.register_forward_hook(lambda m,i,o:t.reshape(2,3,8,8))
        try:hooked=self.m(self.c,self.k,self.e)
        finally:hook.remove()
        torch.testing.assert_close(explicit,hooked)

    def test_none_fuses_identical_features(self):
        captured=[]
        hook=self.m.line_fuse.register_forward_pre_hook(lambda m,i:captured.append(i[0].detach()))
        try:self.m(self.c,self.k,self.e,propagation='none')
        finally:hook.remove()
        torch.testing.assert_close(captured[0][:,:24],captured[0][:,24:])
        other=NoPropagation('DS').eval();other.load_state_dict(self.m.state_dict())
        torch.testing.assert_close(other(self.c,self.k,self.e),self.m(self.c,self.k,self.e,propagation='none'))

    def test_override_shape_checked(self):
        with self.assertRaises(ValueError):self.m(self.c,self.k,self.e,direction_override=torch.zeros(1,3,8,8))


if __name__=='__main__':unittest.main()
