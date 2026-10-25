import unittest
import torch
from models.spatial_direction_refiner import SpatialDirectionRefiner

class CapacityTests(unittest.TestCase):
    def test_shapes_gradients_and_counts(self):
        torch.set_num_threads(1)
        counts=[]
        for width in (1,2):
            model=SpatialDirectionRefiner('DS',width=width)
            c=torch.randn(1,4,8,8);k=torch.randn(1,2,2,8,8);e=torch.randn(1,2,6,8,8);e[:,:,0]=1
            logits,d=model(c,k,e,return_details=True)
            self.assertEqual(tuple(logits.shape),(1,2,8,8));self.assertEqual(tuple(d.shape),(1,2,3,8,8))
            (logits.square().mean()+d.square().mean()).backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            counts.append(sum(p.numel() for p in model.parameters()))
        self.assertGreater(counts[1],3*counts[0])

    def test_default_unchanged(self):
        torch.manual_seed(42);a=SpatialDirectionRefiner('DS')
        torch.manual_seed(42);b=SpatialDirectionRefiner('DS',width=1)
        for k,v in a.state_dict().items():self.assertTrue(torch.equal(v,b.state_dict()[k]))

if __name__=='__main__':unittest.main()
