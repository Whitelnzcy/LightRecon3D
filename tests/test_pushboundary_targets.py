import unittest
import numpy as np
from audit_pushboundary_targets import directions,boundary_targets


class TargetTests(unittest.TestCase):
    def test_unit_and_zero(self):
        d,ok=directions(np.array([[0.,0,0],[3,4,0]]),np.array([[0.,0,0]]))
        np.testing.assert_allclose(d,[[0,0,0],[.6,.8,0]]);np.testing.assert_array_equal(ok,[False,True])
    def test_identity_permutation(self):
        x=np.arange(18).reshape(6,3).astype(float);y=np.array([0,0,0,1,1,1])
        np.testing.assert_array_equal(boundary_targets(x,y),boundary_targets(x,10-y))
    def test_empty(self):
        d,ok=directions(np.ones((2,3)),np.zeros((0,3)))
        self.assertFalse(ok.any());self.assertFalse(d.any())


if __name__=='__main__':unittest.main()
