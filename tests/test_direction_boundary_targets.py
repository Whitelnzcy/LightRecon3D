import unittest
import numpy as np
from spatial_direction_data import extend_boundary_directions


class BoundaryTargetTests(unittest.TestCase):
    def test_toward_own_interior_and_preserve_rest(self):
        x=np.array([[0,0,0],[1,0,0],[2,0,0],[3,0,0]],dtype=np.float32)
        label=np.array([0,0,1,1]);boundary=np.array([False,True,True,False])
        target=np.array([[0,1,0],[0,0,0],[0,0,0],[0,-1,0]],dtype=np.float32)
        result,filled=extend_boundary_directions(x,label,boundary,target)
        np.testing.assert_array_equal(result[~boundary],target[~boundary])
        np.testing.assert_array_equal(result[1],[-1,0,0]);np.testing.assert_array_equal(result[2],[1,0,0]);self.assertEqual(filled,2)
        np.testing.assert_array_equal(target[boundary],0)

    def test_missing_interior_stays_zero(self):
        x=np.zeros((2,3),np.float32);label=np.array([0,1]);boundary=np.ones(2,bool)
        result,filled=extend_boundary_directions(x,label,boundary,np.zeros_like(x))
        self.assertEqual(filled,0);np.testing.assert_array_equal(result,0)


if __name__=='__main__':unittest.main()
