import unittest
import numpy as np
from diagnose_finite_structure_lines import intersect3d,runs,sample_evidence


class FiniteLineTests(unittest.TestCase):
    def test_equations_and_sign(self):
        p,v=intersect3d([1,0,0],-2,[0,1,0],-3)
        np.testing.assert_allclose(p,[2,3,0]);self.assertAlmostEqual(abs(v[2]),1)
        p2,v2=intersect3d([-1,0,0],2,[0,1,0],-3)
        np.testing.assert_allclose(p,p2);self.assertAlmostEqual(abs(v@v2),1)
    def test_parallel_and_runs(self):
        self.assertIsNone(intersect3d([1,0,0],0,[1,0,0],2))
        self.assertEqual(runs([1,1,0,1,0,1,1]),[(0,2),(5,7)])
    def test_no_image_support_no_vote(self):
        p=np.array([[1.,0,0,0],[0,1,0,0],[0,0,1,0]])
        x=np.array([[2.,2,1],[8,8,1]])
        score,valid=sample_evidence(x,p,np.ones((10,10),np.float32),x[:1],np.array([[2.,2]]),.1)
        np.testing.assert_array_equal(valid,[True,False])


if __name__=='__main__':unittest.main()
