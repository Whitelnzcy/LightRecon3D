import unittest
import numpy as np
from spatial_direction_data import registered_grid_boundaries,attach_direction_supervision


class GridTargetTests(unittest.TestCase):
    def test_only_valid_transitions_and_explicit_registry(self):
        labels=np.array([0,0,1,1,-1])
        result=registered_grid_boundaries([np.array([[0,1,2,3,4]])],labels)
        np.testing.assert_array_equal(result,[False,True,True,False,False])

    def test_no_cross_view_adjacency(self):
        result=registered_grid_boundaries([np.array([[0,1]]),np.array([[2,3]])],np.array([0,0,1,1]))
        self.assertFalse(result.any())

    def test_boundary_inward_and_interior_away_in_global_xyz(self):
        labels=np.array([0,0,1,1]);xyz=np.c_[np.arange(4),np.zeros((4,2))].astype(np.float32)
        s=dict(truth=labels,old=dict(points=xyz),views=[dict(registry=np.arange(4)[None])])
        attach_direction_supervision(s,'grid_inward')
        target=s['views'][0]['direction_target']
        np.testing.assert_array_equal(target[0,0],[-1,-1,1,1])
        np.testing.assert_array_equal(target[1:],0)
        np.testing.assert_array_equal(s['views'][0]['direction_boundary'],[[False,True,True,False]])


if __name__=='__main__':unittest.main()
