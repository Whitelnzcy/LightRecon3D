import unittest
import numpy as np
from diagnose_spatial_training_damage import summarize_edits


class DamageDiagnosisTests(unittest.TestCase):
    def test_counts_are_registered_and_invalid_gt_is_excluded(self):
        lines=np.zeros((2,6,2,2),np.float32)
        lines[1,0,0]=1;lines[1,1,0]=[-.5,.5];lines[1,2,0]=[.1,.8]
        residual=np.zeros((2,2,2,2),np.float32)
        residual[0,1]=[[.1,.1],[.2,.2]];residual[1,1]=[[.2,.2],[.1,.1]]
        scene=dict(old=dict(point_plane_ids=np.array([0,0,1,1]),plane_ids=np.array([0,1])),
            truth=np.array([10,11,11,-1]),mapping={0:10,1:11},distance=np.array([2,25,30,1]),
            views=[dict(registry=np.arange(4).reshape(2,2),lines=lines,candidates=residual,
                        patches=[dict(box_xyxy=[0,0,1,2])])])
        result=summarize_edits(scene,np.array([1,1,0,0]));damaged=result['damaged'];fixed=result['corrected']
        self.assertEqual(damaged['total'],2);self.assertEqual(fixed['total'],1)
        self.assertEqual(damaged['within_5px_gt'],1);self.assertEqual(damaged['beyond_20px_gt'],1)
        self.assertEqual(damaged['destination_line_available'],1)
        self.assertEqual(damaged['line_side_agrees_with_destination'],0)
        self.assertEqual(damaged['beyond_20px_predicted_line'],1)
        self.assertEqual(damaged['covered_by_proposed_train_centers'],2)
        self.assertEqual(fixed['covered_by_proposed_train_centers'],0)
        self.assertEqual(fixed['line_side_agrees_with_destination'],1)
        self.assertEqual(fixed['beyond_20px_predicted_line'],0)
        self.assertEqual(damaged['new_plane_residual_greater'],2)


if __name__=='__main__':unittest.main()
