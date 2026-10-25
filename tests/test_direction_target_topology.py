import unittest
import numpy as np
from audit_direction_target_topology import edge_diagnostic


class TopologyTests(unittest.TestCase):
    def test_directed_edges_and_coordinate_order(self):
        registry=np.array([[0,1],[2,-1]])
        xyz=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
        labels=np.array([0,1,0]);target=np.tile([1.,0.,0.],(3,1))
        rows=edge_diagnostic(registry,xyz,labels,target,np.ones(3,bool))
        edges={(int(i),int(j)):float(c) for a,b,cs,_,_ in rows for i,j,c in zip(a,b,cs)}
        self.assertEqual(edges,{(0,1):1.,(1,0):-1.,(0,2):0.,(2,0):0.})

    def test_invalid_labels_removed(self):
        rows=edge_diagnostic(np.array([[0,1]]),np.array([[0.,0.,0.],[1.,0.,0.]]),np.array([-1,0]),np.ones((2,3)),np.ones(2,bool))
        self.assertEqual(sum(len(r[0]) for r in rows),0)


if __name__=='__main__':unittest.main()
