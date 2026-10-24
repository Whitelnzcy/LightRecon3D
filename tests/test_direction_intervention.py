import unittest
import torch
from diagnose_direction_mechanism import expand_direction


class InterventionTests(unittest.TestCase):
    def test_same_all_candidates(self):
        t=torch.randn(2,3,8,8);expanded=expand_direction(t,4).reshape(2,4,3,8,8)
        for k in range(4):torch.testing.assert_close(expanded[:,k],t)
    def test_zero(self):
        self.assertEqual(float(expand_direction(torch.zeros(1,3,4,4),3).abs().sum()),0)


if __name__=='__main__':unittest.main()
