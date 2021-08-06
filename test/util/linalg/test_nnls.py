from unittest import TestCase
import numpy as np
import scipy.optimize

from pynnls.nnls import svd_solve, nnls

class TestNnls(TestCase):
    def test_svd_solve(self):
        A = np.array([[1.0, 2.0], [3.1, 0.7]])
        b = np.array([0.75, 0.1])

        x1 = svd_solve(A, b)
        x2 = np.linalg.solve(A, b)

        self.assertTrue(np.all(np.isclose(x1, x2)))

        A = np.array([[1.0, 2.0], [3.1, 0.7], [0.9, 1.1]])
        b = np.array([0.75, 0.1, 0.9])

        x = svd_solve(A, b)

    def test_nnls(self):
        A = np.array([[1.0, 2.0, 1.5], [3.1, 0.7, 0.8], [0.9, 1.1, 0.4], [0.7, 0.95, 1.2]])
        b = np.array([0.75, 0.1, 0.9, 1.2])

        x1 = nnls(A, b)
        x2 = scipy.optimize.nnls(A, b)[0]
        
        self.assertTrue(np.all(np.isclose(x1, x2)))

