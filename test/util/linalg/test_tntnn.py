from unittest import TestCase
import numpy as np

from pynnls.tntnn import tntnn, TntNN

class TestNnls(TestCase):
    def test_tntnn(self):
        A = np.array([[1, 2, 3, 4], [1.1, 1.2, 1.3, 1.4], [0.1, 0.2, 0.3, 0.4], [0.9, 0.8, 0.7, 0.6]])
        b = np.array([0.5, 0.6, 0.7, 0.8])

        x = tntnn(A, b)

        N = 100
        A = np.random.uniform(-1, 1, size=(N, N))
        b = np.random.uniform(-1, 1, size=(N,))

        x = tntnn(A, b)

        pass

    def test_qrdel(self):
        A = np.array([[1, 2, 3, 4], [1.1, 1.2, 1.3, 1.4], [0.1, 0.2, 0.3, 0.4], [0.9, 0.8, 0.7, 0.6]])
        Q, R = np.linalg.qr(A)

        tntnn = TntNN(None, None)
        R = tntnn._qrdel(R, 1)

        self.assertTrue(np.all(np.isclose(R, np.array(
            [[-1.74068952e+00, -2.92412860e+00, -3.51584814e+00],
             [ 0.00000000e+00,  1.64908215e+00,  2.47362323e+00],
             [ 0.00000000e+00,  0.00000000e+00,  6.44309255e-16]]))))

    def test_planerot(self):
        x = np.array([1, 2])

        tntnn = TntNN(None, None)
        G, x = tntnn._planerot(x)

        self.assertTrue(np.all(np.isclose(x, np.array([2.23606798,0.0]))))
        self.assertTrue(np.all(np.isclose(G, np.array([[ 0.4472136 , -0.89442719],
                                                       [ 0.89442719,  0.4472136 ]]))))