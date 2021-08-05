import numpy as np
import numexpr as ne
from scipy.interpolate import Rbf as SciPyRbf
from pynnls.nnls import nnls
from pynnls.tntnn import TntNN, tntnn
from fnnls import fnnls
from scipy import linalg
from sklearn.metrics import pairwise_distances
from pfsspec.util.timer import Timer

class Rbf(SciPyRbf):
    # This is a modified version of the original SciPy RBF that supports
    # saving and loading state

    def __init__(self):
        pass

    def _numexpr_multiquadric(self, r):
        eps = self.epsilon
        return ne.evaluate('sqrt((1.0/eps*r)**2+1)')

    def _numexpr_inverse_multiquadric(self, r):
        eps = self.epsilon
        return ne.evaluate('1.0/sqrt((1.0/eps*r)**2+1)')

    def _numexpr_gaussian(self, r):
        eps = self.epsilon
        return ne.evaluate('exp(-(1.0/eps*r**2))')

    def _numexpr_linear(self, r):
        return r

    def _numexpr_cubic(self, r):
        return ne.evaluate('r**3')

    def _numexpr_quintic(self, r):
        return ne.evaluate('r**5')

    def _numexpr_thin_plate(self, r):
        raise NotImplementedError()
        #return xlogy(r**2, r)

    def fit(self, *args, **kwargs):
        # `args` can be a variable number of arrays; we flatten them and store
        # them as a single 2-D array `xi` of shape (n_args-1, array_size),
        # plus a 1-D array `di` for the values.
        # All arrays must have the same number of elements

        self.read_kwargs(**kwargs)
        self.read_args(*args)

        self.calculate_weights()

    def read_kwargs(self, **kwargs):
        self.mode = kwargs.pop('mode', '1-D')
        self.method = kwargs.pop('method', 'solve')
        self.norm = kwargs.pop('norm', 'euclidean')
        self.epsilon = kwargs.pop('epsilon', None)
        self.smooth = kwargs.pop('smooth', 0.0)
        self.function = kwargs.pop('function', 'multiquadric')

        # attach anything left in kwargs to self for use by any user-callable
        # function or to save on the object returned.
        for item, value in kwargs.items():
            setattr(self, item, value)

    def read_args(self, *args, **kwargs):
        self.xi = np.asarray([np.asarray(a, dtype=np.float_).flatten()
                              for a in args[:-1]])
        self.N = self.xi.shape[-1]
        self.di = np.asarray(args[-1])

        if self.mode == '1-D':
            self.di = self.di.flatten()
            self._target_dim = 1
        elif self.mode == 'N-D':
            self._target_dim = self.di.shape[-1]
        else:
            raise ValueError("Mode has to be 1-D or N-D.")

    def get_epsilon(self):
        # default epsilon is the "the average distance between nodes" based
        # on a bounding hypercube
        ximax = np.amax(self.xi, axis=1)
        ximin = np.amin(self.xi, axis=1)
        edges = ximax - ximin
        edges = edges[np.nonzero(edges)]
        return np.power(np.prod(edges) / self.N, 1.0 / edges.size)

    def _init_function(self, r):
        # Try with numexpr of fall back to scipy implementation
        if isinstance(self.function, str):
            self.function = self.function.lower()
            _mapped = {'inverse': 'inverse_multiquadric',
                       'inverse multiquadric': 'inverse_multiquadric',
                       'thin-plate': 'thin_plate'}
            if self.function in _mapped:
                self.function = _mapped[self.function]

            func_name = "_numexpr_" + self.function
            if hasattr(self, func_name):
                self._function = getattr(self, func_name)
                a0 = self._function(r)
                return a0
        
        super(Rbf, self)._init_function(r)

    def calculate_kernel_matrix(self):
        # Calculate pairwise distance and evaluate kernel in parallel

        with Timer('Calculating distance matrix...'):
            r = pairwise_distances(self.xi.T, metric=self.norm, n_jobs=-1)

        with Timer('Evaluating kernel...'):
            s = 1.0 / self.epsilon
            A = self._init_function(r)

        return A

    def calculate_weights(self):
        if self.epsilon is None:
            self.epsilon = self.get_epsilon()

        if not all([x.size == self.di.shape[0] for x in self.xi]):
            raise ValueError("All arrays must be equal length.")

        A = self.calculate_kernel_matrix()

        if self.method == 'solve':
            self.c = np.zeros_like(self.di[0, :])
            if self._target_dim > 1:  
                # If we have more than one target dimension,
                # we first factorize the matrix then solve for each variable of di
             
                with Timer('Solving RBF with numpy.linalg.lu_factor...'):
                    lu, piv = linalg.lu_factor(A)
                    self.nodes = np.zeros((self.N, self._target_dim), dtype=self.di.dtype)
                    for i in range(self._target_dim):
                        self.nodes[:, i] = linalg.lu_solve((lu, piv), self.di[:, i])
            else:
                with Timer('Solving RBF with numpy.linalg.solve...'):
                    self.nodes = linalg.solve(A, self.di)
        elif self.method == 'nnls':
            self.c = np.min(self.di, axis=0)
            A = self.A
            di = self.di - self.c
            if self._target_dim > 1:
                self.nodes = np.zeros((self.N, self._target_dim), dtype=self.di.dtype)
                with Timer('Solving RBF with scipy.optimize.nnls...'):
                    for i in range(self._target_dim):
                        n, _ = nnls(A, di[:, i])
                        self.nodes[:, i] = n  
            else:
                with Timer('Solving RBF with scipy.optimize.nnls...'):
                    self.nodes, _ = nnls(A, di)
        else:
            raise ValueError("Method has to be solve or nnls.")

        pass
            
    def load(self, nodes, xi, c=0.0, **kwargs):
        # `args` can be a variable number of arrays; we flatten them and store
        # them as a single 2-D array `xi` of shape (n_args-1, array_size),
        # plus a 1-D array `di` for the values.
        # All arrays must have the same number of elements

        self.read_kwargs(**kwargs)

        self.xi = xi
        self.N = self.xi.shape[-1]
        self.nodes = nodes
        self.c = c

        if self.mode == '1-D':
            self._target_dim = 1
        elif self.mode == 'N-D':
            self._target_dim = self.nodes.shape[-1]
        else:
            raise ValueError("Mode has to be 1-D or N-D.")

        if self.epsilon is None:
            self.epsilon = self.get_epsilon()

        # This is a dummy call to initialize the basis function
        self._init_function(np.array([0.0, 1.0]))

    def __call__(self, *args):
        y = super(Rbf, self).__call__(*args)
        y += self.c
        return y
