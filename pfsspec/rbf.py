import numpy as np
from scipy.interpolate import Rbf as SciPyRbf
from scipy.optimize import nnls
from scipy import linalg

class Rbf(SciPyRbf):
    # This is a modified version of the original SciPy RBF that supports
    # saving and loading state

    def __init__(self):
        pass

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

    def calculate_weights(self):
        if self.epsilon is None:
            self.epsilon = self.get_epsilon()

        if not all([x.size == self.di.shape[0] for x in self.xi]):
            raise ValueError("All arrays must be equal length.")

        if self.method == 'solve':
            self.c = np.zeros_like(self.di[0, :])
            if self._target_dim > 1:  # If we have more than one target dimension,
                # we first factorize the matrix
                self.nodes = np.zeros((self.N, self._target_dim), dtype=self.di.dtype)
                lu, piv = linalg.lu_factor(self.A)
                for i in range(self._target_dim):
                    self.nodes[:, i] = linalg.lu_solve((lu, piv), self.di[:, i])
            else:
                self.nodes = linalg.solve(self.A, self.di)
        elif self.method == 'nnls':
            self.c = np.min(self.di, axis=0)
            A = self.A
            di = self.di - self.c
            if self._target_dim > 1:
                self.nodes = np.zeros((self.N, self._target_dim), dtype=self.di.dtype)
                for i in range(self._target_dim):
                    n, _ = nnls(A, di[:, i])
                    self.nodes[:, i] = n
            else:
                self.nodes, _ = nnls(A, di)
        else:
            raise ValueError("Method has to be solve or nnls.")
            
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
