import numpy as np
import numexpr as ne
from scipy import linalg
from sklearn.metrics import pairwise_distances

from pfsspec.util.linalg.nnls import nnls as nnls
# from pfsspec.util.linalg.tntnn import tntnn

from pfsspec.util.name import func_fullname
from pfsspec.util.timer import Timer

class Rbf():
    # This is a modified version of the original SciPy RBF that supports
    # saving and loading state and uses parallel kernel evaluation

    def __init__(self):
        self.xi = None
        self.r = None
        self.A = None

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

        self._read_kwargs(**kwargs)
        self._read_args(*args)

        self._calculate_weights()

    def _read_kwargs(self, **kwargs):
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

    def _read_args(self, *args, **kwargs):
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

    def _get_epsilon(self):
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
        
        # TODO: implement _init_function of original RBF to evaluate custom kernels
        raise NotImplementedError()
        #super(Rbf, self)._init_function(r)

    def _calculate_distance_matrix(self, x1, x2=None):
        # Calculate pairwise distance

        with Timer('Calculating distance matrix...'):
            r = pairwise_distances(x1, Y=x2, metric=self.norm, n_jobs=-1)

        return r

    def _calculate_kernel_matrix(self, r):
        # Evaluate kernel over the distance matrix in parallel

        with Timer('Evaluating kernel...'):
            s = 1.0 / self.epsilon
            A = self._init_function(r)

        return A

    def _calculate_weights(self):
        if self.epsilon is None:
            self.epsilon = self._get_epsilon()

        if not all([x.size == self.di.shape[0] for x in self.xi]):
            raise ValueError("All arrays must be equal length.")

        if self.r is None:
            self.r = self._calculate_distance_matrix(self.xi.T)

        if self.A is None:
            self.A = self._calculate_kernel_matrix(self.r)

        A = self.A

        if self.method == 'solve':
            # TODO: delete, no offseting self.c = np.zeros_like(self.di[0, :])
            self.c = np.mean(self.di, axis=0)
            di = self.di - self.c
            if self._target_dim > 1:  
                # If we have more than one target dimension,
                # we first factorize the matrix then solve for each variable of di
             
                with Timer('Solving RBF of size {} with numpy.linalg.lu_factor...'.format(A.shape)):
                    lu, piv = linalg.lu_factor(A)
                    self.nodes = np.zeros((self.N, self._target_dim), dtype=di.dtype)
                    for i in range(self._target_dim):
                        self.nodes[:, i] = linalg.lu_solve((lu, piv), di[:, i])
            else:
                with Timer('Solving RBF if size {} with numpy.linalg.solve...'.format(A.shape)):
                    self.nodes = linalg.solve(A, di)
        elif self.method == 'nnls':
            self.c = np.min(self.di, axis=0)
            A = self.A
            di = self.di - self.c
            if self._target_dim > 1:
                self.nodes = np.zeros((self.N, self._target_dim), dtype=self.di.dtype)
                with Timer('Solving RBF of shape {} with {}...'.format(A.shape, func_fullname(nnls))):
                    for i in range(self._target_dim):
                        n = nnls(A, di[:, i])
                        self.nodes[:, i] = n  
            else:
                with Timer('Solving RBF of shape {} with {}...'.format(A.shape, func_fullname(nnls))):
                    self.nodes = nnls(A, di)
        else:
            raise ValueError('Method has to be `solve` or `nnls`.')

        pass
            
    def load(self, nodes, xi, c=0.0, **kwargs):
        # `args` can be a variable number of arrays; we flatten them and store
        # them as a single 2-D array `xi` of shape (n_args-1, array_size),
        # plus a 1-D array `di` for the values.
        # All arrays must have the same number of elements

        self._read_kwargs(**kwargs)

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
            self.epsilon = self._get_epsilon()

        # This is a dummy call to initialize the basis function
        self._init_function(np.array([0.0, 1.0]))

    def eval(self, *args):
        args = [np.asarray(x) for x in args]
        if not all([x.shape == y.shape for x in args for y in args]):
            raise ValueError("Array lengths must be equal")
        if self._target_dim > 1:
            shp = args[0].shape + (self._target_dim,)
        else:
            shp = args[0].shape
        xa = np.asarray([a.flatten() for a in args], dtype=np.float_)

        r = self._calculate_distance_matrix(xa.T, self.xi.T)

        y = np.dot(self._function(r), self.nodes).reshape(shp)
        y += self.c
        return y

    def __call__(self, *args):
        return self.eval(*args)
