import numpy as np
from scipy.interpolate import Rbf as SciPyRbf

class Rbf(SciPyRbf):
    # This is a modified version of the original SciPy RBF that supports
    # saving and loading state

    def __init__(self):
        pass

    def fit(self, *args, **kwargs):
        super(Rbf, self).__init__(*args, **kwargs)

    def load(self, nodes, xi, **kwargs):
        # `args` can be a variable number of arrays; we flatten them and store
        # them as a single 2-D array `xi` of shape (n_args-1, array_size),
        # plus a 1-D array `di` for the values.
        # All arrays must have the same number of elements
        self.xi = xi
        self.N = self.xi.shape[-1]
        self.nodes = nodes

        self.mode = kwargs.pop('mode', '1-D')

        if self.mode == '1-D':
        #     self.di = np.asarray(args[-1]).flatten()
            self._target_dim = 1
        elif self.mode == 'N-D':
        #     self.di = np.asarray(args[-1])
            self._target_dim = self.nodes.shape[-1]
        else:
            raise ValueError("Mode has to be 1-D or N-D.")

        #if not all([x.size == self.di.shape[0] for x in self.xi]):
        #    raise ValueError("All arrays must be equal length.")

        self.norm = kwargs.pop('norm', 'euclidean')
        self.epsilon = kwargs.pop('epsilon', None)

        if self.epsilon is None:
            # default epsilon is the "the average distance between nodes" based
            # on a bounding hypercube
            ximax = np.amax(self.xi, axis=1)
            ximin = np.amin(self.xi, axis=1)
            edges = ximax - ximin
            edges = edges[np.nonzero(edges)]
            self.epsilon = np.power(np.prod(edges)/self.N, 1.0/edges.size)

        self.smooth = kwargs.pop('smooth', 0.0)
        self.function = kwargs.pop('function', 'multiquadric')

        # attach anything left in kwargs to self for use by any user-callable
        # function or to save on the object returned.
        for item, value in kwargs.items():
            setattr(self, item, value)

        # This is a dummy call to initialize the basis function
        self._init_function(np.array([0.0, 1.0]))
