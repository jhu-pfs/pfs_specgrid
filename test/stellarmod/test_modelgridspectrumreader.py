import numpy as np
from test.test_base import TestBase

from pfsspec.stellarmod.modelgrid import ModelGrid
from pfsspec.data.gridaxis import GridAxis
from pfsspec.stellarmod.modelgridreader import ModelGridReader

# TODO: rewrite this to use customized configuration instead of an overloaded
#       grid class.

class TestGrid(ModelGrid):
    def __init__(self):
        super(TestGrid, self).__init__(use_cont=True)

        self.axes['Fe_H'] = GridAxis('Fe_H', np.array([0, 1, 2]))
        self.axes['T_eff'] = GridAxis('T_eff', np.array([1, 2]))
        self.axes['log_g'] = GridAxis('log_g', np.array([0, 5.5, 0.5]))

class TestGridReader(ModelGridReader):
    def process_item(self, i):
        print(i)

class TestModelGridSpectrumReader(TestBase):
    def test_enum_axes(self):
        grid = TestGrid()
        r = ModelGridSpectrumReader(grid)

        g = ModelGridSpectrumReader.EnumAxesGenerator(grid)
        k = 0
        for i in g:
            print(k, i)
            k += 1

        self.assertEqual(18, k)

    def test_read_grid(self):
        g = TestGrid()
        g.init_values(np.linspace(3000, 6000, 1))
        r = TestGridReader(g)
        r.read_grid()