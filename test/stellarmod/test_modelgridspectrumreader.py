import numpy as np
from test.test_base import TestBase

from pfsspec.stellarmod.modelgrid import ModelGrid
from pfsspec.data.gridparam import GridParam
from pfsspec.stellarmod.modelgridspectrumreader import ModelGridSpectrumReader

class TestGrid(ModelGrid):
    def __init__(self):
        super(TestGrid, self).__init__(use_cont=True)

        self.params['Fe_H'] = GridParam('Fe_H', np.array([0, 1, 2]))
        self.params['T_eff'] = GridParam('T_eff', np.array([1, 2]))
        self.params['log_g'] = GridParam('log_g', np.array([0, 5.5, 0.5]))

class TestGridReader(ModelGridSpectrumReader):
    def process_item(self, i):
        print(i)

class TestModelGridSpectrumReader(TestBase):
    def test_enum_parameters(self):
        print('Executing test_enum_parameters')

        grid = TestGrid()
        r = ModelGridSpectrumReader(grid)

        g = ModelGridSpectrumReader.EnumParamsGenerator(grid)
        k = 0
        for i in g:
            print(k, i)
            k += 1

        self.assertEqual(18, k)

    def test_read_grid(self):
        g = TestGrid()
        g.init_data(np.linspace(3000, 6000, 1))
        r = TestGridReader(g)
        r.read_grid()