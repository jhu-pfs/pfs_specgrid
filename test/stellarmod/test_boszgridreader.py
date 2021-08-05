from test.test_base import TestBase
import os

from pfsspec.stellarmod.boszspectrumreader import BoszSpectrumReader
from pfsspec.stellarmod.boszgridreader import BoszGridReader
from pfsspec.stellarmod.modelgrid import ModelGrid

class TestBoszGridReader(TestBase):
    def test_read_grid_bosz(self):
        path = os.path.join(self.PFSSPEC_DATA_PATH, 'models/stellar/grid/bosz_5000/')
        
        reader = BoszSpectrumReader(path, wave_lim=[3000, 9000], res=5000)
        fn = BoszSpectrumReader.get_filename(Fe_H=0.0, T_eff=5000.0, log_g=1.0, O_M=0.0, C_M=0.0, R=5000)
        fn = os.path.join(path, fn)
        spec = reader.read(fn)

        grid = BoszModelGrid()
        grid.preload_arrays = True
        grid.wave = spec.wave
        grid.init_values()
        grid.build_axis_indexes()
        
        BoszGridReader(grid, reader, max=10, parallel=False).read_grid()
        self.assertEqual((14, 67, 11, 6, 4, 10986), grid.values['flux'].shape)

    def test_get_filename(self):
        self.skipTest()