from pfsspec.data.pcagridbuilder import PcaGridBuilder
from pfsspec.stellarmod.modelpcagridbuilder import ModelPcaGridBuilder
from pfsspec.stellarmod.boszmodelgrid import BoszModelGrid
from pfsspec.stellarmod.boszpcagrid import BoszPcaGrid
from pfsspec.stellarmod.bosz import Bosz

class BoszPCAGridBuilder(ModelPcaGridBuilder, Bosz):
    def __init__(self, grid=None, orig=None):
        ModelPcaGridBuilder.__init__(self, grid=grid, orig=orig)
        Bosz.__init__(self)

    def create_grid(self):
        return BoszModelGrid()

    def create_pca_grid(self):
        return BoszPcaGrid()