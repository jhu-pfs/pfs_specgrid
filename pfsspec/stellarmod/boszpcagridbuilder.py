from pfsspec.data.pcagridbuilder import PCAGridBuilder
from pfsspec.stellarmod.modelpcagridbuilder import ModelPCAGridBuilder
from pfsspec.stellarmod.boszmodelgrid import BoszModelGrid
from pfsspec.stellarmod.boszpcagrid import BoszPCAGrid
from pfsspec.stellarmod.bosz import Bosz

class BoszPCAGridBuilder(ModelPCAGridBuilder, Bosz):
    def __init__(self, grid=None, orig=None):
        ModelPCAGridBuilder.__init__(self, grid=grid, orig=orig)
        Bosz.__init__(self)

    def create_grid(self):
        return BoszModelGrid()

    def create_pca_grid(self):
        return BoszPCAGrid()