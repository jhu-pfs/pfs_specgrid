from pfsspec.stellarmod.bosz import Bosz
from pfsspec.stellarmod.modelpcagrid import ModelPcaGrid
from pfsspec.stellarmod.boszmodelgrid import BoszModelGrid

class BoszPcaGrid(ModelPcaGrid, Bosz):
    def __init__(self, orig=None):
        Bosz.__init__(self, orig=orig)
        ModelPcaGrid.__init__(self, orig=orig)

    def init_axes(self):
        ModelPcaGrid.init_axes(self)
        Bosz.init_axes(self)

    def init_values(self):
        ModelPcaGrid.init_values(self)
        Bosz.init_values(self)

    def create_pca_grid(self):
        return BoszModelGrid()