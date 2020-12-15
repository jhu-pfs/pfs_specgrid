from pfsspec.stellarmod.bosz import Bosz
from pfsspec.stellarmod.modelpcagrid import ModelPCAGrid

class BoszPCAGrid(ModelPCAGrid, Bosz):
    def __init__(self, orig=None):
        Bosz.__init__(self, orig=orig)
        ModelPCAGrid.__init__(self, orig=orig)

    def init_axes(self):
        ModelPCAGrid.init_axes(self)
        Bosz.init_axes(self)

    def init_values(self):
        ModelPCAGrid.init_values(self)
        Bosz.init_values(self)