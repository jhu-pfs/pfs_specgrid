import numpy as np

from pfsspec.stellarmod.bosz import Bosz
from pfsspec.stellarmod.modelgrid import ModelGrid

class BoszModelGrid(ModelGrid, Bosz):
    def __init__(self, orig=None):
        Bosz.__init__(self, orig=orig)
        ModelGrid.__init__(self, orig=orig)

    def init_axes(self):
        ModelGrid.init_axes(self)
        Bosz.init_axes(self)

    def init_values(self):
        ModelGrid.init_values(self)
        Bosz.init_values(self)