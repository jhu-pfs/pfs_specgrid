from pfsspec.stellarmod.modelgridfit import ModelGridFit
from pfsspec.stellarmod.boszmodelgrid import BoszModelGrid
from pfsspec.stellarmod.bosz import Bosz
from pfsspec.stellarmod.logchebyshevcontinuummodel import LogChebyshevContinuumModel

class BoszGridContinuumFit(ModelGridFit, Bosz):
    def __init__(self, grid=None, orig=None):
        ModelGridFit.__init__(self, grid=grid, orig=orig)
        Bosz.__init__(self, orig=orig)

    def create_grid(self):
        return BoszModelGrid()