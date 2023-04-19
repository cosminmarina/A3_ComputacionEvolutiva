from ..AE import *
from .DEPopulation import *

class DE(AE):
    """
    Constructor of the Genetic algorithm
    """
    def __init__(self, objfunc, diffev_op, replace_op, params):
        super().__init__("DE", objfunc, params)

        # Data structures of the algorithm
        self.objfunc = objfunc
        self.diffev_op = diffev_op
        self.replace_op = replace_op
        self.population = DEPopulation(objfunc, diffev_op, replace_op, params)

    def restart(self):
        super().restart()

        self.population = DEPopulation(self.objfunc, self.diffev_op, self.replace_op, self.params)

    """
    One step of the algorithm
    """
    def step(self, progress):
        self.population.evolve()

        self.population.selection()

        self.population.step(progress)
        
        super().step(progress)