from ..AE import *
from .ESPopulation import *

class ES(AE):
    """
    Constructor of the Genetic algorithm
    """
    def __init__(self, objfunc, mutation_op, cross_op, parent_sel_op, replace_op, params):
        super().__init__("ES", objfunc, params)

        # Data structures of the algorithm
        self.objfunc = objfunc
        self.mutation_op = mutation_op
        self.cross_op = cross_op
        self.parent_sel_op = parent_sel_op
        self.replace_op = replace_op
        self.population = ESPopulation(objfunc, mutation_op, cross_op, parent_sel_op, replace_op, params)

    def restart(self):
        super().restart()

        self.population = ESPopulation(self.objfunc, self.mutation_op, self.cross_op,self.parent_sel_op, self.replace_op, self.params)

    """
    One step of the algorithm
    """
    def step(self, progress):
        self.population.cross()
        
        self.population.mutate()

        self.population.selection()

        self.population.step(progress)
        
        super().step(progress)