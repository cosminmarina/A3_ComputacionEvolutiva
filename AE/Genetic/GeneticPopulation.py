import random
import numpy as np
from numba import jit
from copy import deepcopy
from AE.Individual import *
from A3_ComputacionEvolutiva.ParamScheduler import ParamScheduler


class GeneticPopulation:
    """
    Population of the Genetic algorithm
    """

    def __init__(self, objfunc, mutation_op, cross_op, parent_sel_op, replace_op, params, population=None):
        """
        Constructor of the GeneticPopulation class
        """

        self.params = params

        # Hyperparameters of the algorithm
        self.size = params["popSize"] if "popSize" in params else 100
        self.pmut = params["pmut"] if "pmut" in params else 0.1
        self.pcross = params["pcross"] if "pcross" in params else 0.9
        self.mutation_op = mutation_op
        self.cross_op = cross_op
        self.parent_sel_op = parent_sel_op
        self.replace_op = replace_op

        # Data structures of the algorithm
        self.objfunc = objfunc

        # Population initialization
        if population is None:
            self.population = []
        else:
            self.population = population
        self.offspring = []       

    def step(self, progress):
        """
        Updates the parameters and the operators
        """

        self.mutation_op.step(progress)
        self.cross_op.step(progress)
        self.parent_sel_op.step(progress)
        self.replace_op.step(progress)

        if isinstance(self.params, ParamScheduler):
            self.params.step(progress)
            self.size = self.params["popSize"]
            self.pmut = self.params["pmut"]
            self.pcross = self.params["pcross"]

    def best_solution(self):
        """
        Gives the best solution found by the algorithm and its fitness
        """

        best_solution = sorted(self.population, reverse=True, key = lambda c: c.fitness)[0]
        best_fitness = best_solution.fitness
        if self.objfunc.opt == "min":
            best_fitness *= -1
        return (best_solution.vector, best_fitness)

    def generate_random(self):
        """
        Generates a random population of individuals
        """

        self.population = []
        for i in range(self.size):
            new_ind = Indiv(self.objfunc, self.objfunc.random_solution())
            self.population.append(new_ind)
    

    def cross(self):
        """
        Applies the crossing operator to the population of individuals
        """

        parent_list = self.parent_sel_op(self.population)

        self.offspring = []
        while len(self.offspring) < self.size:
            parent1 = random.choice(parent_list)
            if random.random() < self.pcross:
                new_solution = self.cross_op.evolve(parent1, parent_list, self.objfunc)
                new_solution = self.objfunc.check_bounds(new_solution)
                new_ind = Indiv(self.objfunc, new_solution)
            else:
                new_ind = deepcopy(parent1)
            
            self.offspring.append(new_ind)
            

    def mutate(self):
        """
        Applies a mutation operator to the offspring generated 
        """

        for idx, ind in enumerate(self.offspring):
            if random.random() < self.pmut:
                new_solution = self.mutation_op(ind, self.population, self.objfunc)
                new_solution = self.objfunc.check_bounds(new_solution)
                self.offspring[idx] = Indiv(self.objfunc, new_solution)
    

    def selection(self):
        """
        Selects the individuals that will pass to the next generation
        """

        self.population = self.replace_op(self.population, self.offspring)
    
    



