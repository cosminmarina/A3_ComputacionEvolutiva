import random
import numpy as np
from numba import jit
from copy import deepcopy
from AE.Individual import *
from A2_ComputacionEvolutiva.ParamScheduler import ParamScheduler

"""
Population of individuals
"""
class DEPopulation:    
    """
    Constructor of the Population class
    """
    def __init__(self, objfunc, diffev_op, replace_op, params, population=None):
        self.params = params

        # Hyperparameters of the algorithm
        self.size = params["popSize"]
        self.diffev_op = diffev_op
        self.replace_op = replace_op

        # Data structures of the algorithm
        self.objfunc = objfunc

        # Population initialization
        if population is None:
            self.population = []
        else:
            self.population = population
        self.offspring = [None for i in range(self.size)]
       
    def step(self, progress):
        self.diffev_op.step(progress)
        self.replace_op.step(progress)

        if isinstance(self.params, ParamScheduler):
            self.params.step(progress)
            self.size = self.params["popSize"]
    """
    Gives the best solution found by the algorithm and its fitness
    """
    def best_solution(self):
        best_solution = sorted(self.population, reverse=True, key = lambda c: c.fitness)[0]
        best_fitness = best_solution.fitness
        if self.objfunc.opt == "min":
            best_fitness *= -1
        return (best_solution.vector, best_fitness)

    """
    Generates a random population of individuals
    """
    def generate_random(self):
        self.population = []
        for i in range(self.size):
            new_ind = Indiv(self.objfunc, self.objfunc.random_solution())
            self.population.append(new_ind)
            

    """
    Applies a random mutation to a small portion of individuals
    """
    def evolve(self):
        for idx, ind in enumerate(self.population):
            new_solution = self.diffev_op(ind, self.population, self.objfunc)
            self.offspring[idx] = Indiv(self.objfunc, new_solution)
    
    """
    Removes the worse solutions of the population
    """
    def selection(self):
        self.population = self.replace_op(self.population, self.offspring)

