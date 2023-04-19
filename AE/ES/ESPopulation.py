import random
import numpy as np
from numba import jit
from copy import deepcopy
from AE.Individual import *
from A3_ComputacionEvolutiva.ParamScheduler import ParamScheduler

"""
Population of individuals
"""
class ESPopulation:    
    """
    Constructor of the Population class
    """
    def __init__(self, objfunc, mutation_op, cross_op, parent_sel_op, replace_op, params, population=None, sigmas=None):
        self.params = params

        # Hyperparameters of the algorithm
        self.size = params["popSize"]
        self.n_offspring = params["offspringSize"]
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

        # Step size parameters = sigma(s)
        if sigmas is None:
            self.sigmas = []
        else:
            self.sigmas = sigmas
        self.sigma_type = params["sigma_type"]
        if "max_sigma" in params:
            self.max_sigma = params["max_sigma"]
        else:
            self.max_sigma = 1
        self.offspring_sigmas = []
        if "tau" in params:
            self.tau = params["tau"]
        else:
            self.tau = 1/np.sqrt(self.objfunc.size) if self.sigma_type == "1stepsize" else 1/np.sqrt(2*self.objfunc.size)
        if "epsilon" in params:
            self.epsilon = params["epsilon"]
        else:
            self.epsilon = 1e-17
        if "tau_multiple" in params:
            self.tau_multiple = params["tau_multiple"]
        else:
            self.tau_multiple = 1/np.sqrt(2*np.sqrt(self.objfunc.size))

    def step(self, progress):
        self.mutation_op.step(progress)
        self.cross_op.step(progress)
        self.parent_sel_op.step(progress)
        self.replace_op.step(progress)

        if isinstance(self.params, ParamScheduler):
            self.params.step(progress)
            self.size = self.params["popSize"]
            self.n_offspring = self.params["offspringSize"]

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
        self.sigmas = []
        for i in range(self.size):
            new_ind = Indiv(self.objfunc, self.objfunc.random_solution())
            self.population.append(new_ind)
            if self.sigma_type == "1stepsize":
                new_sigma = self.objfunc.random_solution(0,self.max_sigma,1)
                self.sigmas.append(new_sigma)
            elif self.sigma_type == "nstepsize":
                new_sigmas = self.objfunc.random_solution(0,self.max_sigma)
                self.sigmas.append(new_sigmas)
        
    
    """
    Crosses individuals of the population
    """
    def cross(self):
        parent_list = self.parent_sel_op(self.population)

        self.offspring = []
        self.offspring_sigmas = []
        #print('Sigmas: ', self.sigmas)
        for i in range(self.n_offspring):
            parent_idx = np.random.choice(np.arange(self.size))
            parent1 = parent_list[parent_idx]
            #print('Iteration: ',i)
            self.offspring_sigmas.append(np.array(self.cross_op(parent1, parent_list, self.objfunc, self.sigmas, parent_idx, (self.sigma_type=="1stepsize"))))
            new_solution = self.objfunc.check_bounds(self.cross_op(parent1, parent_list, self.objfunc))
            new_ind = Indiv(self.objfunc, new_solution)
            self.offspring.append(Indiv(self.objfunc, new_solution))

    """
    Applies a random mutation to a small portion of individuals
    """
    def mutate(self):
        for idx, ind in enumerate(self.offspring):
            self.offspring_sigmas[idx] = np.array(self.mutate_sigma(self.offspring_sigmas[idx]))
            new_solution = self.objfunc.check_bounds(self.mutation_op(ind, self.population, self.objfunc, self.offspring_sigmas, idx))
            self.offspring[idx] = Indiv(self.objfunc, new_solution)
            
    """
    Applies a random mutation to sigma
    """
    def mutate_sigma(self, sigma):
        if self.sigma_type == "1stepsize":
            return max(self.epsilon, np.exp(self.tau * np.random.normal()))
        elif self.sigma_type == "nstepsize":
            base_tau = self.tau * np.random.normal()
            new_sigmas = np.array([sigma[i] * np.exp(base_tau + self.tau_multiple * np.random.normal()) for i in range(len(sigma))])
            return list(np.vectorize((lambda sigma_i: max(self.epsilon,sigma_i))) (new_sigmas))

    """
    Removes the worse solutions of the population
    """
    def selection(self):
        [self.population, self.sigmas] = self.replace_op(self.population, self.offspring, self.sigmas, self.offspring_sigmas)

