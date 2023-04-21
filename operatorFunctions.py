import math
import random

import numpy as np
import scipy as sp
import scipy.stats


## Mutation and recombination methods
def mutateRand(vector, population, params):
    method = params["method"]
    n = params["N"]

    low = params["Low"] if "Low" in params else -1
    up = params["Up"] if "Low" in params else 1
    strength = params["F"] if "F" in params else 1
    
    mask_pos = np.hstack([np.ones(n), np.zeros(vector.size - n)]).astype(bool)
    np.random.shuffle(mask_pos)
    
    rand_vec = sampleDistribution(method, n, 0, strength, low, up)
    
    vector[mask_pos] = vector[mask_pos] + rand_vec
    return vector

def mutateSample(vector, population, params):
    method = params["method"]
    n = params["N"]

    low = params["Low"] if "Low" in params else -1
    up = params["Up"] if "Low" in params else 1
    strength = params["F"] if "F" in params else 1

    mask_pos = np.hstack([np.ones(n), np.zeros(vector.size - n)]).astype(bool)
    np.random.shuffle(mask_pos)
    popul_matrix = np.vstack([i.vector for i in population])
    mean = popul_matrix.mean(axis=0)[mask_pos]
    std = (popul_matrix.std(axis=0)[mask_pos] + 1e-6)*strength # ensure there will be some standard deviation
    
    rand_vec = sampleDistribution(method, n, mean, std, low, up)
    
    vector[mask_pos] = rand_vec
    return vector

def randSample(vector, population, params):
    method = params["method"]
    
    low = params["Low"] if "Low" in params else -1
    up = params["Up"] if "Low" in params else 1
    strength = params["F"] if "F" in params else 1

    popul_matrix = np.vstack([i.vector for i in population])
    mean = popul_matrix.mean(axis=0)
    std = (popul_matrix.std(axis=0) + 1e-6)*strength # ensure there will be some standard deviation
    
    rand_vec = sampleDistribution(method, vector.shape, mean, std, low, up)
    
    return rand_vec

def randNoise(vector, params):
    method = params["method"]

    low = params["Low"] if "Low" in params else -1
    up = params["Up"] if "Low" in params else 1
    strength = params["F"] if "F" in params else 1
    
    noise = sampleDistribution(method, vector.shape, 0, strength, low, up)
    
    return vector + noise

def sampleDistribution(method, n, mean=0, strength=0.01, low=0, up=1):
    sample = 0 
    if method == "Gauss":
        sample = np.random.normal(mean, strength, size=n)
    elif method == "Uniform":
        sample = np.random.uniform(low, up, size=n)
    elif method == "Cauchy":
        sample = sp.stats.cauchy.rvs(mean, strength, size=n)
    elif method == "Poisson":
        sample = sp.stats.poisson.rvs(mean, strength, size=n)
    elif method == "Laplace":
        sample = sp.stats.laplace.rvs(mean, strength, size=n)
    else:
        print(f"Error: distribution \"{method}\" not defined")
        exit(1)
    return sample

def laplace(vector, strength):
    """
    Adds random noise following a Laplace distribution to the vector.
    """

    return randNoise(vector, {"method":"Laplace", "F":strength})


def cauchy(vector, strength):
    """
    Adds random noise following a Cauchy distribution to the vector.
    """
    
    return randNoise(vector, {"method":"Cauchy", "F":strength})


def gaussian(vector, strength):
    """
    Adds random noise following a Gaussian distribution to the vector.
    """
    
    return randNoise(vector, {"method":"Gauss", "F":strength})


def uniform(vector, low, up):
    """
    Adds random noise following an Uniform distribution to the vector.
    """
    
    return randNoise(vector, {"method":"Uniform", "Low":low, "Up":up})


def poisson(vector, mu):
    """
    Adds random noise following a Poisson distribution to the vector.
    """
    
    return randNoise(vector, {"method":"Poisson", "F":mu})

def cross1p(vector1, vector2):
    """
    Performs a 1 point cross between two vectors.
    """
    
    cross_point = random.randrange(0, int(vector1.size/15))
    return np.hstack([vector1[:int(cross_point*15)], vector2[int(cross_point*15):]])

def cross2p(vector1, vector2):
    """
    Performs a 2 point cross between two vectors.
    """
    
    cross_point1 = random.randrange(0, int(vector1.size/15)-2)
    cross_point2 = random.randrange(cross_point1, int(vector1.size/15))
    return np.hstack([vector1[:int(cross_point1*15)], vector2[int(cross_point1*15):int(cross_point2*15)], vector1[int(cross_point2*15):]])

def crossDiscrete(vector, population, n, is_sigma=False, is_1stepsize=False):
    result = np.copy(vector)
    other_parents_idx = np.random.choice(np.arange(len(population)), n-1, replace=False)
    other_parents = list(np.array(population)[other_parents_idx])
    for i in np.arange(n-1):
        if is_sigma:
            if is_1stepsize:
                result = other_parents[0]
            else:
                discretization = np.random.randint(0,n,len(vector)) - 1
                result[discretization==i] = other_parents[i][discretization==i]
        else:
            discretization = np.random.randint(0,n,len(vector)) - 1
            result[discretization==i] = other_parents[i].vector[discretization==i]
    return result


def crossInterAvg(vector, population, n, is_sigma=False):
    other_parents_idx = np.random.choice(np.arange(len(population)), n-1, replace=False)
    other_parents = list(np.array(population)[other_parents_idx])
    if is_sigma:
        parents = list(other_parents)
    else:
        parents = [parent.vector for parent in other_parents]
        parents.append(vector)
    return np.mean(parents, axis=0)
    

def crossMp(vector1, vector2):
    mask_pos = 1*(np.random.rand(vector1.size) > 0.5)
    aux = np.copy(vector1)
    aux[mask_pos==1] = vector2[mask_pos==1]
    return aux

def multiCross(vector, population, n_ind):
    result = vector
    if n_ind <= len(population):
        n_ind = len(population)-1
        mask_pos = np.random.randint(n_ind, size=vector.size)
        parents = random.sample(population, n_ind)
        aux = np.copy(vector)
        for i in range(1, n_ind-1):
            aux[mask_pos==i] = parents[i].vector[mask_pos==i]
        result = aux
    return result

def DERand1(vector, population, F, CR):
    if len(population) > 3:
        r1, r2, r3 = random.sample(population, 3)

        v = r1.vector + F*(r2.vector-r3.vector)
        mask_pos = np.random.random(vector.shape) <= CR
        vector[mask_pos] = v[mask_pos]
    return vector

def DEBest1(vector, population, F, CR):
    if len(population) > 3:
        fitness = [i.fitness for i in population]
        best = population[fitness.index(max(fitness))]
        r1, r2 = random.sample(population, 2)

        v = best.vector + F*(r1.vector-r2.vector)
        mask_pos = np.random.random(vector.shape) <= CR
        vector[mask_pos] = v[mask_pos]
    return vector

def DERand2(vector, population, F, CR):
    if len(population) > 5:
        r1, r2, r3, r4, r5 = random.sample(population, 5)

        v = r1.vector + F*(r2.vector-r3.vector) + F*(r4.vector-r5.vector)
        mask_pos = np.random.random(vector.shape) <= CR
        vector[mask_pos] = v[mask_pos]
    return vector

def DEBest2(vector, population, F, CR):
    if len(population) > 5:
        fitness = [i.fitness for i in population]
        best = population[fitness.index(max(fitness))]
        r1, r2, r3, r4 = random.sample(population, 4)

        v = best.vector + F*(r1.vector-r2.vector) + F*(r3.vector-r4.vector)
        mask_pos = np.random.random(vector.shape) <= CR
        vector[mask_pos] = v[mask_pos]
    return vector

def DECurrentToBest1(vector, population, F, CR):
    if len(population) > 3:
        fitness = [i.fitness for i in population]
        best = population[fitness.index(max(fitness))]
        r1, r2 = random.sample(population, 2)

        v = vector + F*(best.vector-vector) + F*(r1.vector-r2.vector)
        mask_pos = np.random.random(vector.shape) <= CR
        vector[mask_pos] = v[mask_pos]
    return vector

def DECurrentToRand1(vector, population, F, CR):
    if len(population) > 3:
        r1, r2, r3 = random.sample(population, 3)

        v = vector + np.random.random()*(r1.vector-vector) + F*(r2.vector-r3.vector)
        mask_pos = np.random.random(vector.shape) <= CR
        vector[mask_pos] = v[mask_pos]
    return vector

def DECurrentToPBest1(vector, population, F, CR, P):
    if len(population) > 3:
        fitness = [i.fitness for i in population]
        upper_idx = max(1, math.ceil(len(population)*P))
        pbest_idx = random.choice(np.argsort(fitness)[:upper_idx])
        pbest = population[pbest_idx]
        r1, r2 = random.sample(population, 2)

        v = vector + F*(pbest.vector-vector) + F*(r1.vector-r2.vector)
        mask_pos = np.random.random(vector.shape) <= CR
        vector[mask_pos] = v[mask_pos]
    return vector