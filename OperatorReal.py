from .Operator import Operator
from .operatorFunctions import *


class OperatorReal(Operator):
    """
    Operator class that has continuous mutation and cross methods
    """

    def __init__(self, name, params = None):
        """
        Constructor for the OperatorReal class
        """

        self.name = name
        super().__init__(self.name, params)
    
    def evolve(self, solution, population, objfunc, sigmas=None, pos=0, is_1stepsize=False):
        """
        Evolves a solution with a different strategy depending on the type of operator
        """
        result = None
        others = [i for i in population if i != solution]
        if len(others) > 1:
            solution2 = random.choice(others)
        else:
            solution2 = solution
        
        if self.name == "Multipoint":
            result = crossMp(solution.vector.copy(), solution2.vector.copy())
        elif self.name == "Multicross":
            result = multiCross(solution.vector.copy(), others, self.params["N"])
        elif self.name == "CrossDiscrete":
            if sigmas:
                # print('\n Sigmas in operator: ',sigmas)
                # print('\n Pos: ',pos)
                # print('\n Sigmas[pos]: ',sigmas[pos])
                result = crossDiscrete(sigmas[pos].copy(), sigmas, self.params["N"], is_sigma=True, is_1stepsize=is_1stepsize)
            else:
                result = crossDiscrete(solution.vector.copy(), others, self.params["N"])
        elif self.name == "CrossInterAvg":
            if sigmas:
                result = crossInterAvg(sigmas[pos].copy(), sigmas, self.params["N"], is_sigma=True)
            else:
                result = crossInterAvg(solution.vector.copy(), others, self.params["N"])
        elif self.name == "Perm":
            result = permutation(solution.vector.copy(), self.params["N"])
        elif self.name == "Gauss":
            if sigmas:
                result = gaussian(solution.vector.copy(), sigmas[pos].copy())
            else:
                result = gaussian(solution.vector.copy(), self.params["F"])
        elif self.name == "Uniform":
            result = uniform(solution.vector.copy(), self.params["Low"], self.params["Up"])
        elif self.name == "MutRand":
            result = mutateRand(solution.vector.copy(), population, self.params)
        elif self.name == "RandNoise":
            result = randNoise(solution.vector.copy(), self.params)
        elif self.name == "RandSample":
            result = randSample(solution.vector.copy(), population, self.params)
        elif self.name == "MutSample":
            result = mutateSample(solution.vector.copy(), population, self.params)
        elif self.name == "DE/rand/1":
            result = DERand1(solution.vector.copy(), others, self.params["F"], self.params["Cr"])
        elif self.name == "DE/best/1":
            result = DEBest1(solution.vector.copy(), others, self.params["F"], self.params["Cr"])
        elif self.name == "DE/rand/2":
            result = DERand2(solution.vector.copy(), others, self.params["F"], self.params["Cr"])
        elif self.name == "DE/best/2":
            result = DEBest2(solution.vector.copy(), others, self.params["F"], self.params["Cr"])
        elif self.name == "DE/current-to-rand/1":
            result = DECurrentToRand1(solution.vector.copy(), others, self.params["F"], self.params["Cr"])
        elif self.name == "DE/current-to-best/1":
            result = DECurrentToBest1(solution.vector.copy(), others, self.params["F"], self.params["Cr"])
        elif self.name == "DE/current-to-pbest/1":
            result = DECurrentToPBest1(solution.vector.copy(), others, self.params["F"], self.params["Cr"], self.params["P"])
        elif self.name == "Nothing":
            result = solution.vector.copy()
        elif self.name == "Custom":
            fn = self.params["function"]
            result = fn(solution, population, objfunc, self.params)
        else:
            print(f"Error: evolution method \"{self.name}\" not defined")
            exit(1)
            
        return result