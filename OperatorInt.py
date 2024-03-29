from .Operator import Operator
from .operatorFunctions import *


class OperatorInt(Operator):
    """
    Operator class that has discrete mutation and cross methods
    """

    def __init__(self, name, params = None):
        """
        Constructor for the Operator class
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
        
        if self.name == "1point":
            result = cross1p(solution.vector.copy(), solution2.vector.copy())
        elif self.name == "2point":
            result = cross2p(solution.vector.copy(), solution2.vector.copy())
        elif self.name == "Multipoint":
            result = crossMp(solution.vector.copy(), solution2.vector.copy())
        elif self.name == "WeightedAvg":
            result = weightedAverage(solution.vector.copy(), solution2.vector.copy(), self.params["F"])
        elif self.name == "BLXalpha":
            result = blxalpha(solution.vector.copy(), solution2.vector.copy(), self.params["Cr"])
        elif self.name == "Multicross":
            result = multiCross(solution.vector.copy(), others, self.params["N"])
        elif self.name == "CrossInterAvg":
            result = crossInterAvg(solution.vector.copy(), others, self.params["N"])
        elif self.name == "Perm":
            result = permutation(solution.vector.copy(), self.params["N"])
        elif self.name == "Xor":
            result = xorMask(solution.vector.copy(), self.params["N"])
        elif self.name == "XorCross":
            result = xorCross(solution.vector.copy(), solution2.vector.copy())
        elif self.name == "MutRand":
            result = mutateRand(solution.vector.copy(), population, self.params)
        elif self.name == "RandNoise":
            result = randNoise(solution.vector.copy(), self.params)
        elif self.name == "RandSample":
            result = randSample(solution.vector.copy(), population, self.params)
        elif self.name == "MutSample":
            result = mutateSample(solution.vector.copy(), population, self.params)
        elif self.name == "Gauss":
            result = gaussian(solution.vector.copy(), self.params["F"])
        elif self.name == "Laplace":
            result = laplace(solution.vector.copy(), self.params["F"])
        elif self.name == "Cauchy":
            result = cauchy(solution.vector.copy(), self.params["F"])
        elif self.name == "Uniform":
            result = uniform(solution.vector.copy(), self.params["Low"], self.params["Up"])
        elif self.name == "Poisson":
            result = poisson(solution.vector.copy(), self.params["F"])
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

        return np.round(result)