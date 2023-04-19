import numpy as np
import random
import matplotlib.pyplot as plt
from ObjectiveFunc import ObjectiveFunc


class SumPowell(ObjectiveFunc):
    """
    Sum of Powell function
    """
    def __init__(self, size, opt="min", lim_min=-1, lim_max=1):
        self.size = size
        self.lim_min = lim_min
        self.lim_max = lim_max
        super().__init__(self.size, opt, "Sum Powell")

    def objective(self, solution):
        return (np.abs(solution)**np.arange(2,solution.shape[0]+2)).sum()
    
    def random_solution(self, lim_min=None, lim_max=None, different_size=None):
        if not lim_min:
            lim_min = self.lim_min
        if not lim_max:
            lim_max = self.lim_max
        if different_size:
            size = different_size
        else:
            size = self.size
        return np.random.random(size) * (lim_max - lim_min) - lim_min
            
    
    def check_bounds(self, solution, parent=None):
        # bounce back method
        if parent:
            mask_inf = (solution < self.lim_min) 
            mask_sup = (solution > self.lim_max)
            solution[mask_inf] = parent[mask_inf] + np.random.random() * (self.lim_min - parent[mask_inf])
            solution[mask_sup] = parent[mask_sup] + np.random.random() * (parent[mask_sup] - self.lim_max)
        # random in range
        else:
            mask = (solution < self.lim_min) | (solution > self.lim_max)
            solution[mask] = np.random.random(len(mask[mask==True])) * (self.lim_max - self.lim_min) - self.lim_min
        return solution

class N4XinSheYang(ObjectiveFunc):
    """
    N4 Xin-She Yang function
    """
    def __init__(self, size, opt="min", lim_min=-10, lim_max=10):
        self.size = size
        self.lim_min = lim_min
        self.lim_max = lim_max
        super().__init__(self.size, opt, "N4 Xin-She Yang")

    def objective(self, solution):
        sum_1 = np.e ** -(solution**2).sum()
        sum_2 = np.e ** -(np.sin(np.sqrt(np.abs(solution)))**2).sum()
        return (np.sin(solution)**2 - sum_1).sum() * sum_2
    
    def random_solution(self, lim_min=None, lim_max=None):
        if not lim_min:
            lim_min = self.lim_min
        if not lim_max:
            lim_max = self.lim_max
        return np.random.random(self.size) * (lim_max - lim_min) - lim_min
    
    def check_bounds(self, solution, parent=None):
        # bounce back method
        if parent:
            mask_inf = (solution < self.lim_min) 
            mask_sup = (solution > self.lim_max)
            solution[mask_inf] = parent[mask_inf] + np.random.random() * (self.lim_min - parent[mask_inf])
            solution[mask_sup] = parent[mask_sup] + np.random.random() * (parent[mask_sup] - self.lim_max)
        # random in range
        else:
            mask = (solution < self.lim_min) | (solution > self.lim_max)
            solution[mask] = np.random.random(len(mask[mask==True])) * (self.lim_max - self.lim_min) - self.lim_min
        return solution

if __name__ == "__main__":
    plot_functions_n_eq_2 = True
    if plot_functions_n_eq_2:
        x = np.linspace(-1,1,100)
        obj_sum_Powell = SumPowell(2)
        fig = plt.figure()
        ax = plt.axes(projection ='3d')
        z = np.reshape((lambda x,y: [obj_sum_Powell.objective(np.array([x[i],y[j]])) for j in range(len(x)) for i in range(len(x))])(x,x), (len(x),len(x)))
        ax.plot_surface(x, x, z)
        plt.savefig('./figures/sum_powell_n2_sample.png')
        #plt.show()
        x = np.linspace(-10,10,100)
        obj_n4_XinSheYang = N4XinSheYang(2)
        fig = plt.figure()
        ax = plt.axes(projection ='3d')
        z = np.reshape((lambda x,y: [obj_n4_XinSheYang.objective(np.array([x[i],y[j]])) for j in range(len(x)) for i in range(len(x))])(x,x), (len(x),len(x)))
        ax.plot_surface(x, x, z)
        plt.savefig('./figures/n4_xinsheyang_n2_sample.png')
        #plt.show()

