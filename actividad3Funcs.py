import numpy as np
import random
import matplotlib.pyplot as plt
from ObjectiveFunc import ObjectiveFunc

def kernel_gauss(c, gamma, _, x):
    return np.exp(-gamma * (c - x)**2)

def kernel_polyn(alpha, beta, d, x):
    return (alpha * x + beta)**d

def kernel_sigmoid(delta, theta, _, x):
    return np.tanh(delta * x + theta)

def real(list, signo=["+", "-"]):
    return str(list[0]+1) + "." + str(list[1]) + "e" + str(signo[list[2]]) + str(list[3])

def decode_without_wrapping(list_decoded, encoded:[list, np.ndarray], mod_list:np.ndarray, expr, signo, continue_flag):
    for elem in encoded:
        if continue_flag:
            computed_mod = elem % mod_list
            kernel_weight = eval(signo[computed_mod[1]] + real(computed_mod[2:6]))
            kernel_name = expr[computed_mod[0]]
            param1, param2, param3 = eval(real(computed_mod[6:10])), eval(real(computed_mod[10:14])), computed_mod[14]
            list_decoded.append([kernel_name, kernel_weight, param1, param2, param3])
            if computed_mod[0] % 2!=0:
                continue_flag = False
        else:
            break
    return list_decoded, continue_flag

def decodingGE(encoded_vec:[list,np.ndarray], mod_list:np.ndarray, wrapping:int=1):
    expr = ["kernel_gauss", "kernel_gauss", "kernel_polyn", "kernel_polyn", "kernel_sigmoid", "kernel_sigmoid"]
    signo = ["+", "-"]
    wrapping = wrapping
    encoded = np.reshape(encoded_vec, (int(len(encoded_vec)/15),15))
    continue_flag = True
    list_decoded = []
    while wrapping >= 0 and continue_flag:
        list_decoded, continue_flag = decode_without_wrapping(list_decoded, encoded, mod_list, expr, signo, continue_flag)
        wrapping-=1
    return list_decoded
    


class MinApproxFun(ObjectiveFunc):
    """
    Function of minimum error to approximate a target function with kernels
    """
    def __init__(self, size, m_samples, target_function, mod_list, opt="min", lim_min=-1, lim_max=1, threshold=1e-1, penalty0=1, penalty1=10):
        self.size = size
        self.lim_min = lim_min
        self.lim_max = lim_max
        self.target_function=target_function
        self.threshold = threshold
        self.penalty0 = penalty0
        self.penalty1 = penalty1
        self.mod_list = mod_list
        self.m_samples = m_samples
        self.x = np.linspace(self.lim_min, self.lim_max, self.m_samples)
        super().__init__(self.size, opt, "MinApproxFun")
    
    def objective(self, solution):
        decoded = decodingGE(solution, self.mod_list)
        f_hat = [kernel[1]*eval(kernel[0])(kernel[2], kernel[3], kernel[4], self.x) for kernel in decoded]
        f = self.target_function(self.x)
        abs_err = np.abs(f - f_hat)
        mask_penalty = (abs_err > self.threshold).astype(np.int8)
        abs_err[mask_penalty==0] = abs_err[mask_penalty==0] * self.penalty0
        abs_err[mask_penalty==1] = abs_err[mask_penalty==1] * self.penalty1
        return abs_err.sum()/self.m_samples
    
    def random_solution(self, lim_min=None, lim_max=None, different_size=None):
        if not lim_min:
            lim_min = self.lim_min
        if not lim_max:
            lim_max = self.lim_max
        if different_size:
            size = different_size
        else:
            size = self.size
        return np.random.random_integers(lim_min, lim_max, size)
            
    def check_bounds(self, solution, parent=None):
        mask = (solution < self.lim_min) | (solution > self.lim_max)
        solution[mask] = np.random.random_integers(self.lim_min,self.lim_max,len(mask[mask==True]))
        return solution
e
if __name__ == "__main__":
    mod_list = np.array([6, 2, 9, 10, 2, 10, 9, 10, 2, 10, 9, 10, 2, 10, 5])
    obj = MinApproxFun(5*15, 20, (lambda x : x**2+4*x-1), mod_list)
    codones = np.random.randint(30, size=(5*15))
    fitness = obj.objective(codones)
    print(fitness)
    # plot_functions_n_eq_2 = True
    # if plot_functions_n_eq_2:
    #     x = np.linspace(-1,1,100)
    #     obj_sum_Powell = SumPowell(2)
    #     fig = plt.figure()
    #     ax = plt.axes(projection ='3d')
    #     z = np.reshape((lambda x,y: [obj_sum_Powell.objective(np.array([x[i],y[j]])) for j in range(len(x)) for i in range(len(x))])(x,x), (len(x),len(x)))
    #     ax.plot_surface(x, x, z)
    #     plt.savefig('./figures/sum_powell_n2_sample.png')
    #     #plt.show()
    #     x = np.linspace(-10,10,100)
    #     obj_n4_XinSheYang = N4XinSheYang(2)
    #     fig = plt.figure()
    #     ax = plt.axes(projection ='3d')
    #     z = np.reshape((lambda x,y: [obj_n4_XinSheYang.objective(np.array([x[i],y[j]])) for j in range(len(x)) for i in range(len(x))])(x,x), (len(x),len(x)))
    #     ax.plot_surface(x, x, z)
    #     plt.savefig('./figures/n4_xinsheyang_n2_sample.png')
    #     #plt.show()

