import sys
sys.path.append("..")

from A3_ComputacionEvolutiva import *
from ParamScheduler import *
from actividad2Funcs import *
from actividad3Funcs import *

import argparse
import numpy as np
import pandas as pd

def comparar_funciones(list_min_aprox, list_best, objfunc, name):
    idx = np.argsort(list_best).astype(np.int32)
    best = list_min_aprox[idx[0]]
    best_decoded = decodingGE(best, objfunc.mod_list, objfunc.wrapping)
    np.savetxt(f'./best-sol{name}.txt', best_decoded, fmt="%s", delimiter=',')
    
    fig = plt.figure()

    x = objfunc.x

    plt.plot(objfunc.target_function(x), label="Original")
    f_hat = np.array([kernel[1]*eval(kernel[0])(kernel[2], kernel[3], kernel[4], x) for kernel in best_decoded])
    plt.plot(f_hat.sum(axis=0), label="Aproximated")
    plt.ylabel("Funcion")
    plt.xlabel("X")
    plt.title("Comparativa aproximación")
    plt.legend()
    plt.ticklabel_format(axis='y', style="sci", scilimits=None)
    plt.savefig(f'./figures/best-sol{name}.png')
    

def generar_curva_progreso(evolucion_fitness, array_bars, pasos_intervalos, ngen, file_name):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    x = np.arange(ngen)

    plt.errorbar(x, evolucion_fitness, yerr=array_bars, errorevery=pasos_intervalos, capsize=3.0, ecolor='black')
    plt.ylabel("Fitness")
    plt.xlabel("Generaciones")
    plt.title("Curva de progreso")
    plt.ticklabel_format(axis='y', style="sci", scilimits=None)
    ax.set_yscale('log')
    plt.savefig(file_name)

    fig = plt.figure()

    x = np.arange(ngen)

    plt.errorbar(x, evolucion_fitness, yerr=array_bars, errorevery=pasos_intervalos, capsize=3.0, ecolor='black')
    plt.ylabel("Fitness")
    plt.xlabel("Generaciones")
    plt.title("Curva de progreso")
    plt.ticklabel_format(axis='y', style="sci", scilimits=None)
    plt.savefig(file_name[:10]+'no-log-'+file_name[10:])
    #plt.show()

def get_pex_for_1(history, counter, success):
    pex = -1
    success_idx = np.where(history < success)
    if np.any(success_idx):
        pex = int(counter * success_idx[0][0]/len(history))
    return pex 

def get_pex(list_history, list_counter, success):
    list_pex = np.array([get_pex_for_1(history, list_counter[idx], success) for idx, history in enumerate(list_history)])
    return np.mean(list_pex[list_pex != -1])

def get_te(list_history, success):
    amount_succes = (np.array(list_history)[:,-1] < success).astype(int)
    return amount_succes.mean()

def get_vamm(list_best):
    return np.mean(list_best)
    

def run_algorithm(alg_name, what_to_compare):
    params = {
        # Population-based
        "popSize": 70,

        # Genetic algorithm
        "pmut": 0.5,
        "pcross":0.9,

        # Evolution strategy
        "offspringSize":400,
        "sigma_type":"nstepsize",
        "epsilon":1e-25,
        #"tau":1/np.sqrt(10),

        # General
        "stop_cond": "ngen",
        "time_limit": 20.0,
        "Ngen": 1500,
        "Neval": 1e5,
        "fit_target": 0,

        "verbose": False,
        "v_timer": 0.5,
        "interval_steps":20,

        # Metrics
        "success":0.6,

        # Problem
        "max_kernels":5,
        "wrapping":2
    }

    mutation_operators = [
        OperatorInt("Gauss", {"F":1}),
        OperatorInt("Gauss", {"F":2}),
        #OperatorInt("Gauss", {"F":3}),
        OperatorInt("Gauss", {"F":5}),
        #OperatorInt("Gauss", {"F":10}),
        OperatorInt("Cauchy", {"F":1}),
        OperatorInt("Cauchy", {"F":2}),
        #OperatorInt("Cauchy", {"F":3}),
        OperatorInt("Cauchy", {"F":5}),
        #OperatorInt("Cauchy", {"F":10}),
        OperatorInt("Poisson", {"F":1}),
        OperatorInt("Poisson", {"F":2}),
        #OperatorInt("Poisson", {"F":3}),
        OperatorInt("Poisson", {"F":5}),
        #OperatorInt("Poisson", {"F":10}),
    ]

    cross_operators = [
        #OperatorInt("2point"),
        OperatorInt("Multipoint"),
        #OperatorInt("Multicross"),
        OperatorInt("CrossInterAvg", {"N":5}),
        OperatorInt("1point"),
    ]

    selection_operators = [
        ParentSelection("Tournament", {"amount":params["popSize"], "p" : 0.1}),
        #ParentSelection("Tournament", {"amount":params["popSize"], "p" : 0.1}),
        ParentSelection("Tournament", {"amount":params["popSize"], "p" : 0.5}),
        ParentSelection("Tournament", {"amount":params["popSize"], "p" : 0.9}),
        ParentSelection("Nothing"),
    ]

    replace_operators = [
        SurvivorSelection("Elitism", {"amount":1}),
        SurvivorSelection("Elitism", {"amount":5}),
        #SurvivorSelection("Elitism", {"amount":7}),
        SurvivorSelection("Elitism", {"amount":10}),
        SurvivorSelection("Generational"),
        SurvivorSelection("One-to-one"),
    ]

    operators_to_compare = {
        #"mutation_operators":mutation_operators,
        "cross_operators":cross_operators,
        "selection_operators":selection_operators,
        "replace_operators":replace_operators
    }

    params_in_objfunc = {
        "max_kernels" : [2, 5, 6, 10, 20],
        "wrapping" : [0, 1, 2, 3],
    }

    params_to_compare = {
        "popSize" : [10, 50, 70, 100, 150],
        "pmut" : [0.1, 0.2, 0.5, 0.8, 0.9],
        "pcross" : [0.1, 0.3, 0.6, 0.9, 0.99],
    }

    # coeffs = np.random.random_integers(10, size=4*15)
    # print(coeffs)
    mod_list = np.array([6, 2, 9, 10, 2, 10, 9, 10, 2, 10, 9, 10, 2, 10, 5], dtype=np.int32)
    print(mod_list)
    
    prob1_fun = (lambda x : 8 * np.exp(-2 * (x - 2)**2) + 2*x + 1 + 3 * np.tanh(3 * x + 2))
    prob2_fun = (lambda x : 2 * np.exp(-2 * (x - 1)**2) - np.exp(-(x - 1)**2))
    prob3_fun = (lambda x : np.sqrt(x))
    prob4_fun = (lambda x : np.exp(-x) * np.sin(2*x))

    list_target_names = ["prob1_fun", "prob2_fun", "prob3_fun", "prob4_fun"]

    list_objfunc = [
        MinApproxFun(params["max_kernels"]*15, 61, prob1_fun, mod_list, lim_min=-2, lim_max=4, wrapping=params["wrapping"]),
        MinApproxFun(params["max_kernels"]*15, 41, prob2_fun, mod_list, lim_min=-1, lim_max=3, wrapping=params["wrapping"]),
        MinApproxFun(params["max_kernels"]*15, 41, prob3_fun, mod_list, lim_min=0, lim_max=4, wrapping=params["wrapping"]),
        MinApproxFun(params["max_kernels"]*15, 41, prob4_fun, mod_list, lim_min=0, lim_max=4, wrapping=params["wrapping"])
    ]

    mutation_op = OperatorInt("Gauss", {"F": 1})
    cross_op = OperatorInt("Multipoint")
    parent_select_op = ParentSelection("Tournament", {"amount":params["popSize"], "p" : 0.5})
    replace_op = SurvivorSelection("One-to-one")

    if what_to_compare=='p':
        for idx, objfunc in enumerate(list_objfunc):
            for key in params_to_compare.keys():
                metrics_list = []
                for value in params_to_compare[key]:
                    params[key]=value
                    list_history = []
                    list_best = []
                    list_counter = []
                    for i in range(5):
                        objfunc.counter = 0
                        if alg_name == "ES":
                            alg = ES(objfunc, mutation_op, cross_op, parent_select_op, replace_op, params)
                        elif alg_name == "DE":
                            alg = DE(objfunc, OperatorReal("DE/current-to-best/1", {"F":0.8, "Cr":0.9}), SurvivorSelection("One-to-one"), params)
                        elif alg_name == "GA":
                            alg = Genetic(objfunc, mutation_op, cross_op, parent_select_op, replace_op, params)
                        else:
                            print(f"Error: Algorithm \"{alg_name}\" doesn't exist.")
                            exit()
                            
                        ind, fit = alg.optimize()
                        list_history.append(alg.history)
                        list_best.append(alg.best_solution()[1])
                        alg.display_report(show_plots=False)
                        list_counter.append(alg.objfunc.counter)
                    list_history = np.array(list_history)
                    list_counter = np.array(list_counter)
                    pex = get_pex(list_history, list_counter, params["success"])
                    vamm = get_vamm(list_best)
                    te = get_te(list_history, params["success"])
                    metrics_list.append([pex, vamm, te])
                    mean_history = list_history.mean(axis=0)
                    std_history = list_history.std(axis=0)
                    generar_curva_progreso(mean_history, std_history, params["interval_steps"], params["Ngen"], f'./figures/studing-{key}{value}-function{objfunc.name}-{list_target_names[idx]}-pex{pex}-vamm{vamm}-te{te}.png')
                metrics_df = pd.DataFrame(np.array(metrics_list), index=[params_to_compare[key]], columns=['pex','vamm','te'])
                metrics_df.to_csv(f'./comparison-csv/metrics-objfunc{objfunc.name}-{list_target_names[idx]}-{key}.csv')
    elif what_to_compare=='i':
        for idx, objfunc in enumerate(list_objfunc):
            for key in params_in_objfunc.keys():
                metrics_list = []
                for value in params_in_objfunc[key]:
                    setattr(objfunc, key, value)
                    list_history = []
                    list_best = []
                    list_counter = []
                    for i in range(5):
                        objfunc.counter = 0
                        if alg_name == "ES":
                            alg = ES(objfunc, mutation_op, cross_op, parent_select_op, replace_op, params)
                        elif alg_name == "DE":
                            alg = DE(objfunc, OperatorReal("DE/current-to-best/1", {"F":0.8, "Cr":0.9}), SurvivorSelection("One-to-one"), params)
                        elif alg_name == "GA":
                            alg = Genetic(objfunc, mutation_op, cross_op, parent_select_op, replace_op, params)
                        else:
                            print(f"Error: Algorithm \"{alg_name}\" doesn't exist.")
                            exit()
                            
                        ind, fit = alg.optimize()
                        list_history.append(alg.history)
                        list_best.append(alg.best_solution()[1])
                        alg.display_report(show_plots=False)
                        list_counter.append(alg.objfunc.counter)
                    list_history = np.array(list_history)
                    list_counter = np.array(list_counter)
                    pex = get_pex(list_history, list_counter, params["success"])
                    vamm = get_vamm(list_best)
                    te = get_te(list_history, params["success"])
                    metrics_list.append([pex, vamm, te])
                    mean_history = list_history.mean(axis=0)
                    std_history = list_history.std(axis=0)
                    generar_curva_progreso(mean_history, std_history, params["interval_steps"], params["Ngen"], f'./figures/studing-{key}{value}-function{objfunc.name}-{list_target_names[idx]}-pex{pex}-vamm{vamm}-te{te}.png')
                metrics_df = pd.DataFrame(np.array(metrics_list), index=[params_in_objfunc[key]], columns=['pex','vamm','te'])
                metrics_df.to_csv(f'./comparison-csv/metrics-objfunc{objfunc.name}-{list_target_names[idx]}-{key}.csv')
    elif what_to_compare=='b':
        metrics_list = []
        for idx, objfunc in enumerate(list_objfunc):
            list_history = []
            list_best = []
            list_counter = []
            list_min_aprox = []
            for i in range(5):
                objfunc.counter = 0
                if alg_name == "ES":
                    alg = ES(objfunc, mutation_op, cross_op, parent_select_op, replace_op, params)
                elif alg_name == "DE":
                    alg = DE(objfunc, OperatorReal("DE/current-to-best/1", {"F":0.8, "Cr":0.9}), SurvivorSelection("One-to-one"), params)
                elif alg_name == "GA":
                    alg = Genetic(objfunc, mutation_op, cross_op, parent_select_op, replace_op, params)
                else:
                    print(f"Error: Algorithm \"{alg_name}\" doesn't exist.")
                    exit()

                ind, fit = alg.optimize()
                list_history.append(alg.history)
                list_best.append(alg.best_solution()[1])
                list_min_aprox.append(alg.best_solution()[0])
                alg.display_report(show_plots=False)
                list_counter.append(alg.objfunc.counter)
            list_history = np.array(list_history)
            list_counter = np.array(list_counter)
            pex = get_pex(list_history, list_counter, params["success"])
            vamm = get_vamm(list_best)
            te = get_te(list_history, params["success"])
            comparar_funciones(list_min_aprox, list_best, objfunc, list_target_names[idx])
            metrics_list.append([pex, vamm, te])
            mean_history = list_history.mean(axis=0)
            std_history = list_history.std(axis=0)
            generar_curva_progreso(mean_history, std_history, params["interval_steps"], params["Ngen"], f'./figures/best-params-function{objfunc.name}-{list_target_names[idx]}-pex{pex}-vamm{vamm}-te{te}.png')
        metrics_df = pd.DataFrame(np.array(metrics_list), index=[list_target_names], columns=['pex','vamm','te'])
        metrics_df.to_csv(f'./comparison-csv/metrics-best-params-objfunc{objfunc.name}-{list_target_names}.csv')
    else:
        for idx, objfunc in enumerate(list_objfunc):
            for key in operators_to_compare.keys():
                metrics_list = []
                for value in operators_to_compare[key]:
                    if key=="mutation_operators":
                        mutation_op = value
                    elif key=="cross_operators":
                        cross_op = value
                    elif key=="selection_operators":
                        parent_select_op = value
                    elif key=="replace_operators":
                        replace_op = value
                    list_history = []
                    list_best = []
                    list_counter = []
                    for i in range(5):
                        objfunc.counter = 0
                        if alg_name == "ES":
                            alg = ES(objfunc, mutation_op, cross_op, parent_select_op, replace_op, params)
                        elif alg_name == "DE":
                            alg = DE(objfunc, OperatorReal("DE/current-to-best/1", {"F":0.8, "Cr":0.9}), SurvivorSelection("One-to-one"), params)
                        elif alg_name == "GA":
                            alg = Genetic(objfunc, mutation_op, cross_op, parent_select_op, replace_op, params)
                        else:
                            print(f"Error: Algorithm \"{alg_name}\" doesn't exist.")
                            exit()
                            
                        ind, fit = alg.optimize()
                        list_history.append(alg.history)
                        list_best.append(alg.best_solution()[1])
                        alg.display_report(show_plots=False)
                        list_counter.append(alg.objfunc.counter)
                    list_history = np.array(list_history)
                    list_counter = np.array(list_counter)
                    pex = get_pex(list_history, list_counter, params["success"])
                    vamm = get_vamm(list_best)
                    te = get_te(list_history, params["success"])
                    metrics_list.append([pex, vamm, te])
                    mean_history = list_history.mean(axis=0)
                    std_history = list_history.std(axis=0)
                    generar_curva_progreso(mean_history, std_history, params["interval_steps"], params["Ngen"], f'./figures/studing-{key}{value.name}-function{objfunc.name}-{list_target_names[idx]}-pex{pex}-vamm{vamm}-te{te}.png')
                metrics_df = pd.DataFrame(np.array(metrics_list), index=[operators_to_compare[key]], columns=['pex','vamm','te'])
                metrics_df.to_csv(f'./comparison-csv/metrics-objfunc{objfunc.name}-{list_target_names[idx]}-{key}.csv')


def main(what_to_compare='p'):
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algorithm", dest='alg', help='Specify an algorithm')
    args = parser.parse_args()

    algorithm_name = "GA"
    if args.alg:
        algorithm_name = args.alg
   
    run_algorithm(alg_name = algorithm_name, what_to_compare = what_to_compare)

if __name__ == "__main__":
    main('b')