import sys
sys.path.append("..")

from A3_ComputacionEvolutiva import *
from ParamScheduler import *
from actividad2Funcs import *
from actividad3Funcs import *

import argparse
import numpy as np
import pandas as pd

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
    

def run_algorithm(alg_name):
    params = {
        # Population-based
        "popSize": 70,

        # Genetic algorithm
        "pmut": 0.2,
        "pcross":0.9,

        # Evolution strategy
        "offspringSize":400,
        "sigma_type":"nstepsize",
        #"tau":1/np.sqrt(10),

        # General
        "stop_cond": "ngen",
        "time_limit": 20.0,
        "Ngen": 150,
        "Neval": 1e5,
        "fit_target": 1000,

        "verbose": True,
        "v_timer": 0.5,
        "interval_steps":50,

        # Metrics
        "success":1e-8
    }

    operators = [
        OperatorReal("Multipoint"),
        #OperatorReal("DE/best/1", {"F":0.7, "Cr":0.8}),
        OperatorReal("Gauss", {"F":0.001}),
        OperatorReal("Cauchy", {"F":0.005}),
    ]

    #objfunc = SumPowell(10)
    list_objfunc = [
        SumPowell(10),
        N4XinSheYang(10)
    ]

    mutation_op = OperatorReal("Gauss", {"F": 0.001})
    # mutation_op = OperatorReal("Gauss", ParamScheduler("Lineal", {"F":[0.1, 0.001]}))
    #cross_op = OperatorReal("CrossDiscrete", {"N": 5})
    cross_op = OperatorReal("CrossInterAvg", {"N": 5})
    # parent_select_op = ParentSelection("Tournament", {"amount": 3, "p":0.1})
    parent_select_op = ParentSelection("Nothing")#, ParamScheduler("Lineal", {"amount": [2, 7], "p":0.1}))
    replace_op = SurvivorSelection("(m+n)")

    list_epsilon = [
        #"1stepsize",
        #"nstepsize"
        #1e-10,
        #1e-17,
        1e-25
    ]
    #mutation_operators 
    for objfunc in list_objfunc:
        metrics_list = []
        for epsilon in list_epsilon:
            params["epsilon"]=epsilon
            list_history = []
            list_best = []
            list_counter = []
            for i in range(2):
                objfunc.counter = 0
                if alg_name == "ES":
                    alg = ES(objfunc, mutation_op, cross_op, parent_select_op, replace_op, params)
                elif alg_name == "DE":
                    alg = DE(objfunc, OperatorReal("DE/current-to-best/1", {"F":0.8, "Cr":0.9}), SurvivorSelection("One-to-one"), params)
                elif alg_name == "GA":
                    alg = Genetic(objfunc, mutation_op, OperatorReal("Multipoint"), parent_select_op, SurvivorSelection("Elitism", {"amount":5}), params)
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
            generar_curva_progreso(mean_history, std_history, params["interval_steps"], params["Ngen"], f'./figures/studing-epsilon{params["epsilon"]}-function{objfunc.name}-pex{pex}-vamm{vamm}-te{te}.png')
        metrics_df = pd.DataFrame(np.array(metrics_list), index=[list_epsilon], columns=['pex','vamm','te'])
        metrics_df.to_csv(f'./comparison-csv/metrics-objfunc{objfunc.name}-epsilon.csv')
        params["success"]=0.000165

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--algorithm", dest='alg', help='Specify an algorithm')
    args = parser.parse_args()

    algorithm_name = "GA"
    if args.alg:
        algorithm_name = args.alg
   
    run_algorithm(alg_name = algorithm_name)

if __name__ == "__main__":
    main()