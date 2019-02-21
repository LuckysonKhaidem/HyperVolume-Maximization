from geneticAlgorithm import GeneticAlgorithm
from hypervolume import Front, HyperVolume
from problem import Kursawe, DTLZ1,ZDT1
from qpso import QPSO
import copy
from matplotlib import pyplot as plt
from jmetal.component.ranking import FastNonDominatedRanking
from collections import defaultdict


r_point = 5.0
def plot(sol, prob):
    x = []
    y = []
    for ix in range(0,len(sol),prob.number_of_variables):
        front = prob.evaluate(sol[ix:ix+prob.number_of_variables])
        if front.objectives[0] <= r_point and front.objectives[1] <= r_point:
            x.append(front.objectives[0])
            y.append(front.objectives[1])
    plt.scatter(x,y)
    plt.show()

def count_dominated_solutions(fronts):
    is_dominated = defaultdict(bool)
    for i in range(len(fronts)):
        if is_dominated[i] == True:
            continue
        for j in range(len(fronts)):
            if fronts[j].objectives[0] < fronts[i].objectives[0] and fronts[j].objectives[1] < fronts[i].objectives[1]:
                is_dominated[i] = True
            elif  fronts[i].objectives[0] < fronts[j].objectives[0] and fronts[i].objectives[1] < fronts[j].objectives[1]:
                is_dominated[j] = True
        
    return len(list(filter(lambda x: x[1] == True, is_dominated.items())))
            

def main():
    number_of_variables = 3
    leader_size = 50
    gene_length = number_of_variables * leader_size
    prob = ZDT1(number_of_variables=number_of_variables)
    hv = HyperVolume(reference_point = [r_point] * prob.number_of_objectives)
    ranker = FastNonDominatedRanking()
    def objective(x, verbose = False):
        pareto_front = []
        k = 10e3
        c = 0
        for ix in range(0,gene_length,number_of_variables):
            solution = prob.evaluate(x[ix:ix + number_of_variables])
            pareto_front.append(solution)
            if not solution.objectives[0] <= r_point or not solution.objectives[1] <= r_point:
                c+= 1
        d = count_dominated_solutions(pareto_front)
        ind = -hv.compute(pareto_front) 
        if verbose == True:
            print("{} {} {}".format(ind, d, c))
        
        return ind + k*(c**2 + d**2)
        
    ga = GeneticAlgorithm(objective,1000,gene_length,leader_size,prob.lower_bound[0],prob.upper_bound[0],0.25,0.01,0.1,100)
    ga.evolve()
    objective(ga.best_individual,True)

  
    plot(ga.best_individual,prob)


    
    # qpso = QPSO(objective, 1000,gene_length, [prob.lower_bound[0]]*gene_length, [prob.upper_bound[0]]*gene_length, 0.95, 1000)
    # soln = qpso.run()
    # plot(soln, prob)

if __name__ == "__main__":
    main()
        
