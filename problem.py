from math import sqrt, exp, pow, sin,cos,pi
from hypervolume import Front
import copy

class Kursawe:
    """ Class representing problem Kursawe. """

    def __init__(self, number_of_variables=3, rf_path = None):
        self.number_of_objectives = 2
        self.number_of_variables = number_of_variables
        self.number_of_constraints = 0

        self.lower_bound = [-5.0 for _ in range(number_of_variables)]
        self.upper_bound = [5.0 for _ in range(number_of_variables)]

    def evaluate(self, variables):
        fx = [0.0 for _ in range(self.number_of_objectives)]
        for i in range(self.number_of_variables - 1):
            xi = variables[i] * variables[i]
            xj = variables[i + 1] * variables[i + 1]
            aux = -0.2 * sqrt(xi + xj)
            fx[0] += -10 * exp(aux)
            fx[1] += pow(abs(variables[i]), 0.8) + 5.0 * sin(pow(variables[i], 3.0))

        solution = Front(copy.copy(fx),copy.copy(variables))

        return solution

    def get_name(self):
        return 'Kursawe'

class DTLZ1:
    """ Problem DTLZ1. Continuous problem having a flat Pareto front

    .. note:: Unconstrained problem. The default number of variables and objectives are, respectively, 7 and 3.
    """

    def __init__(self, number_of_variables: int = 7, number_of_objectives=3, rf_path: str=None):
        """ :param number_of_variables: number of decision variables of the problem.
        :param rf_path: Path to the reference front file (if any). Default to None.
        """
        self.number_of_variables = number_of_variables
        self.number_of_objectives = number_of_objectives
        self.number_of_constraints = 0

        self.lower_bound = self.number_of_variables * [0.0]
        self.upper_bound = self.number_of_variables * [1.0]

    def evaluate(self, variables):
        k = self.number_of_variables - self.number_of_objectives + 1

        g = sum([(x - 0.5) * (x - 0.5) - cos(20.0 * pi * (x - 0.5))
                 for x in variables[self.number_of_variables - k:]])

        g = 100 * (k + g)

        objectives = [(1.0 + g) * 0.5] * self.number_of_objectives

        for i in range(self.number_of_objectives):
            for j in range(self.number_of_objectives - (i + 1)):
                objectives[i] *= variables[j]

            if i != 0:
                objectives[i] *= 1 - variables[self.number_of_objectives - (i + 1)]
        solution = Front(objectives, variables)

        return solution

    def get_name(self):
        return 'DTLZ1'

class ZDT1:

    def __init__(self, number_of_variables: int=30, rf_path: str=None):
    
        self.number_of_variables = number_of_variables
        self.number_of_objectives = 2
        self.number_of_constraints = 0

        self.lower_bound = self.number_of_variables * [0.0]
        self.upper_bound = self.number_of_variables * [1.0]

    def evaluate(self, variables):
        g = self.__eval_g(variables)
        h = self.__eval_h(variables[0], g)
        objectives  = [None, None]
        objectives[0] = variables[0]
        objectives[1] = h * g

        solution = Front(objectives, variables)

        return solution

    def __eval_g(self, variables):
        g = sum(variables) - variables[0]

        constant = 9.0 / (self.number_of_variables - 1)
        g = constant * g
        g = g + 1.0

        return g

    def __eval_h(self, f, g):
        return 1.0 - sqrt(f / g)

    def get_name(self):
        return 'ZDT1'
