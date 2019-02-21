import numpy as np 
import random
import copy

class GeneticAlgorithm:
    def __init__(self, 
                fitness_function,
                population_size, 
                gene_length,
                leader_size, 
                lower_bound,
                upper_bound,
                retention = 0.25,
                survival_probability = 0.01,
                mutation_probability = 0.01,
                max_generations = 1000
                ):
        self.fitness_function = fitness_function
        self.population_size = population_size
        self.gene_length = gene_length
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.retention = retention
        self.survival_probability = survival_probability
        self.mutation_probability = mutation_probability
        self.max_generations = max_generations
        self.mutation_index = 0
        self.leader_size = leader_size
        self.number_of_variables = int(gene_length/leader_size)
        
    def initialize_population(self):
        self.population = None
        self.fitness_scores = None
        self.best_individual = None
        self.best_score = None
        return np.random.uniform(self.lower_bound, 
                                self.upper_bound, 
                                size = (self.population_size, self.gene_length)) 
                                
    def calculate_fitness(self):
        self.fitness_scores = [self.fitness_function(individual) for individual in self.population]

    def selection(self):
        elitist_population = []
        elitist_indices = np.argsort(self.fitness_scores)
        number_of_elitists = int(self.retention * self.population_size)

        for i in elitist_indices[:number_of_elitists]:
            elitist_population.append(self.population[i].copy())
        
        for i in elitist_indices[number_of_elitists:]:
            if self.survival_probability >= random.random():
                elitist_population.append(self.population[i].copy())
        
        self.population = copy.copy(elitist_population)
    
        if self.best_score == None or min(self.fitness_scores) < self.best_score:
            self.best_individual = self.population[0].copy()
            self.best_score = min(self.fitness_scores)
    
    def crossover(self):
        number_of_parents = len(self.population)
        required_children_number = self.population_size - number_of_parents
        children = []
        while len(children) < required_children_number:
            male_index,female_index = np.random.randint(0, number_of_parents, size=2)
            male = self.population[male_index]
            female = self.population[female_index]
            a,b = np.sort(np.random.randint(0,self.leader_size,2))
            a *= self.number_of_variables
            b *= self.number_of_variables
            child_A = np.hstack((male[:a],female[a:b],male[b:]))
            children.append(child_A)
            if len(children) < required_children_number:
                child_B = np.hstack((female[:a],male[a:b],female[b:]))
                children.append(child_B)
        
        self.population.extend(children)
        self.population = np.array(self.population)
    
    def mutation(self):
        for individual in self.population:
            if self.mutation_probability >= random.random():
                # mutation_index = np.random.randint(0, self.gene_length)
                mi = self.mutation_index*self.number_of_variables
                individual[mi:mi + self.number_of_variables] = np.random.uniform(self.lower_bound, self.upper_bound, size = self.number_of_variables) 
                self.mutation_index = (self.mutation_index + 1) % self.leader_size
    def evolve(self):

        self.population = self.initialize_population()

        for i in range(self.max_generations):
            self.calculate_fitness()
            self.selection()
            self.crossover()
            self.mutation()
            print("GENERATION : {}, BEST FITNESS SCORE: {}".format(i,self.best_score))
        
        self.calculate_fitness()
        best_index = np.argmin(self.fitness_scores)
        self.best_individual = self.population[best_index].copy()
        self.best_score = self.fitness_scores[best_index]