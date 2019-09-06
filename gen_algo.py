'''----------------------------------------------------------------
   Keystroke Biometric Adversary Framework Using Genetic Algorithms
----------------------------------------------------------------'''

import random
import math
import numpy as np
from scipy.spatial.distance import euclidean
import pandas

''' This is the reference threshold to verify the match score '''
reference_threshold = 0.85

''' We want 40% of individuals in every new generation to be the result of crossover operator'''
crossover_rate = 0.4

''' The probability that a new offspring is mutated'''
mutation_rate = 0.7

''' The probability that offspring will limit the mutation operation'''
mutation_resilience = 0.9

''' We use the number of generations as termination condition '''
termination = 30

''' A list to store all the individuals in the population '''
population = list()

path = "dataset/DSL-StrongPasswordData.csv"
data = pandas.read_csv(path)
subjects = data["subject"].unique()
target_subject = 's002'

target_subject_data = data.loc[data.subject == target_subject, "H.period":"H.Return"]
target_data = target_subject_data[0:200]

individual_per_population = 2000 # Population size.
num_parents_mating = 2 # Number of parents inside the mating pool.
num_feature_elements = target_data.shape[1]
population_shape = (individual_per_population, num_feature_elements)

def create_chromosome():
    chromosome = []

    for i in range(population_shape[1]):
        gene = round(random.uniform(0.0,1), 4)
        chromosome.append(gene)

    return chromosome

'''--------------------------------------
    Calculate the Fitness Function
--------------------------------------'''

def calculate_fitness():
     """
     This method calculates the fitness for each individual in the population
     """
     for individual in population:
        chromosome = individual['chromosome']
        if reference_threshold > individual['fitness_score'] > -1:
            continue

        mean = target_data.mean()
        distance = euclidean(chromosome, mean.values)
        max_distance = euclidean(mean.values, target_data.max().values)
        beta = distance / max_distance
        distance = 1 / (1 + (beta * distance))
        individual['fitness_score'] = round(distance, 4)
        individual['chromosome'] = chromosome

'''--------------------------------------
    Initialize the Population
--------------------------------------'''
def initialize_population():

    count = 0
    while count < population_shape[0]:

        chromosome = create_chromosome()
        '''each individual in the population stores the solution (chromosome) '''
        individual = dict()
        individual['chromosome'] = list(chromosome)
        individual['fitness_score'] = -1

        population.append(individual)
        count += 1

def select_parent():
    """
    This method implements the fitness as a proportionate selection operator
    """
    fitness_sum = 0

    ''' the size of the wheel equal the total sum of the fitness values for the entire population '''
    for individual in population:
        fitness_sum += individual['fitness_score']

    selection_point = random.randint(0, math.floor(fitness_sum))

    rotate = 0
    for individual in population:
        rotate += individual['fitness_score']
        if rotate >= selection_point:
            return individual

def crossover(parent_a, parent_b):
    """
     This method implements a one point crossover technique
    """

    ''' pick a random point between 0 and length of the chromosome '''
    single_point = random.randint(0, population_shape[1] - 1)

    ''' switch the head and tail of the two parents to create two new offspring '''
    offspring_a = parent_a['chromosome'][:single_point] + parent_b['chromosome'][single_point:]

    offspring_b = parent_b['chromosome'][:single_point] + parent_a['chromosome'][single_point:]

    individual_one = dict()
    individual_one['chromosome'] = offspring_a
    individual_one['fitness_score'] = -1

    individual_two = dict()
    individual_two['chromosome'] = offspring_b
    individual_two['fitness_score'] = -1

    ''' return the new offsprings'''
    return individual_one, individual_two


def mutation(individual):
    """
     This method implements the bit flip mutation
    """

    ''' generate a  uniform random value as the mutation chance '''
    mutation_chance = random.uniform(0, 1)

    ''' not all the offsprings will be mutated '''
    if mutation_chance < mutation_rate:

        chromosome = individual['chromosome']

        individual['chromosome'] = chromosome

    return individual


''' This the evolution process of the genetic algorithm '''
generation_counter = 0

initialize_population()
calculate_fitness()

while generation_counter < termination:

    generation_counter += 1

    ''' use the crossover rate to decide how many new offsprings will be generated  '''
    new_offsprings = math.floor(len(population) * crossover_rate/2)
    new_generation = list()

    ''' generate new offspring '''
    while new_offsprings > 0:

        ''' pick two parents '''
        parent_one = select_parent()
        parent_two = select_parent()

        ''' apply crossover '''
        offspring_one, offspring_two = crossover(parent_one, parent_two)

        ''' apply mutation '''
        offspring_one = mutation(offspring_one)
        offspring_two = mutation(offspring_two)

        '''store the new offsprings in a list '''
        new_generation.append(offspring_one)
        new_generation.append(offspring_two)

        new_offsprings -= 1

    ''' apply elitism and replace the weakest individuals from the current generation and add the new offsprings '''
    population = sorted(population, key=lambda k: k['fitness_score'])
    del population[0:len(new_generation)]

    ''' this is the new generation '''
    population = new_generation + population

    '''calculate the fitness for the new generation'''
    calculate_fitness()

    print("------------- Generation " + str(generation_counter) + " -------------")
    print("Best Match Score: ", population[len(population) - 1]["fitness_score"])