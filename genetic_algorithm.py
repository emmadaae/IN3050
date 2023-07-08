import csv
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import pandas
import time
import itertools
import statistics
import random

from exhaustive_search import plot_plan, get_all_permutations, get_distance, calculate_dictance

#Map of Europe
europe_map =plt.imread('map.png')

#Lists of city coordinates
city_coords={"Barcelona":[2.154007, 41.390205], "Belgrade": [20.46,44.79], "Berlin": [13.40,52.52], "Brussels":[4.35,50.85],"Bucharest":[26.10,44.44], "Budapest": [19.04,47.50], "Copenhagen":[12.57,55.68], "Dublin":[-6.27,53.35], "Hamburg": [9.99, 53.55], "Istanbul": [28.98, 41.02], "Kiev": [30.52,50.45], "London": [-0.12,51.51], "Madrid": [-3.70,40.42], "Milan":[9.19,45.46], "Moscow": [37.62,55.75], "Munich": [11.58,48.14], "Paris":[2.35,48.86], "Prague":[14.42,50.07], "Rome": [12.50,41.90], "Saint Petersburg": [30.31,59.94], "Sofia":[23.32,42.70], "Stockholm": [18.06,60.33],"Vienna":[16.36,48.21],"Warsaw":[21.02,52.24]}

with open("european_cities.csv", "r") as f:
    data = list(csv.reader(f, delimiter=';'))
    cities = data[0]
#print(data)
fig, ax = plt.subplots(figsize=(10,10))

ax.imshow(europe_map, extent=[-14.56,38.43, 37.697 +0.3 , 64.344 +2.0], aspect = "auto")

# Map (long, lat) to (x, y) for plotting
for city,location in city_coords.items():
    x, y = (location[0], location[1])
    plt.plot(x, y, 'ok', markersize=5)
    plt.text(x, y, city, fontsize=12);

import random

# 1. Initialization: Create an initial population of potential solutions randomly.
# Each potential solution is represented as a set of genes or parameters, and the
# population typically consists of a large number of individuals.

def create_population(size, plan):
    population = []
    for i in range(size):
        new_individual = random.sample(plan, len(plan))
        population.append(new_individual)
    return population

#2. Evaluation: Evaluate the fitness or objective function value of each individual
# in the population. The fitness function measures how well each individual solves
# the problem at hand.
def get_fitness(individual):
    return -calculate_dictance(individual)

# 3. Selection: Select a subset of the individuals from the population to be parents
# for the next generation. The selection process is typically based on the fitness
# function, with better-fit individuals having a higher probability of being selected.
def selection(population):
    fitnesses = [get_fitness(individual) for individual in population]
    max_fitness = max(fitnesses)
    selected_index = fitnesses.index(max_fitness)
    return population[selected_index]

# 4. Crossover: Create new offspring individuals by combining the genes of two
# parent individuals using a crossover operator. The crossover operator typically
# involves randomly selecting a crossover point in the parent chromosomes and
# exchanging the genetic material on either side of that point.
def total_fitness(population):
    total_fitness = 0
    for i in range(len(population)):
        total_fitness += get_fitness(population[i])
    return total_fitness

def parent_selection(population):
    selected = None
    i = random.randint(0, len(population) - 1)
    while selected == None:
        if i == len(population):
            i = 0

        #Selecting parents based on fitness
        rand = random.uniform(0, 1)
        chance = get_fitness(population[i]) / total_fitness(population)
        if(rand < chance):
            selected = population[i]
        i += 1

    return selected

#def crossover(parent_1, parent_2):

def crossover(parent_1, parent_2):
    # Select a random subset of cities from parent 1
    # Choose two random indices to slice the parents
    slice_1, slice_2 = sorted(random.sample(range(len(parent_1)), 2))

    # Initialize offspring as copies of parents
    offspring_1 = parent_1.copy()
    offspring_2 = parent_2.copy()

    # Loop over the slice and swap elements
    for i in range(slice_1, slice_2):
        # Find the corresponding city in the other parent
        corresponding_city = parent_1[i]
        while corresponding_city in offspring_2[slice_1:slice_2]:
            corresponding_city = parent_1[parent_2.index(corresponding_city)]

        # Swap the cities in the offspring
        offspring_1[i], offspring_2[parent_2.index(corresponding_city)] = corresponding_city, offspring_1[i]

    return offspring_1, offspring_2

# 5. Mutation: Introduce small random changes to the genes of the offspring individuals
# using a mutation operator. The mutation operator helps to introduce new genetic
# material into the population and prevent the algorithm from getting stuck in
# local optima.
def mutation(individual):
    # Select two random cities to swap
    swap_indices = random.sample(range(len(individual)), 2)
    # Swap the positions of the selected cities
    individual[swap_indices[0]], individual[swap_indices[1]] = individual[swap_indices[1]], individual[swap_indices[0]]
    return individual
# 6. Replacement: Replace the least fit individuals in the current population with
# the new offspring individuals to form the next generation.
# 7. Termination: Repeat steps 2 to 6 for a fixed number of generations or until a
# termination condition is met (e.g. a satisfactory solution is found, or the
# algorithm reaches a maximum number of iterations).
"""def genetic_algorithm(num_of_generations, population_size, plan):
    population = create_population(population_size, plan)
    #print(population)

    for i in range(num_of_generations):
        fitnesses = [get_fitness(individual) for individual in population]

        best_individual = selection(population)

        new_population = [best_individual]
        for j in range(population_size-1):
            parent_1 = random.choice(population)
            parent_2 = random.choice(population)
            child_1,child_2 = crossover(parent_1, parent_2)
            new_population.append(child_1)
            new_population.append(child_2)

        population = new_population

        print("genereation", i+1, "Best fitness: ", max(fitnesses))"""


def main():
    plan = list(city_coords.keys())
    plan = plan[:7]

    #print(genetic_algorithm(1, 10, plan))
    population_size = 10
    num_of_generations = 3

    population = create_population(population_size, plan)
    #print(population)

    for i in range(num_of_generations):
        fitnesses = [get_fitness(individual) for individual in population]

        best_individual = selection(population)

        new_population = [best_individual]
        for j in range(population_size-1):
            parent_1 = parent_selection(population)
            parent_2 = parent_selection(population)
            child_1,child_2 = crossover(parent_1, parent_2)
            new_population.append(child_1)
            new_population.append(child_2)

        population = new_population

        print("genereation", i+1, "Best fitness: ", max(fitnesses))



main()
