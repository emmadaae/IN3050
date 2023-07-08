import csv
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import pandas
import time
import itertools

import statistics
import random
import math

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




def fitness(individual):
    #Siden vi vil ha en høyere fitness score for rutene som er
    #kortest, returnerer vi inversen av lengden på ruten.
    distance = calculate_dictance(individual)
    fitness_score = 1/distance
    return fitness_score

def generate_population(plan, population_size):
    population = []
    for i in range(population_size):
        individual = random.sample(plan, len(plan))
        population.append(individual)
    return population

def select_parents(population):
    total_fitness = sum(fitness(individual) for individual in population)

    #genererer spin for å få et random individual
    spin = random.uniform(0, 1)
    selection_prob = [fitness(individual)/total_fitness for individual in population]
    total_prob = 0
    for i, individual in enumerate(population):
        total_prob += selection_prob[i]
        if spin <= total_prob:
            return individual

def ordered_crossover(parent1, parent2):
    length = len(parent1)
    start = random.randint(0, length - 1)
    end = random.randint(start, length - 1)
    child = [-1] * length
    for i in range(start, end + 1):
        child[i] = parent1[i]
    for i in range(length):
        if parent2[i] not in child:
            for j in range(length):
                if child[j] == -1:
                    child[j] = parent2[i]
                    break
    return child

def swap_mutation(individual, mutation_rate):
    for swapped in range(len(individual)):
        if random.random() < mutation_rate:
            swap_with = int(random.random() * len(individual))

            city1 = individual[swapped]
            city2 = individual[swap_with]

            individual[swapped] = city2
            individual[swap_with] = city1

    return individual

def genetic_algorithm(plan, population_size, generations, mutation_rate, elite_size):
    population = generate_population(plan, population_size)

    fitness_scores = [fitness(individual) for individual in population]

    #holder rede på hva som er current best
    best_fitness_score = max(fitness_scores)
    best_index = fitness_scores.index(best_fitness_score)
    best_route = population[best_index]
    best_distance = calculate_dictance(best_route)

    best_fit_per_gen =[]

    for i in range(generations):
        best_individuals = [population[i] for i in np.argsort(fitness_scores)[-elite_size:]]
        new_population = best_individuals
        best_fit_per_gen.append(best_fitness_score)

        while len(new_population) < population_size:
            parent_1 = select_parents(new_population)
            parent_2 = select_parents(new_population)
            child = ordered_crossover(parent_1, parent_2)
            if random.random() < mutation_rate:
                child = swap_mutation(child, mutation_rate)
            new_population.append(child)

        population = new_population

        fitness_scores = [fitness(individual) for individual in population]

        if max(fitness_scores) > best_fitness_score:
            best_fitness_score = max(fitness_scores)
            best_index = fitness_scores.index(best_fitness_score)
            best_route = population[best_index]
            best_distance = calculate_dictance(best_route)

    avg_best_fit = sum(best_fit_per_gen)/len(best_fit_per_gen)

    return best_fitness_score, best_route, best_distance, avg_best_fit


def main():
    plan = list(city_coords.keys())
    plan = plan[:10]
    print("plan: ", plan, "\n")
    """population = generate_population(plan, 30)
    #print("population: \n")
    #print(new_pop)
    print("popu_len:", len(population), "\n")
    parent_1 = select_parents(population)
    parent_2 = select_parents(population)
    print("parents: ", parent_1, parent_2)"""

    population_size = 30
    generations = 100
    mutation_rate = 0.01
    elite_size = 10

    #print(genetic_algorithm(plan, population_size, generations, mutation_rate, elite_size))

    results = []
    for i in range(20):
        best_solution = genetic_algorithm(plan, population_size, generations, mutation_rate, elite_size)
        #x+= 1
        #print("kjører for gang", x, "av 20")
        results.append(best_solution)
    """print(results[0][3])
    avg_best_fit_gen = []
    best_fit_per_gen = [best_fit for fitness_scores, plans, distance, best_fit in results]
    print(best_fit_per_gen)

    results_values = [distance for fitness_scores, plans, distance, best_fit in results]
    results_values.sort()
    mean_index = int(len(results)/2)
    mean_value = results_values[mean_index]

    best_distance = min(item[2] for item in results)
    worst_distance = max(item[2] for item in results)
    best = [i for i in results if i[2]==best_distance]
    worst = [i for i in results if i[2]==worst_distance]
    mean = [i for i in results if i[2]==mean_value]
    #mean = mean[0]
    #print(mean)
    standard_deviation = statistics.stdev(results_values)
    print("population size: ", population_size, "\n")
    print("best: ", best, "\n")
    print("worst: ", worst, "\n")
    print( "mean: ", mean, "\n")
    print("standard deviation: ", standard_deviation, "\n")

    #best_fitnesses_generation = [fitness for fitness, plans, distance, best_fit in results]
    #print("best fitnesses for 20 genereations of size", population_size, ": ", best_fitnesses_generation)

    


#main()

plan =  ['Madrid', 'Munich', 'Prague', 'Stockholm', 'Saint Petersburg', 'Moscow', 'Kiev', 'Brussels', 'Paris', 'Dublin', 'London', 'Hamburg', 'Berlin', 'Copenhagen', 'Warsaw', 'Vienna', 'Budapest', 'Belgrade', 'Bucharest', 'Istanbul', 'Sofia', 'Rome', 'Milan', 'Barcelona']

plot_plan(plan)
