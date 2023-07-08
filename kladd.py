import random
import numpy as np

# Define the number of cities and the population size
num_cities = 20
pop_size = 100

# Generate a list of cities with their coordinates
cities = []
for i in range(num_cities):
    x = random.uniform(0, 100)
    y = random.uniform(0, 100)
    cities.append((x, y))



# Define the fitness function that calculates the total distance
def fitness(individual, cities):
    total_distance = 0
    for i in range(len(individual)):
        j = (i + 1) % len(individual)
        city_i = individual[i]
        city_j = individual[j]
        distance = np.sqrt((cities[city_i][0] - cities[city_j][0]) ** 2 +
                           (cities[city_i][1] - cities[city_j][1]) ** 2)
        print(distance)
        total_distance += distance
    return 1 / total_distance

def swap_mutation(individual, mutation_rate):
    for swapped in range(len(individual)):
        if random.random() < mutation_rate:
            swap_with = int(random.random() * len(individual))

            city1 = individual[swapped]
            city2 = individual[swap_with]

            individual[swapped] = city2
            individual[swap_with] = city1

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

# Define the genetic algorithm parameters
elite_size = 20
mutation_rate = 0.01
generations = 50

# Define the genetic algorithm function
def run_ga_tsp(cities, pop_size, elite_size, mutation_rate, generations):
    # Create the initial population
    population = []
    for i in range(pop_size):
        individual = list(range(len(cities)))
        random.shuffle(individual)
        population.append(individual)

    # Evaluate the fitness of each individual
    fitness_scores = [fitness(individual, cities) for individual in population]

    # Keep track of the best individual and its fitness score
    best_fitness = max(fitness_scores)
    best_index = fitness_scores.index(best_fitness)
    best_route = population[best_index]

    # Iterate through generations
    for generation in range(generations):
        # Select the parents for crossover
        elite_individuals = [population[i] for i in np.argsort(fitness_scores)[-elite_size:]]
        mating_pool = elite_individuals.copy()
        while len(mating_pool) < pop_size:
            parent1 = random.choice(elite_individuals)
            parent2 = random.choice(elite_individuals)
            child = ordered_crossover(parent1, parent2)
            if random.random() < mutation_rate:
                child = swap_mutation(child, mutation_rate)
            mating_pool.append(child)

        # Create the new population
        population = mating_pool

        # Evaluate the fitness of each individual
        fitness_scores = [fitness(individual, cities) for individual in population]

        # Keep track of the best individual and its fitness score
        if max(fitness_scores) > best_fitness:
            best_fitness = max(fitness_scores)
            best_index = fitness_scores.index(best_fitness)
            best_route = population[best_index]

    return best_fitness, best_route

def main():
    print(run_ga_tsp(cities, pop_size, elite_size, mutation_rate, generations))
main()
