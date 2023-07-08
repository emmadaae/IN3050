#!/usr/bin/env python3

import csv
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import pandas
import time
import itertools

import statistics

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

def hill_climbing_outline(objective_function, initial_solution, neighbor_generator, max_iterations):
    current_solution = initial_solution
    current_value = objective_function(current_solution)

    for i in range(max_iterations):
        neighbor = neighbor_generator(current_solution)
        neighbor_value = objective_function(neighbor)

        if neighbor_value > current_value:
            current_solution = neighbor
            current_value = neighbor_value
        else:
            break

    return current_solution

def hill_climbing(initial_solution, max_iterations):
    current_best = initial_solution #sett en random rute for første
    current_value = calculate_dictance(current_best)

    x= 0
    for i in range(max_iterations):
        x += 1
        neighbor = neighbor_generator(current_best)
        neighbor_value = calculate_dictance(neighbor)
        #print(x)
        if neighbor_value < current_value:
            current_best = neighbor
            current_value = neighbor_value
        """else:
            break"""
    return current_best, current_value

def neighbor_generator(list):
    return random.sample(list, len(list))


def main():
    #print("test")
    plan = list(city_coords.keys())
    #plan = plan[:10]
    """print(plan)
    new = neighbor_generator(plan)
    print(new)"""
    max_iterations = 10000
    #print(hill_climbing(plan, max_iterations))

    results = []
    x = 0
    for i in range(20):
        best_solution = hill_climbing(plan, max_iterations)
        #x+= 1
        #print("kjører for gang", x, "av 20")
        results.append(best_solution)

    results_values = [distance for plans, distance in results]
    results_values.sort()
    mean_index = int(len(results)/2)
    mean_value = results_values[mean_index]

    best = min(results, key=lambda tup: tup[1])
    worst = max(results, key=lambda tup: tup[1])
    mean = [i for i in results if i[1]==mean_value]
    mean = mean[0]
    #print(mean)
    standard_deviation = statistics.stdev(results_values)
    print("best: ", best, "\n")
    print("worst: ", worst, "\n")
    print( "mean: ", mean, "\n")
    print("standard deviation: ", standard_deviation)

    #plot_plan(best[0])

start = time.process_time()
main()
end = time.process_time()
seconds = end - start
print("tid brukt:", seconds, "sekunder")
