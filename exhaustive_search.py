
import csv
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import pandas
import time
import itertools

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


def plot_plan(city_order):
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(europe_map, extent=[-14.56,38.43, 37.697 +0.3 , 64.344 +2.0], aspect = "auto")

    # Map (long, lat) to (x, y) for plotting
    for index in range(len(city_order) -1):
        current_city_coords = city_coords[city_order[index]]
        next_city_coords = city_coords[city_order[index+1]]
        x, y = current_city_coords[0], current_city_coords[1]
        #Plotting a line to the next city
        next_x, next_y = next_city_coords[0], next_city_coords[1]
        plt.plot([x,next_x], [y,next_y])

        plt.plot(x, y, 'ok', markersize=5)
        plt.text(x, y, index, fontsize=12);
    #Finally, plotting from last to first city
    first_city_coords = city_coords[city_order[0]]
    first_x, first_y = first_city_coords[0], first_city_coords[1]
    plt.plot([next_x,first_x],[next_y,first_y])
    #Plotting a marker and index for the final city
    plt.plot(next_x, next_y, 'ok', markersize=5)
    plt.text(next_x, next_y, index+1, fontsize=12);
    plt.show();


def get_distance(city_1, city_2):
    """

    finner avstanden mellom to byer basert på data fra csv filen

    """
    if city_1 and city_2 in cities:
        city1_index = cities.index(city_1)
        #print(city1_index)
        city2_index = cities.index(city_2)
    row = data[city1_index +1]
    distance = float(row[city2_index])
    return distance

def calculate_dictance(plan):
    """
    regner ut summen for avstanden i en gitt rute

    """
    distance = 0
    for i in range(len(plan)-1):
        current_city = plan[i]
        next_city= plan[i+1]
        """print(current_city)
        print(next_city)
        print(get_distance(current_city, next_city))"""
        distance += get_distance(current_city, next_city)
    distance += get_distance(plan[-1], plan[0])
    return distance

def get_all_permutations(list):
    """
    tar inn en liste med byer, og returnerer alle permuatsjoner av
    den gitte listen

    """
    perms = itertools.permutations(list, len(list)-1)
    return perms


def exhaustive_search(plan):
    perms = list(get_all_permutations(plan))
    shortest = (perms[0], calculate_dictance(perms[0]))

    for i in perms:
        distance = calculate_dictance(i)
        if distance < shortest[1]:
            shortest = (i, distance)
    return shortest


def test(plan):
    print("tester exhaustive search for", len(plan)-1, "byer: ")
    resultat = exhaustive_search(plan)
    print("korteste rute:" , resultat[0])
    print("lengde på ruten: ", resultat[1])

def main():
    plan = list(city_coords.keys())
    plan = plan[:11]
    test(plan)
    #plot_plan(plan)



"""start = time.process_time()
main()
end = time.process_time()
seconds = end - start
print("tid brukt:", seconds, "sekunder")"""
