# Assignment1 IN3050
# Emma Daae - emmadaa@ifi.uio.no

# kommentarer

# prekode fra jupityr notebook-fil under:

#%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import pandas
#Map of Europe
europe_map =plt.imread('map.png')

#Lists of city coordinates
city_coords={"Barcelona":[2.154007, 41.390205], "Belgrade": [20.46,44.79], "Berlin": [13.40,52.52], "Brussels":[4.35,50.85],"Bucharest":[26.10,44.44], "Budapest": [19.04,47.50], "Copenhagen":[12.57,55.68], "Dublin":[-6.27,53.35], "Hamburg": [9.99, 53.55], "Istanbul": [28.98, 41.02], "Kiev": [30.52,50.45], "London": [-0.12,51.51], "Madrid": [-3.70,40.42], "Milan":[9.19,45.46], "Moscow": [37.62,55.75], "Munich": [11.58,48.14], "Paris":[2.35,48.86], "Prague":[14.42,50.07], "Rome": [12.50,41.90], "Saint Petersburg": [30.31,59.94], "Sofia":[23.32,42.70], "Stockholm": [18.06,60.33],"Vienna":[16.36,48.21],"Warsaw":[21.02,52.24]}

#Helper code for plotting plans
#First, visualizing the cities.
import csv
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

# metode for å plotte inn planen slik at den kan visualisseres
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
"""
eksempel på bruk av kode for visualisering
"""
"""#Example usage of the plotting-method.
plan = list(city_coords.keys()) # Gives us the cities in alphabetic order
print(plan)
plot_plan(plan)"""

"""
første steg blir å definere en metode som kan finne avstanden for de
forskjellige søkene, slik at vi kan fastslå hvilken sti som vil være kortest
eller "raskest".

jeg måtte google litt for å finne formelen for avstand mellom koordinater, og
er den jeg bruker i metoden min under. (antar litt at jeg ikke må skrive noe
kilde for denne da det kun er et enekelt googlesøk)
"""
#from math import radians, cos, sin, asin, sqrt

#data_dist = pandas.read_csv("european_cities.csv")
#print(data_dist)

#print(data_dist(1))
#print(data_dist.iloc[1])
#print(data)
#print(cities)

def get_distance(city_1, city_2):
    if city_1 and city_2 in cities:
        city1_index = cities.index(city_1)
        #print(city1_index)
        city2_index = cities.index(city_2)
    row = data[city1_index +1]
    distance = row[city2_index]
    return float(distance)

def calculate_dictance(plan):
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

print(get_distance('Barcelona', 'Brussels'))

"""plan = list(city_coords.keys()) # Gives us the cities in alphabetic order
#print(plan)
plan = plan[:4]
print(plan)
print(calculate_dictance(plan))"""


"""def dist_between(lat1, long1, lat2, long2):
    # konverterer fra grader til radianer
    longtitude1 = radians(long1)
    longtitude2 = radians(long2)
    latitude1 = radians(lat1)
    latitude2 = radians(lat2)

    # bruker haversinformelen for å beregne sirkelavstand mellom to punkter
    dlon = longtitude2 - longtitude1
    dlat = latitude2 - latitude1
    a = sin(dlat/2)**2 + cos(latitude1)*cos(latitude2)*sin(dlon/2)**2

    c = 2*asin(sqrt(a))

    r = 6371

    return (c*r)"""


"""def calculate_dictance(plan):
    distance = 0
    for i in range(len(plan)-1):
        current_city_coords = city_coords[plan[i]]
        next_city_coords = city_coords[plan[i+1]]
        distance += dist_between(current_city_coords[0], current_city_coords[1], next_city_coords[0], next_city_coords[1])
    return distance"""





"""
Starter med exhautive search

"""
import itertools

def get_all_permutations(list):
    perms = itertools.permutations(list, len(list)-1)
    return perms

def exhaustive_search(plan):
    # lagrer alle avstander i en liste
    distances = {}
    #finner alle permutasjoner av koordinatene våre
    permutations = list(get_all_permutations(plan))
    #beregner avstand for alle permuatsjoner og legger de i listen
    for i in permutations:
        distances[i] = calculate_dictance(i)
        #distances_list.append(calculate_dictance(i))
    #finner og returnerer korteste avstand og tilhørende "sti"
    shortest_dist = min(distances.values())
    shortest_path = [i for i in distances if distances[i] == shortest_dist]
    if len(shortest_path)==2:
        shortest_path = shortest_path[0]
    return shortest_path, shortest_dist

# starter med å teste for de 6 første byene
import time
"""
start = time.process_time()
plan = list(city_coords.keys())
plan = plan[:11]

result = exhaustive_search(plan)
shortest_path = result[0]
distance = result[1]
print(" ")
print("shortest path: ", shortest_path)
print("distance: ", distance)
#plot_plan(shortest_path)
end = time.process_time()
seconds = end - start
print("time used: ", seconds, "Seconds")"""



def hill_climbing(plan):
    #ikke lagre i en liste?
    permutations = list(get_all_permutations(plan))
    index = 0
    current = (permutations[index], calculate_dictance(permutations[index]))
    while index < len(permutations)-1:
        next = (permutations[index +1], calculate_dictance(permutations[index+1]))
        if next[1] < current[1]:
            current = next
        index += 1
    return current

def hill_climbing_1(plan):
    permutations = itertools.permutations(plan, len(plan)-1)
    current_item = next(permutations)
    current_best = next(permutations)
    for next_item in permutations:
        if calculate_dictance(next_item) < calculate_dictance(current_best):
            current_best = next_item
        current_item = next_item
    return current_best

start = time.process_time()

plan = list(city_coords.keys())
#plan = plan[:10]
#print(exhaustive_search(plan))
print(hill_climbing(plan))
"""permutations = get_all_permutations(plan)
#print(len(permutations))
for i in permutations:
    print(i)
    print(next(permutations))"""
#print(hill_climbing(plan))

end = time.process_time()
seconds = end - start
print("time used: ", seconds, "Seconds")
