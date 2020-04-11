import sys
import math
import numpy as np
from collections import namedtuple
from random import shuffle


Point = namedtuple('Point', ['x', 'y'])

nodes = []
distances = []
distances_np = None


def main(input_file, output_file, time):
    """
    Main driver for the program, kicks of search
    """
    print("Program started ({}, {}, {})".format(input_file, output_file, time))

    read_nodes(input_file)
    gen_distances(nodes)
    print(distances_np)

    distances_reduced_rows_np = distances_np - np.min(distances_np, axis=1)[:, None]
    distances_reduced_cols_np = distances_reduced_rows_np - np.min(distances_reduced_rows_np, axis=0)[None, :]
    
    print(distances_reduced_cols_np)

    min_cost = float("inf")
    min_tour = None

    for i in range(100000):
        cost, tour = random_tour(nodes)
        if cost < min_cost:
            min_cost = cost
            min_tour = tour
    
    write_results(min_cost, min_tour)


def gen_distances(nodes):
    """
    Generate the distances between all nodes
    """
    print("Generating distances between all nodes...")

    # Setup node matrix
    for i in range(len(nodes)):
        distances.append([0] * len(nodes))

    for i, n1 in enumerate(nodes):
        for j, n2 in enumerate(nodes):
            if i == j:
                distances[i][j] = sys.maxsize
            else:
                distances[i][j] = euclidean_distance(n1, n2)
    
    # Convert to to numpy array
    global distances_np
    distances_np = np.array(distances, dtype=int)

def get_distance(n1, n2):
    """
    Once distances have been generated, use this function to 
    return the distance from the cache
    """
    return distances[n1][n2]


def euclidean_distance(p1, p2):
    """
    Returns the euclidean distance between two points
    """
    return int(math.sqrt((p1.x - p2.x)**2 + (p1.x - p2.x)**2))


def random_tour(points):
    """
    Returns cost and path of a random tour
    """
    tour = [x for x in range(len(points))]
    shuffle(tour)

    tour.append(tour[0])

    cost = 0
    for i in range(len(tour) - 1):
        cost += get_distance(tour[i], tour[i + 1])

    return (cost, tour)


def read_nodes(input_file):
    """
    Reads in the nodes from the input file and creates
    a list of points in memory
    """
    f = open(input_file, "r")

    for line in f.readlines():
        _, x_str, y_str = line.split(" ")
        point = Point(float(x_str), float(y_str))
        nodes.append(point)

    return nodes


def write_results(cost, tour):
    """
    Writes the results to the output file
    """
    f = open(output_file, "w")

    f.write(str(cost))
    f.write("\n")

    for node in tour:
        f.write("{} ".format(node))

    f.write("\n")
    f.close()

    print("Results written to file")


# If main, execute!
if __name__ == "__main__":
    # Quit if invalid arguments
    if len(sys.argv) != 4:
        print("Invalid arguments!")
        quit(1)
    
    _, input_file, output_file, time = sys.argv

    main(input_file, output_file, time)

