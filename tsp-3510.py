import sys
import math
from collections import namedtuple
from random import shuffle


Point = namedtuple('Point', ['x', 'y'])


def main(input_file, output_file, time):
    """
    Main driver for the program, kicks of search
    """
    print("Program started ({}, {}, {})".format(input_file, output_file, time))

    nodes = read_nodes(input_file)
    print(nodes)

    cost, tour = random_tour(nodes)
    
    write_results(cost, tour)


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
        cost += euclidean_distance(points[tour[i]], points[tour[i + 1]])

    return (cost, tour)


def read_nodes(input_file):
    """
    Reads in the nodes from the input file and creates
    a list of points in memory
    """
    f = open(input_file, "r")

    nodes = []
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

