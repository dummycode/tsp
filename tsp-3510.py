import sys
import heapq
import multiprocessing
import math
import random
import numpy as np

from copy import copy, deepcopy
from collections import namedtuple
from random import shuffle
from ctypes import c_char_p
from pprint import pprint


# Types
Point = namedtuple('Point', ['x', 'y'])
Horizon = namedtuple('Horizon', ['order', 'heuristic', 'distances', 'level', 'curr_path', 'visited'])


# Constants
INF = float('inf')


# Process variables
nodes = []
"""
distances = [
    [float('inf'), 20, 30, 10, 11],
    [15, float('inf'), 16, 4, 2],
    [3, 5, float('inf'), 2, 4],
    [19, 6, 18, float('inf'), 3],
    [16, 4, 7, 16, float('inf')],
       ]
"""
distances = []

horizon = []

# Keeps track of visited nodes in a particular path
visited = None



# Global variables
global final_cost, final_tour

def main():
    """
    Main driver for the program, kicks of search
    """
    print("Program started ({}, {}, {})".format(input_file, output_file, time))

    gen_distances(nodes)
    print(distances)

    tsp(distances) 


def update_final(curr_path, curr_cost): 
    """
    Update the final path with the current path
    """
    print("Found new minimum tour!")

    # Update final tour to curr_path
    final_tour[:N + 1] = curr_path
    # Add start back to end
    final_tour[N] = curr_path[0]

    # Update final cost to curr cost
    final_cost.value = curr_cost

    print("Updated final to {}".format(curr_cost))


def row_reduce(adj):
    total = 0
    for i, row in enumerate(adj):
        min_val = float('inf')
        for col in row:
            min_val = min(col, min_val)

        # If not inf, reduce row
        if min_val != float('inf'):
            for j, col in enumerate(row):
                adj[i][j] = adj[i][j] - min_val

            total += min_val

    return total


def col_reduce(adj):
    total = 0
    for j in range(N):
        min_val = float('inf')
        for i in range(N):
            min_val = min(adj[i][j], min_val)

        # If not inf, reduce row
        if min_val != float('inf'):
            for i in range(N):
                adj[i][j] = adj[i][j] - min_val

            total += min_val

    return total


def set_inf(adj, row, col):
    """
    Set a given row and column to infinity
    """
    adj[row] = [INF] * N
    for i in range(N):
        adj[i][col] = INF


def tsp_helper(heuristic, adj, level, curr_path, visited): 
    """
    Helper function to build the path
    adj: Adjacency matrix
    level: Current level
    curr_path: Path we are building
    visited: Nodes in the path
    """
    #print("Perfoming tsp helper at {} cost {}".format(curr_path[level - 1], heuristic))
    global max_level
    if level > max_level:
        print("Exploring level {}".format(level))
        max_level = level

    # At leaf node
    if level == N: 
        curr_res = heuristic 
        if curr_res < final_cost.value: 
            update_final(curr_path, curr_res) 
        return

    # Iterate through all vertices
    for i in range(N): 
        #print(i)
        # If not already visited
        if not visited[i]: 
            prev = curr_path[level - 1]
            cost = adj[prev][i]

            new_adj = deepcopy(adj)

            set_inf(new_adj, prev, i) 
            new_adj[i][0] = float('inf')
                
            reduction = row_reduce(new_adj) + col_reduce(new_adj)

            total_cost = heuristic + cost + reduction

            if total_cost < final_cost.value and total_cost / (level + 1) < 1.20 * (final_cost.value / N): 
                curr_path[level] = i 
                new_visited = visited[:]
                new_visited[i] = True

                # Push to heap
                #print("Adding {} with cost {}".format(i, total_cost))
                heapq.heappush(horizon, Horizon(total_cost / (level + 1), total_cost, new_adj, level + 1, curr_path[:], new_visited[:]))

            else:
                # print("Not going there cause it's too expensive")
                continue


def tsp(adj): 
    """
    Set up the tour and call the helper function
    """
    curr_path = [-1] * (N + 1) 
    visited = [False] * N 

    heuristic = row_reduce(adj) + col_reduce(adj)
    print("Reduced cost at start is {}".format(heuristic))

    # We start at vertex 1 so the first vertex 
    # in curr_path[] is 0 
    visited[0] = True
    curr_path[0] = 0

    global max_level
    max_level = -1

    # Call to tsp_helper for curr_weight 
    # equal to 0 and level 1 
    heapq.heappush(horizon, Horizon(heuristic, heuristic, adj, 1, curr_path, visited))
    while len(horizon) > 0:
        order, heuristic, adj, level, curr_path, visited = heapq.heappop(horizon)
        if heuristic < final_cost.value:
            tsp_helper(heuristic, adj, level, curr_path, visited) 
        else:
            continue
            # print("Not exploring because cost is too high")


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
                distances[i][j] = float('inf')
            else:
                distances[i][j] = euclidean_distance(n1, n2)

    return

    # Convert to to numpy array
    global distances_np
    distances_np = np.array(distances)


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
    return int(math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2))


def random_tour():
    """
    Returns cost and path of a random tour
    """
    tour = [x for x in range(N)]
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

    f.write(str(int(cost)))
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
    time = int(time)

    read_nodes(input_file)
    global N
    N = len(nodes)

    final_tour = multiprocessing.Array("i", [-1] * (N + 1))
    final_cost = multiprocessing.Value("f", INF)

    # Start main as a process
    p = multiprocessing.Process(target=main, name="Main", args=())
    p.start()

    p.join(time)

    # If thread is active
    if p.is_alive():
        print("It has been {}s... Stopping program...".format(time))

        # Terminate Main
        p.terminate()
        p.join()

    write_results(final_cost.value, final_tour[:])

    print("Minimum cost: {}".format(int(final_cost.value)))
    for node in final_tour:
        print("{} ".format(node), end="")
    print()
    
