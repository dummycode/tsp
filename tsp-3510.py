import sys
import multiprocessing
import math
import numpy as np
from collections import namedtuple
from random import shuffle
from ctypes import c_char_p


# Types
Point = namedtuple('Point', ['x', 'y'])


# Constants
INF = float('inf')


# Process variables
nodes = []
distances = []
tour = None

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

    min_cost = float("inf")
    min_tour = None

    global tour

    tour = [None] * (N + 1)

    TSP(distances) 


def updateFinal(curr_path): 
    """
    Update the final path with the current path
    """
    global tour

    tour[:N + 1] = curr_path[:] 
    tour[N] = curr_path[0] 

    final_tour[:] = tour


def firstMin(adj, i): 
    """
    Finds the minimum edge cost from any node to node i
    """
    min = INF 
    for k in range(N): 
        if adj[i][k] < min and i != k: 
            min = adj[i][k] 

    return min


def secondMin(adj, i): 
    """
    Finds the second minimum edge cost from any node to node i
    """
    first, second = INF, INF 
    for j in range(N): 
        if i == j: 
            continue
        if adj[i][j] <= first: 
            second = first 
            first = adj[i][j] 

        elif(adj[i][j] <= second and adj[i][j] != first): 
            second = adj[i][j] 

    return second 


# function that takes as arguments: 
# curr_bound -> lower bound of the root node 
# curr_weight-> stores the weight of the path so far 
# level-> current level while moving 
# in the search space tree 
# curr_path[] -> where the solution is being stored 
# which would later be copied to tour[] 
def TSPRec(adj, curr_bound, curr_weight, level, curr_path, visited): 
    # At leaf node
    if level == N: 
        # curr_res has the total weight 
        # of the solution we got 
        curr_res = curr_weight + adj[curr_path[level - 1]][curr_path[0]] 
        if curr_res < final_cost.value: 
            updateFinal(curr_path) 
            final_cost.value = curr_res 
        return

    # for any other level iterate for all vertices 
    # to build the search space tree recursively 
    for i in range(N): 
        # Consider next vertex if it is not same 
        # (diagonal entry in adjacency matrix and 
        # not visited already) 
        if (adj[curr_path[level-1]][i] != 0 and
                visited[i] == False): 
            temp = curr_bound 
            curr_weight += adj[curr_path[level - 1]][i] 

            # different computation of curr_bound 
            # for level 2 from the other levels 
            if level == 1: 
                curr_bound -= ((firstMin(adj, curr_path[level - 1]) + firstMin(adj, i)) / 2) 
            else: 
                curr_bound -= ((secondMin(adj, curr_path[level - 1]) + firstMin(adj, i)) / 2) 

            # curr_bound + curr_weight is the actual lower bound 
            # for the node that we have arrived on. 
            # If current lower bound < final_cost.value, 
            # we need to explore the node further 
            if curr_bound + curr_weight < final_cost.value: 
                curr_path[level] = i 
                visited[i] = True

                # call TSPRec for the next level 
                TSPRec(adj, curr_bound, curr_weight, level + 1, curr_path, visited) 

            # Else we have to prune the node by resetting 
            # all changes to curr_weight and curr_bound 
            curr_weight -= adj[curr_path[level - 1]][i] 
            curr_bound = temp 

            # Also reset the visited array 
            visited = [False] * len(visited) 
            for j in range(level): 
                if curr_path[j] != -1: 
                    visited[curr_path[j]] = True


# This function sets up tour 
def TSP(adj): 
    # Calculate initial lower bound for the root node 
    # using the formula 1/2 * (sum of first min + 
    # second min) for all edges. Also initialize the 
    # curr_path and visited array 
    curr_bound = 0
    curr_path = [-1] * (N + 1) 
    visited = [False] * N 

    # Compute initial bound 
    for i in range(N): 
        curr_bound += (firstMin(adj, i) + secondMin(adj, i)) 

    curr_bound = int(curr_bound / 2) 

    # We start at vertex 1 so the first vertex 
    # in curr_path[] is 0 
    visited[0] = True
    curr_path[0] = 0

    # Call to TSPRec for curr_weight 
    # equal to 0 and level 1 
    TSPRec(adj, curr_bound, 0, 1, curr_path, visited) 


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
            distances[i][j] = euclidean_distance(n1, n2)

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
