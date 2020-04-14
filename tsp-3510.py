import sys
import heapq
import multiprocessing
import math
import random

from collections import namedtuple
from random import shuffle


# Types
Point = namedtuple('Point', ['x', 'y'])
Horizon = namedtuple('Horizon', ['order', 'heuristic', 'lower_bound', 'level', 'curr_path', 'visited'])


# Constants
INF = float('inf')


# Process variables
nodes = []
distances = []
horizon = []
# Keeps track of visited nodes in a particular path
visited = None
found = False


# Global variables
global final_cost, final_tour


def main():
    """
    Main driver for the program, kicks of search
    """
    gen_distances(nodes)

    global first_mins, second_mins
    first_mins = [INF] * N
    second_mins = [INF] * N

    tsp(distances) 


def update_final(curr_path, curr_cost): 
    """
    Update the final path with the current path
    """
    print("Found new minimum tour!")

    global found
    found = True

    # Update final tour to curr_path
    final_tour[:N + 1] = curr_path
    # Add start back to end
    final_tour[N] = curr_path[0]

    # Update final cost to curr cost
    final_cost.value = curr_cost

    print("Updated final to {}".format(curr_cost))


def gen_mins():
    """
    Pre-gen all first and second min distances
    """
    print("Generating all first and second minimum distances")

    for i in range(N):
        first_mins[i] = first_min(i)
        second_mins[i] = second_min(i)


def first_min(i): 
    """
    Finds the minimum edge cost from node i to any node
    """
    min_dist = INF 
    for j in range(N): 
        if i != j:
            dist = get_distance(i, j)
            min_dist = min(dist, min_dist) 

    if min_dist == INF:
        print(i)

    return min_dist


def second_min(i): 
    """
    Finds the second minimum edge cost from node i to any node
    """
    first = second = INF
    for j in range(N): 
        if i == j: 
            continue

        dist = get_distance(i, j)

        if dist <= first: 
            second = first 
            first = dist  

        elif dist <= second and dist != first: 
            second = dist 

    return second 


def tsp_helper(heuristic, lower_bound, level, curr_path, visited): 
    """
    Helper function to build the path
    adj: Adjacency matrix
    level: Current level
    curr_path: Path we are building
    visited: Nodes in the path
    """
    global max_level
    if level > max_level:
        print("Exploring level {}".format(level))
        max_level = level

    # At leaf node
    if level == N: 
        prev = curr_path[level - 1]
        cost_to_return = get_distance(prev, 0) # Gotta go back to 0
        curr_res = heuristic + cost_to_return

        if curr_res < final_cost.value: 
            update_final(curr_path, curr_res) 
        return

    # Iterate through all neighbors
    for i in range(N): 
        if not visited[i]: 
            prev = curr_path[level - 1]
            cost = get_distance(prev, i)

            total_cost = heuristic + cost
            new_level = level + 1


            new_lower_bound = lower_bound
            if level == 1: 
                new_lower_bound -= (first_mins[prev] + first_mins[i]) / 2
            else: 
                new_lower_bound -= (second_mins[prev] + first_mins[i]) / 2


            if new_lower_bound + total_cost < final_cost.value and not (False): 
                curr_path[level] = i 
                new_visited = visited[:]
                new_visited[i] = True

                global found

                if found:
                    order = total_cost / new_level
                elif max_level < 0.25 * N:
                    order = total_cost / new_level**1.20
                elif max_level < 0.5 * N:
                    order = total_cost / new_level**1.15
                elif max_level < 0.75 * N:
                    order = total_cost / new_level**1.10
                else:
                    order = total_cost / new_level**1.05

                # Push to heap
                heapq.heappush(horizon, Horizon(order, total_cost, new_lower_bound, new_level, curr_path[:], new_visited[:]))

            else:
                # Not going there because it is too expensive
                continue


def tsp(adj): 
    """
    Set up the tour and call the helper function
    """
    print("Starting TSP solver")

    curr_path = [-1] * (N + 1) 
    visited = [False] * N 

    heuristic = 0

    # We start at vertex 1
    visited[0] = True
    curr_path[0] = 0

    global max_level
    max_level = -1
    
    total = 0

    gen_mins()

    # Compute initial lower bounds 
    lower_bound = 0
    for i in range(N): 
        lower_bound += (first_mins[i] + second_mins[i]) 
    lower_bound = int(lower_bound / 2) 

    # Call to tsp_helper for starting node 
    heapq.heappush(horizon, Horizon(heuristic, heuristic, lower_bound, 1, curr_path, visited))
    while len(horizon) > 0:
        order, heuristic, lower_bound, level, curr_path, visited = heapq.heappop(horizon)
        if lower_bound + heuristic < final_cost.value:
            total += 1
            if total % 10000 == 0:
                print("Checking {}th node".format(total))

            tsp_helper(heuristic, lower_bound, level, curr_path, visited) 
        else:
            # Not going there because it is too expensive
            continue


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
                distances[i][j] = 0
            else:
                distances[i][j] = euclidean_distance(n1, n2)

    return


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
    return int(round(math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)))


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
        f.write("{} ".format(node + 1))

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

    if final_cost.value == float('inf'):
        print("Unfortunately, our algorithm was unable to compute a valid path within the alloted time ({}s). Please try a smaller problem set or increase the time limit.".format(time))
    else:
        write_results(final_cost.value, final_tour[:])

        print(int(final_cost.value))
        for node in final_tour:
            print("{} ".format(node + 1), end="")
        print()
    
