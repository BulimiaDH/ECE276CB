import numpy as np
import sys
import os.path


def load_data(input_file):
    '''
    Read deterministic shortest path specification
    '''
    with np.load(input_file) as data:
        n = data["number_of_nodes"]
        s = data["start_node"]
        t = data["goal_node"]
        C = data["cost_matrix"]
    return n, s, t, C


def plot_graph(C, path_nodes, output_file):
    '''
    Plot a graph with edge weights sepcified in matrix C.
    Saves the output to output_file.
    '''
    from graphviz import Digraph

    G = Digraph(filename=output_file, format='png', engine='neato')
    G.attr('node', colorscheme='accent3', color='1', shape='oval', style="filled", label="")

    # Normalize the edge weights to [1,11] to fit the colorscheme
    maxC = np.max(C[np.isfinite(C)])
    minC = np.min(C)
    norC = 10 * np.nan_to_num((C - minC) / (maxC - minC)) + 1

    # Add edges with non-infinite cost to the graph
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            if C[i, j] < np.inf:
                G.edge(str(i), str(j), colorscheme="rdylbu11", color="{:d}".format(int(norC[i, j])))

    # Display path
    for n in path_nodes:
        G.node(str(n), colorscheme='accent3', color='3', shape='oval', style="filled")

    G.view()


def save_results(path, cost, output_file):
    '''
    write the path and cost arrays to a text file
    '''
    with open(output_file, 'w') as fp:
        for i in range(len(path)):
            fp.write('%d ' % path[i])
        fp.write('\n')
        for i in range(len(cost)):
            fp.write('%.2f ' % cost[i])


def dsp(n, s, t, C):
    '''
    DSP algorithm using label correction.
    '''
    openList = [s]
    g = [float('inf') for i in range(n)]
    g[s] = 0
    parent = [-1 for i in range(n)]
    while (len(openList) > 0):
        i = openList.pop()
        for j in range(n):
            if (g[i] + C[i, j]) < g[j] and (g[i] + C[i, j]) < g[t]:
                g[j] = g[i] + C[i, j]
                parent[j] = i
                if j != t:
                    openList = [j] + openList
    end = int(t)
    path, cost = [],[]

    while (end != -1):
        path.append(int(end))
        cost.append(g[end])
        end = parent[end]
    return path[::-1], cost


if __name__ == "__main__":
    import os

    os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

    # input_file = sys.argv[1]
    input_file = '../data/problem1.npz'
    file_name = os.path.splitext(input_file)[0]

    # Load data
    n, s, t, C = load_data(input_file)
    # np.savetxt('C.txt', C)
    # print(n,s,t,C)

    path, cost = dsp(n, s, t, C)
    print(path, cost)
    # Generate results
    # path = np.array([42, 43, 44, 53, 61, 70, 79, 80, 81, 82, 83, 84, 85, 86, 87, 98, 109])
    # cost = np.array([16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4., 3.0, 2.0, 1.0, 0.0])
    # Visualize (requires: pip install graphviz --user)
    plot_graph(C, path, file_name)

    # Save the results
    save_results(path, cost, file_name + "_results.txt")
