import sys
import networkx as nx
import random
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    raise ValueError(
        "Call this script as `python blueprint_generator.py <seed>`, "
        "where <seed> is an arbitrary integer."
    )
seed = sys.argv[1]
p = 0.8

random.seed(seed)
initial_graph = nx.grid_2d_graph(10, 5)
initial_graph = nx.convert_node_labels_to_integers(initial_graph)

repeat = True
while repeat:
    edges = []
    for e in initial_graph.edges:
        if random.random() < p:
            edges.append(e)

    graph = nx.Graph(edges)
    repeat = not nx.is_connected(graph)

with open(f"blueprints/{seed}.txt", "w") as f:
    for e in graph.edges:
        f.write(str(e)+"\n")

print(f"New blueprint created at `blueprints/{seed}.txt`")

#nx.draw(graph)
#plt.show()
