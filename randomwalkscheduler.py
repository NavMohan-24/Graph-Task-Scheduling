import random
import numpy as np
import networkx as nx

class auxGraphConstructer:
    """
    A class for constructing graph and its auxillary graph from edges provided.
    Auxillary graph is constructed by adding labels and weights to nodes and edges respectively.
    
    Attributes:
    edges (list): The list containing edges of the graph
    """

    def __init__(self,edges:list):
        self.edges = edges
        
    def construct_aux_graph(self):
        """
        constructs the auxillary graph. 
        Nodes will have label "isVisted". 
        Edges have 3 features: weights, probability and color.
        Weights = Time for the combination of nodes.
        Probability = assigned in such a manner that, edges taking least time will have highest probability.
        Color = its a parameter used in scheduling task. If color two pair of edges are same, 
        then edge contraction can be performed simultaneously.
        """
        edges = self.edges
        G = nx.Graph()
        G.add_edges_from(edges)
        nodes = list(G.nodes())
        weights = []
        
        for edge in edges:

            dela = G.degree(edge[0])
            delb = G.degree(edge[1])
            n_edges = G.number_of_edges(*(edge))
            t = 2**(dela+delb-n_edges)
            weights.append(t)
        
        probabilities = [((max(weights)-w)+0.01)/max(weights) for w in weights]
        probability =  [p/sum(probabilities) for p in probabilities]

        aux_graph = nx.Graph()
        for i in range(len(edges)):
            aux_graph.add_edge(*(edges[i]),weights=weights[i],probability=probability[i],color=-1)
        
        for node in nodes:
            aux_graph.nodes[node]["isVisited"] = False
        
        return aux_graph




class RandomWalkScheduler(auxGraphConstructer):
    """
    Schedules the edge contraction using edge coloring algorithm. The traversal on the graph
    is done by Random Walks.

    Attributes:
    auxG (networkx.Graph) :The auxillary graph constructed from the edges provided.
    num_of_steps (int) : Maximum number of steps the random walker takes. Fixed to a value which is 10 times the nodes in th graph.
    current_node (int) : Gives the info about the current node occupied by the Random walker during traversal.
    memory (set) : Stores the info about the neighbors in the previous node.
    num_labs(int): Number of labs to which task is shared.
    lab_buffer (list): Stores info about the labs.
    buffer_index (int): Index used to assign tasks to labs in a periodic fashion.


    """
    def __init__(self,edges, n_labs:int): # make num_of_steps optional
        super().__init__(edges)
        self.auxG = self.construct_aux_graph()
        self.num_of_steps = len(self.auxG.nodes())*100 # number of steps taken by random walker
        self.current_node = self.start() # select one of the central node as starting point
        self.memory = {}
        self.num_labs = n_labs
        self.lab_buffer = [f'Lab-{i+1}' for i in range(n_labs)] 
        self.buffer_index = 0
        
    
    def allocate(self):
        """
        Allocates Labs at each time steps.
        """
        if self.buffer_index >= len(self.lab_buffer):
            self.buffer_index = 0  # Reset index if it exceeds buffer length
        value = self.lab_buffer[self.buffer_index]
        self.buffer_index += 1
        return value
    
    def _maxdegree_(self):
        """
        Finds the maximum degree of nodes in the graph.

        Returns:
        Maximum of degree in the graph.
        """
        nodes = self.auxG.nodes()
        degrees = max([self.auxG.degree(node) for node in nodes]) # maximum number of degrees in the graph
        return degrees
    
    def start(self):
        """
        Chose an random initial node to start the traversal of the random walker

        Returns:
        Node for starting traversal.
        """
        nodes = list(self.auxG.nodes())
        initial_point = random.choices(nodes)[0]
        return initial_point

    
    def lookup(self,G):
        """
        Color the edges of a node. Number of colors used to mark the edges is equal to maximum degree.
        If degree = 4, then colors avialable for marking is (1,2,3,4).

        Parameters:
        G (networkx.Graph): The graph whose edges has to be painted.

        Returns:
        (network.Graph) : Colored graph
        """
        node = self.current_node
        d = self._maxdegree_()
        colors_avail = set(range(1,d+1))
        color_used  = set([])
        edges_to_color = {}
        for neighbor in G.neighbors(node):
            if G.get_edge_data(node,neighbor)['color'] != -1:
                color_used.add(G.get_edge_data(node,neighbor)['color'])
            else:
                edges_to_color[(node,neighbor)] = G.get_edge_data(node,neighbor)['weights']
            
        if color_used:
            c = colors_avail-color_used
        else:
            c = colors_avail
        edges = dict(sorted(edges_to_color.items(), key=lambda item:item[1]))
        for edge in edges:
            G.get_edge_data(*(edge))['color'] = min(c)
            c = c-set([min(c)])

        return G

    def rectify(self,G):

        '''
        While at nodes, the randomwalker looks into mistakes made in coloring during previous iterations:

        Parameters:
        G (networkx.Graph): Colored graph

        Returns:
        dict : containins memory of the current iteration. Dict contains info of the 
        neighbors of the current node and color of the edges shared between them.
        '''

        node = self.current_node
        updated_memory = {}
        for neighbor in G.neighbors(node):
            updated_memory[neighbor] = G.get_edge_data(node,neighbor)['color']
        keys1 = self.memory.keys()
        keys2 = updated_memory.keys()
        common_neighbors = set(keys1).intersection(set(keys2))
        if common_neighbors:
            for neighbor in common_neighbors:
                if self.memory[neighbor] == updated_memory[neighbor]:
                    G.get_edge_data(node,neighbor)['color'] = -1
                    updated_memory[neighbor] = -1

        self.auxG = G
        return updated_memory

    def traverse(self):
        '''
        Traverse to next node

        Returns:
        node = node where the randomwalker is going to be occupy.
        '''
        node = self.current_node
        G = self.auxG
        neighbors = list(G.neighbors(node))

        probability_distribution = [G.get_edge_data(node,neighbor)['probability'] for neighbor in neighbors]

        node = random.choices(neighbors,probability_distribution,k=1)[0]

        return node
    
    def make_colored_graph(self):

        """
        Colors the entire graph.

        Returns:
        colored graph
        """

        G = self.auxG
        register = {node: G.nodes[node]['isVisited'] for node in G.nodes()}
        for i in range(self.num_of_steps):
            # if all(register.values()): # check if all nodes have been visited
            #     colored_graph = G
            #     return colored_graph 
            
            if G.nodes[self.current_node]['isVisited']: # checks whether the node was visted
                self.memory = self.rectify(G) # identifies mistakenly marked edges and try to rectify it **can make error sometime**
                self.current_node = self.traverse() # move to the next node by updating the current node
            else:
                G.nodes[self.current_node]['isVisited']=True
                register[self.current_node] = True  
                G = self.lookup(G) # colors the edges of the node given.
                self.memory = self.rectify(G)
                self.current_node = self.traverse()
        colored_graph = G
        return colored_graph

    def create_schedule(self):

        """
        Create a shedules to perform combinations of components. (contraction of edges)

        Returns:

        dict : contains info about the edges to contract and labs assigned.

        """

        edges = self.auxG.edges()
        schedule = {}
        for i in range(0,len(edges)):
            self.auxG = self.construct_aux_graph()

            if len(self.auxG.nodes()) <= 2:
                schedule[f'time_slot-{i+1}'] = {'Labs_Allocated': [self.allocate()],
                                                'Edges_To_Combine':list(self.auxG.edges())}
                return schedule
            self.current_node = self.start()
            colored_graph = self.make_colored_graph()
            edges_to_contract = [edge for edge in colored_graph.edges() if colored_graph.get_edge_data(*(edge))['color']==1][0:self.num_labs]

            
            new_label = {}
            for edge in edges_to_contract:
                if edge in colored_graph.edges():
                    nx.contracted_nodes(colored_graph,*(edge),self_loops=False,copy=False)
                    name = f'{edge[0]}'.split('_')[0]
                    new_label[edge[0]] = f'{name}_t{i+1}*'
                else:
                    edges_to_contract.remove(edge)
            nx.relabel_nodes(colored_graph,new_label,copy=False)
            self.edges = list(colored_graph.edges())
            schedule[f'time_slot-{i+1}']= {'Labs_Allocated': [self.allocate() for i in range(len(edges_to_contract))], 
                                            'Edges_To_Combine': edges_to_contract}
            
            
    
        
        return schedule






    

        









