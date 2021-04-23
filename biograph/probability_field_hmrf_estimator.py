import networkx as nx
import numpy as np
from tqdm import tqdm

from collections import Counter

class hmrf():

    """

    Create an instance of the hmrf class.

    Parameters:
     - self.graph: networkx object containing 
    """

    def __init__(self,
                 G,
                 K = None,
                 gamma = 5,
                 epochs = 50):

        self.graph = G
        self.epochs = epochs
        self.gamma = gamma
        self.node_attributes = np.unique(np.array([list(self.graph.nodes[n].keys()) for n in self.graph.nodes()]).flatten())

        assert 'cell_type' in self.node_attributes

        cell_types = nx.get_node_attributes(self.graph, 'cell_type')

        self.cell_types = np.unique([cell_types[node] for node in cell_types.keys()])
        self.number_of_cell_types = len(self.cell_types)

        if K == None:
            self.K = self.number_of_cell_types


    def initiate_latent_probability_field(self):

        cell_type_dict = nx.get_node_attributes(self.graph, 'cell_type')

        G = self.graph
        
        for node in tqdm(G.nodes):
            
            neighbour_cell_types = [cell_type_dict[n] for n in G.neighbors(node)]
            neighbour_cell_types_counter = Counter(neighbour_cell_types)
            
            latent_probability_vector = np.zeros(self.number_of_cell_types)
                    
            for cell_type in neighbour_cell_types_counter.keys():
                
                latent_probability_vector[cell_type] = neighbour_cell_types_counter[cell_type]
            
            latent_probability_vector /= np.sum(latent_probability_vector)
            
            nx.set_node_attributes(G, {node:latent_probability_vector}, 'latent_probability_field')

        self.graph = G


    def run(self):

        for i in tqdm(range(self.epochs)):

            self.update_latent_probability_field()


    def calculate_latent_probability_vector(self,
                                            local_probability_vector, 
                                            neighbor_latent_probability_field):
        
        latent_probability_vector =  np.zeros(self.number_of_cell_types)
        
        for k in range(self.number_of_cell_types):
            
            latent_probability_vector[k] = self.gamma*local_probability_vector[k] + np.sum([neighbor_probability_vector[k] for neighbor_probability_vector in neighbor_latent_probability_field])
            
        latent_probability_vector /= np.sum(latent_probability_vector)
        
        return latent_probability_vector

    def update_latent_probability_field(self):
        
        latent_probability_field_dict = nx.get_node_attributes(self.graph, 'latent_probability_field')
        cell_type_dict = nx.get_node_attributes(self.graph, 'cell_type')

        G = self.graph
        
        for node in self.graph.nodes:
                    
            local_probability_vector =  np.zeros(self.number_of_cell_types)
            local_probability_vector[cell_type_dict[node]] = 1
            
            neighbor_latent_probability_field = [latent_probability_field_dict[n] for n in G.neighbors(node)]
            
            
            latent_probability_vector = self.calculate_latent_probability_vector(local_probability_vector,
                                                                            neighbor_latent_probability_field)
                    
            nx.set_node_attributes(G, {node:latent_probability_vector}, 'latent_probability_field')
            
        self.graph = G
            