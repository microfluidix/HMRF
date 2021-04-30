import networkx as nx
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas
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
                 KMeans = None,
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
        else:
            self.K = K

        self.color_list = [plt.cm.Set3(i) for i in range(self.K)]
        self.KMeans = KMeans


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

    def assign_cell_class(self, 
                          K = None):

        if K != None:
            self.K = K
            
        if self.KMeans == None:

            latent_probability_field_properties = get_latent_probability_field_properties(self.graph, 
                                            self.number_of_cell_types)

            latent_probability_field_nodes = list(latent_probability_field_properties.node.values)
            latent_probability_field_properties = latent_probability_field_properties.drop(['node'], axis = 1)

            n_rows, n_cols = latent_probability_field_properties.shape
            G = self.graph

            # add 1e-4 in case of zero values. Avoids divergence in log.

            X = np.log(latent_probability_field_properties + 1e-4).values.reshape(-1,n_cols)
            X = preprocessing.StandardScaler().fit_transform(X)

            kmeans = KMeans(n_clusters=self.K, random_state=0).fit(X)
            self.color_list = [plt.cm.Set3(i) for i in range(len(np.unique(kmeans.labels_)))]

            for node in sorted(G.nodes):

                nx.set_node_attributes(G, {node:kmeans.labels_[latent_probability_field_nodes.index(node)]}, 'class')
                nx.set_node_attributes(G, {node:self.color_list[kmeans.labels_[latent_probability_field_nodes.index(node)]]}, 'color')
                nx.set_node_attributes(G, {node:kmeans.labels_[latent_probability_field_nodes.index(node)]}, 'legend')

            self.graph = G
            self.KMeans = kmeans
            
        else:
            
            latent_probability_field_properties = get_latent_probability_field_properties(self.graph, 
                                            self.number_of_cell_types)

            latent_probability_field_nodes = list(latent_probability_field_properties.node.values)
            latent_probability_field_properties = latent_probability_field_properties.drop(['node'], axis = 1)

            n_rows, n_cols = latent_probability_field_properties.shape
            G = self.graph

            X = np.log(latent_probability_field_properties).values.reshape(-1,n_cols)
            X = preprocessing.StandardScaler().fit_transform(X)
            
            labels = self.KMeans.predict(X)
            self.color_list = [plt.cm.Set3(i) for i in range(len(np.unique(labels)))]

            for node in sorted(G.nodes):

                nx.set_node_attributes(G, {node:labels[latent_probability_field_nodes.index(node)]}, 'class')
                nx.set_node_attributes(G, {node:self.color_list[labels[latent_probability_field_nodes.index(node)]]}, 'color')
                nx.set_node_attributes(G, {node:labels[latent_probability_field_nodes.index(node)]}, 'legend')

            self.graph = G
        
            

def get_latent_probability_field_properties(G, number_of_cell_types):

    resultframe = pandas.DataFrame()
    i = 0
    
    latent_probability_field = nx.get_node_attributes(G, 'latent_probability_field')

    for node in sorted(G.nodes):
        
        resultframe.loc[i, 'node'] = node
        
        for k in range(number_of_cell_types):
            resultframe.loc[i, k] = latent_probability_field[node][k]
            
        i += 1

    return resultframe.fillna(0)