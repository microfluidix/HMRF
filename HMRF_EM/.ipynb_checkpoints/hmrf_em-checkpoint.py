import networkx as nx
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas
from collections import Counter

import biograph
from biograph import graphplot
from biograph import hmrf_estimator
from biograph import probability_field_hmrf_estimator

class hmrf():
    
    """

    Create an instance of the hmrf class.

    Parameters:
     - self.graph: networkx object containing 
    """
    
     def __init__(self,
                 G,
                 K = 5,
                 beta = 1
                 epsilon = 10**(-4)):

        self.graph = G
        self.K = K
        self.beta = beta
        
        self.mu = []
        self.cov = []
        
        self.node_attributes = np.unique(np.array([list(self.graph.nodes[n].keys()) for n in self.graph.nodes()]).flatten())

        assert 'cell_type' in self.node_attributes
        
        cell_types = nx.get_node_attributes(self.graph, 'cell_type')

        self.cell_types = np.unique([cell_types[node] for node in cell_types.keys()])
        self.number_of_cell_types = len(self.cell_types)

        self.color_list= color_list = [plt.cm.Set3(i) for i in range(self.K)]
        
    
    def initiate_model(self):
        biograph = probability_field_hmrf_estimator.hmrf(G, epochs = 1, gamma = self.beta, K = 6)
        biograph.initiate_latent_probability_field()
        
        latent_probability_field_properties = probability_field_hmrf_estimator.get_latent_probability_field_properties(biograph.graph, biograph.number_of_cell_types)
        
        G = biograph.graph
        n_rows, n_cols = latent_probability_field_properties.shape
        X = latent_probability_field_properties.values.reshape(-1,n_cols)
        X = preprocessing.StandardScaler().fit_transform(X)
        
        kmeans = KMeans(n_clusters= self.K, random_state=0).fit(X)
        
        for node in sorted(G.nodes):
            nx.set_node_attributes(G, {node:kmeans.labels_[node]}, 'class')
            nx.set_node_attributes(G, {node:self.color_list[kmeans.labels_[node]]}, 'color')
            nx.set_node_attributes(G, {node:kmeans.labels_[node]}, 'legend')
            
        self.graph = G
        
        # initialisation
        cl = categorical_vector(self.graph, 'class')
        ct = categorical_vector(self.graph, 'cell_type')
        for j in range(K):
            cell_type_j = ct[cl == j]
            self.mu.append(np.mean(cell_type_j))
            self.cov.append(np.cov(cell_type_j))
        
    
        
def categorical_vector(G, category):
    
    cat = nx.get_node_attributes(G, category)
    type_of_data = type(cat[0])
    V = np.array(list(cat.items()), dtype=type_of_data)
    a = map(int, V[:,0])
    a = np.array(list(a))
    ind = np.argsort(a)
    Vect = V[:,1][ind]
    
    return Vect