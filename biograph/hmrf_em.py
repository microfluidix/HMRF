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


    The default legend of the returned graph is the cell class.
    """
    
    def __init__(self,
                 G,
                 K = 5,
                 beta = 1,
                 max_it = 50,
                 KMeans = None):

        cell_types = nx.get_node_attributes(G, 'cell_type')
        
        self.graph = G
        self.K = K
        self.beta = beta
        self.max_it = max_it
        self.mu = []
        self.sigma2 = []
        self.node_attributes = np.unique(np.array([list(self.graph.nodes[n].keys()) for n in self.graph.nodes()]).flatten())
        self.cell_types = np.unique([cell_types[node] for node in cell_types.keys()])
        self.number_of_cell_types = len(self.cell_types)
        self.color_list = [plt.cm.Set3(i) for i in range(self.K)]
        self.KMean = KMeans
        self.parameters = None
        
    
    def initiate_model(self):

        # Fill the latent space

        biograph = probability_field_hmrf_estimator.hmrf(self.graph, epochs = 1, gamma = self.beta, K = 6)
        biograph.initiate_latent_probability_field()
        
        latent_probability_field_properties = probability_field_hmrf_estimator.get_latent_probability_field_properties(biograph.graph, biograph.number_of_cell_types)
        
        G = biograph.graph
        n_rows, n_cols = latent_probability_field_properties.shape
        X = np.log(latent_probability_field_properties+1e-4).values.reshape(-1,n_cols)
        X = preprocessing.StandardScaler().fit_transform(X)

        # We initialize the labels based on another tissue
        if self.KMean:
            kmeans = self.KMean.predict(X)
        else:
            kmeans = KMeans(n_clusters= self.K, random_state=0).fit(X)
        
        for node in sorted(G.nodes):
            nx.set_node_attributes(G, {node:kmeans.labels_[node]}, 'class')
            nx.set_node_attributes(G, {node:self.color_list[kmeans.labels_[node]]}, 'color')
            nx.set_node_attributes(G, {node:kmeans.labels_[node]}, 'legend')
        
        # graph actualization & initialisation of parameters
        self.graph = G
        self.mu, self.sigma2 = self.update_parameters()

        
    def update_labels(self):
        
        # Important quantities
        cell_class_dict = nx.get_node_attributes(self.graph, 'class') # dict of cell labels
        cell_type_list = categorical_vector(self.graph, 'cell_type') # list of cell types

        N = len(cell_type_list) # Number of cells
        T = len(np.unique(cell_type_list)) # Number of cell types

        # Create matrix from cell type
        mat_cell_type = np.zeros((N, T))
        for i in range(N):
            mat_cell_type[i, cell_type_list[i]] = 1
        
        # Influence of neighbors labels

        log_P_neigh = np.zeros((len(self.graph.nodes), self.K))

        for node in self.graph.nodes:

            neighbour_cell_class = [cell_class_dict[n] for n in self.graph.neighbors(node)]
            neighbour_cell_class_counter = Counter(neighbour_cell_class)

            for cell_class in neighbour_cell_class_counter.keys():

                for k in range(self.K):
                    log_P_neigh[node, k] += self.beta*int(k == cell_class)*neighbour_cell_class_counter[cell_class]
                    
        # Emission log-probability

        log_P_gauss = np.zeros((len(self.graph.nodes), self.K))
        
        # PAS MEGA ELEGANT...

        for k in range(self.K):
            var = self.sigma2[k]
            for i in range(N):
                xi = mat_cell_type[i]
                for t in range(T):
                    a = (-0.5*(xi[t]-self.mu[k,t])**2)/var[t, t]
                    if ~np.isnan(a):
                        if a == -np.inf:
                            log_P_gauss[i, k] += -1e10
                        else:
                            log_P_gauss[i, k] += a
                            
        # MAP criterion to determine new labels

        sum_prob = log_P_gauss + log_P_neigh

        new_class = np.argmax(sum_prob, axis=1)
        
        # Update labels in the graph

        for node in sorted(self.graph.nodes):
            nx.set_node_attributes(self.graph, {node:new_class[node]}, 'class')
            nx.set_node_attributes(self.graph, {node:self.color_list[new_class[node]]}, 'color')
            nx.set_node_attributes(self.graph, {node:new_class[node]}, 'legend')

    def update_parameters(self):

        # List of cell types
        cell_type_list = categorical_vector(self.graph, 'cell_type')

        # Number of cells
        N = len(cell_type_list)

        # Number of cell types
        T = len(np.unique(cell_type_list))

        # Create matrix from cell type
        mat_cell_type = np.zeros((N, T))
        for i in range(N):
            mat_cell_type[i, cell_type_list[i]] = 1

        # List of labels
        cell_class_list = categorical_vector(self.graph, 'class')

        # Number of cell labels
        classes, card_classes = np.unique(cell_class_list, return_counts=True)

        # Little trick to allow loosing some labels
        card_classes2 = np.zeros(self.K)
        card_classes2[classes] = card_classes
        card_classes = card_classes2

        # Count frequencies of cell types in each latent class
        freq = np.zeros((self.K, T))

        # Variability inside class
        sig = [np.eye(T) for j in range(self.K)]

        for j in range(self.K):

            # Count number of cells of each type in class j
            x1, x2 = np.unique(cell_type_list[cell_class_list == j], return_counts=True)
            freq[j, x1] = x2
            freq[j, :] /= card_classes[j]
            freq[j, :] /= np.sum(freq[j, :])

            # Compute variability inside class j
            cell_type_in_j = mat_cell_type[cell_class_list == j]
            sig_j = np.sum((cell_type_in_j - freq[j])**2, axis = 0)
            sig_j /= card_classes[j]
            sig[j] *= sig_j

        return freq, sig
        
             
    def run(self):
        list_param = [[self.mu, self.sigma2]]
        for cpt in tqdm(range(self.max_it)):
            self.update_labels()
            self.mu, self.sigma2 = self.update_parameters()
            list_param.append([self.mu, self.sigma2])
        self.parameters = list_param
        
def categorical_vector(G, category):
    
    cat = nx.get_node_attributes(G, category)
    type_of_data = type(cat[0])
    V = np.array(list(cat.items()), dtype=type_of_data)
    a = map(int, V[:,0])
    a = np.array(list(a))
    ind = np.argsort(a)
    Vect = V[:,1][ind]
    
    return Vect