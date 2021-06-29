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


    The default legend of the returned graph is the cell class.
    """
    
    def __init__(self,
                 G,
                 K = 5,
                 beta = 1,
                 max_it = 50,
                 KMeans = None):
        
        cell_types = nx.get_node_attributes(G, 'cell_type')
        
        self.graph = G.copy()
        self.K = K
        self.beta = beta
        self.max_it = max_it
        self.mu = []
        self.sigma2 = []
        self.node_attributes = np.unique(np.array([list(self.graph.nodes[n].keys()) for n in self.graph.nodes()]).flatten())
        self.cell_types = np.unique([cell_types[node] for node in cell_types.keys()])
        self.number_of_cell_types = len(self.cell_types)
        self.color_list = [plt.cm.Set2(i) for i in range(self.K)]
        self.KMean = KMeans
        self.parameters = None
   
        
    def initiate_model(self):
        
        # Initiate latent field
        
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
        
        latent_probability_field_properties = hmrf.get_latent_probability_field_properties(self.graph, self.number_of_cell_types)
        
        # Apply Kmeans clustering on this initial latent field
        
        G = self.graph
        n_rows, n_cols = latent_probability_field_properties.shape
        X = np.log(latent_probability_field_properties+1e-4).values.reshape(-1,n_cols)
        X = preprocessing.StandardScaler().fit_transform(X)
        
        # We initialize the labels based on another tissue
        
        if self.KMean:
            labels = self.KMean.predict(X)
        else:
            kmeans = KMeans(n_clusters= self.K, random_state=0).fit(X)
            labels = kmeans.predict(X)
        
        for node in sorted(G.nodes):
            nx.set_node_attributes(G, {node:labels[node]}, 'class')
            nx.set_node_attributes(G, {node:self.color_list[labels[node]]}, 'color')
            nx.set_node_attributes(G, {node:labels[node]}, 'legend')
            
        # graph actualization & initialisation of parameters
        self.graph = G
        self.mu, self.sigma2 = self.update_parameters()

    def update_labels(self):
        
        # Important quantities
        cell_class_dict = nx.get_node_attributes(self.graph, 'class') # dict of cell labels
        cell_type_list = hmrf.categorical_vector(self.graph, 'cell_type') # list of cell types

        N = len(cell_type_list) # Number of cell
        M = len(np.unique(cell_type_list)) # Number of cell types

        # Create matrix from cell type
        mat_cell_type = np.zeros((N, M))
        for i in range(N):
            mat_cell_type[i, cell_type_list[i]] = 1
        
        # Influence of neighbors labels

        log_P_neigh = np.zeros((len(self.graph.nodes), self.K))

        for node in self.graph.nodes:

            neighbour_cell_class = [cell_class_dict[n] for n in self.graph.neighbors(node)]
            neighbour_cell_class_counter = Counter(neighbour_cell_class)

            for cell_class in neighbour_cell_class_counter.keys():

                for j in range(self.K):
                    log_P_neigh[node, j] += self.beta*int(j == cell_class)*neighbour_cell_class_counter[cell_class]
                    
        # Emission log-probability

        log_P_gauss = np.zeros((len(self.graph.nodes), self.K))
        
        # PAS MEGA ELEGANT...

        for j in range(self.K):
            var = self.sigma2[j]
            
            for i in range(N):
                xi = mat_cell_type[i]
                
                for m in range(M):
                    
                    a = (-0.5*(xi[m]-self.mu[j,m])**2)
                
                    if a != 0:
                        if var[m,m] == 0:
                            a = -np.inf
                        else: 
                            a /= var[m,m]
                        
                    if ~np.isnan(a):
                        if a == -np.inf:
                            log_P_gauss[i, j] += -1e10
                        else:
                            log_P_gauss[i, j] += a
                            
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
        cell_type_list = hmrf.categorical_vector(self.graph, 'cell_type')

        # Number of cells
        N = len(cell_type_list)

        # Number of cell types
        M = len(np.unique(cell_type_list))

        # Create matrix from cell type
        mat_cell_type = np.zeros((N, M))
        for i in range(N):
            mat_cell_type[i, cell_type_list[i]] = 1

        # List of labels
        cell_class_list = hmrf.categorical_vector(self.graph, 'class')

        # Number of cell labels
        classes, card_classes = np.unique(cell_class_list, return_counts=True)

        # Little trick to allow loosing some labels
        card_classes2 = np.zeros(self.K)
        card_classes2[classes] = card_classes
        card_classes = card_classes2

        # Count frequencies of cell types in each latent class
        freq = np.zeros((self.K, M))

        # Variability inside class
        sig = [np.eye(M) for j in range(self.K)]

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
        
# ---------------------------------------------------------------------------------------------------------------  

    @staticmethod
    def categorical_vector(G, category):
    
        cat = nx.get_node_attributes(G, category)
        type_of_data = type(cat[0])
        V = np.array(list(cat.items()), dtype=type_of_data)
        a = map(int, V[:,0])
        a = np.array(list(a))
        ind = np.argsort(a)
        Vect = V[:,1][ind]

        return Vect

    @staticmethod
    def get_latent_probability_field_properties(G, number_of_cell_types):

        resultframe = pandas.DataFrame()
        i = 0

        latent_probability_field = nx.get_node_attributes(G, 'latent_probability_field')

        for node in sorted(G.nodes):

            for k in range(number_of_cell_types):
                resultframe.loc[i, k] = latent_probability_field[node][k]

            i += 1

        return resultframe.fillna(0)
