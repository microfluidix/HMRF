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
                 beta = 1,
                 epsilon = 0.0001):
        
        self.graph = G
        self.K = K
        self.beta = beta
        self.epsilon = epsilon

        self.mu = []
        self.cov = []

        self.node_attributes = np.unique(np.array([list(self.graph.nodes[n].keys()) for n in self.graph.nodes()]).flatten())

        assert 'cell_type' in self.node_attributes

        cell_types = nx.get_node_attributes(self.graph, 'cell_type')

        self.cell_types = np.unique([cell_types[node] for node in cell_types.keys()])
        self.number_of_cell_types = len(self.cell_types)

        self.color_list= color_list = [plt.cm.Set3(i) for i in range(self.K)]
        
    
    def initiate_model(self):
        biograph = probability_field_hmrf_estimator.hmrf(self.graph, epochs = 1, gamma = self.beta, K = 6)
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
        for j in range(self.K):
            cell_type_j = ct[cl == j]
            self.mu.append(np.mean(cell_type_j))
            self.cov.append(np.cov(cell_type_j))
             
        
    def expectation_step(self):
        
        G = self.graph

        cell_class_dict = nx.get_node_attributes(G, 'class')

        P_neigh = np.zeros((len(G.nodes), self.K))

        for node in G.nodes:

            neighbour_cell_class = [cell_class_dict[n] for n in G.neighbors(node)]
            neighbour_cell_class_counter = Counter(neighbour_cell_class)

            for cell_class in neighbour_cell_class_counter.keys():

                for k in range(self.K):
                    P_neigh[node, k] += self.beta*int(k == cell_class)*neighbour_cell_class_counter[cell_class]

        P_gauss = np.zeros((len(G.nodes), self.K))

        cell_type_dict = nx.get_node_attributes(G, 'cell_type')

        for node in G.nodes:

            x = cell_type_dict[node]

            for j in range(self.K):
                pref = 1/np.sqrt(2*np.pi*self.cov[j])
                P_gauss[node, j] = pref*np.exp(-0.5*((x-self.mu[j])**2)/self.cov[j])

        # calculate product proba

        prod_proba = P_gauss * P_neigh

        # calculate gamma proba

        gamma = np.zeros((len(G.nodes), self.K))

        for j in range(self.K):
            gamma[:, j] = prod_proba[:, j]/np.sum(prod_proba,1)
            
        return gamma
    
    
    def maximisation_step(self):
        
        gamma = self.expectation_step()
        
        G = self.graph
        
        new_class = np.argmax(gamma, axis=1)

        for node in sorted(G.nodes):
            nx.set_node_attributes(G, {node:new_class[node]}, 'class')
            nx.set_node_attributes(G, {node:self.color_list[new_class[node]]}, 'color')
            nx.set_node_attributes(G, {node:new_class[node]}, 'legend')
            
        self.graph = G
        
        ct = categorical_vector(self.graph, 'cell_type')
        
        n = np.sum(gamma, axis = 0)

        new_mu = np.dot(gamma.T, ct)/n

        new_cov = np.zeros(self.K)
        
        for j in range(self.K):
            for i in range(len(ct)):
                new_cov[j] += gamma[i, j]*((ct[i]-new_mu[j])**2)
        new_cov /= n
        
        return new_mu, new_cov
    
    def run(self):
        
        stopcrit_mu = self.epsilon + 1
        stopcrit_cov = self.epsilon + 1
        
        cpt = 0
        print(cpt)
        
        while (stopcrit_mu >= self.epsilon) or (stopcrit_cov >= self.epsilon):
            new_mu, new_cov = self.maximisation_step()
            
            copy_stopcrit_mu = np.copy(stopcrit_mu)
            copy_stopcrit_cov = np.copy(stopcrit_cov)
            
            # Stopping criteria
            stopcrit_mu = np.max(np.abs(new_mu - self.mu))/(1 + np.max([new_mu, self.mu]))
            stopcrit_cov = np.max(np.abs(new_cov - self.cov))/(1 + np.max([new_cov, self.cov]))
            print(stopcrit_mu, stopcrit_cov, cpt)
            
            self.mu = new_mu
            self.cov = new_cov
            
            cpt += 1
            
            if round(float(copy_stopcrit_mu), 6) == round(stopcrit_mu, 6):
                if round(float(copy_stopcrit_cov), 6) == round(stopcrit_cov, 6):
                    print('stationnary state : convergence issue')
                    break
            
            if cpt > 100:
                break
        
        print(cpt)
            
        
def categorical_vector(G, category):
    
    cat = nx.get_node_attributes(G, category)
    type_of_data = type(cat[0])
    V = np.array(list(cat.items()), dtype=type_of_data)
    a = map(int, V[:,0])
    a = np.array(list(a))
    ind = np.argsort(a)
    Vect = V[:,1][ind]
    
    return Vect