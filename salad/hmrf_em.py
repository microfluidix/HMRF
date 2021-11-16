import networkx as nx
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas
from collections import Counter

class hmrf:

    """
    Create an instance of the hmrf class.

    Main idea:
    - take as an input a network of cells (nodes) having an attribute
            "cell_type" corresponding to the cell's phenotypes
    - compute a new attribute for each cell (node) called cell_class -- being
            a latent label in the hidden markov random field framework --
            allowing to determine regions (or patterns) in the tissue.

    Parameters
    ----------
     - G (networkX graph object): Network to regionalize. Has at least
            cell_type as an attribute. cell_type is an integer between
            0 and M-1 where M is the number of cell types in the graph.
     - K (int): the number of final regions in the tissue
     - beta (float): the strength of the region coupling. A small value
            leads to less homogeneous regions.
     - max_it (int): number of iterations
     - Kmeans (sklearn.cluster.KMeans object, optional): Kmeans to impose
            an initial latent configuration of cell classes regions

    Returns
    -------
    - hmrf (hmrf object): an instance of the hmrf class.
    """

    def __init__(self, G, K=5, beta=1, max_it=50, KMeans=None):

        cell_types = nx.get_node_attributes(G, "cell_type")

        # check that cell_types are integers
        for type in list(cell_types[key] for key in cell_types.keys()):
            assert isinstance(type, int)

        # check that cell_types are between 0 and M-1
        assert max(cell_types.values()) == len(np.unique(cell_types.values())) - 1

        self.graph = G.copy()
        self.K = K
        self.beta = beta
        self.max_it = max_it
        self.mu = []
        self.sigma2 = []
        self.node_attributes = np.unique(
            np.array(
                [list(self.graph.nodes[n].keys()) for n in self.graph.nodes()]
            ).flatten()
        )
        self.cell_types = np.unique([cell_types[node] for node in cell_types.keys()])
        self.number_of_cell_types = len(self.cell_types)
        self.color_list = [plt.cm.Set2(i) for i in range(self.K)]
        self.KMean = KMeans

        # Parameters of the gaussian influence of cell phenotypes (= cell types)
        self.mu = []  # fraction of cells of each type per region
        self.sigma2 = []  # variability of cell types in a given region
        self.parameters = None  # [self.mu, self.sigma2] for all iterations of the algorithm (to check convergence)

        # Number of cell phenotypes
        cell_type_dict = nx.get_node_attributes(
            G, "cell_type"
        )  # dictionary of all cell types
        self.number_of_cell_types = len(
            np.unique([cell_type_dict[node] for node in cell_type_dict.keys()])
        )  # number of cell types

        self.color_list = [
            plt.cm.viridis(i / (self.K - 1)) for i in range(self.K)
        ]  # colors of each node

    def initiate_model(self):
        # Initialization of the latent field (cell classes)

        # If initial configuration is given as an input
        if self.KMean:
            labels = self.KMean.predict(X)

        # Otherwise, lets compute it
        else:

            cell_type_dict = nx.get_node_attributes(
                self.graph, "cell_type"
            )  # dictionary of all cell types

            G = self.graph

            # For each node, compute the composition of nearest neighbors phenotypes in a vector
            for node in tqdm(G.nodes):

                neighbour_cell_types = [
                    cell_type_dict[n] for n in G.neighbors(node)
                ]  # list of neighbors for this node
                neighbour_cell_types_counter = Counter(
                    neighbour_cell_types
                )  # count number of neighbors of each type

                # Vector containing composition of nearest neighbors (compo_nn)
                compo_nn = np.zeros(self.number_of_cell_types)
                for cell_type in neighbour_cell_types_counter.keys():
                    compo_nn[cell_type] = neighbour_cell_types_counter[cell_type]
                compo_nn /= np.sum(compo_nn)

                # Store it as a node attribute
                nx.set_node_attributes(G, {node: compo_nn}, "compo_nn")

            self.graph = G

            # Get this new node attribute (compo_nn) as a pandas.DataFrame
            compo_nn_properties = hmrf.get_compo_nn_properties(
                self.graph, self.number_of_cell_types
            )

            # Apply Kmeans clustering on this initial field of composition in nearest neighbors
            G = self.graph
            n_rows, n_cols = compo_nn_properties.shape
            X = np.log(compo_nn_properties + 1e-4).values.reshape(-1, n_cols)
            X = preprocessing.StandardScaler().fit_transform(X)

            kmeans = KMeans(n_clusters=self.K, random_state=0).fit(X)
            self.KMeans = kmeans  # Save this initialization
            labels = kmeans.predict(X)

        # Save this initial clustering (kmeans) as the first configuration of the latent field (of cell classes)
        # by attributing each value as an attribute 'class' for each node
        G = self.graph

        for node in sorted(G.nodes):
            nx.set_node_attributes(G, {node: labels[node]}, "class")
            nx.set_node_attributes(G, {node: self.color_list[labels[node]]}, "color")
            nx.set_node_attributes(G, {node: labels[node]}, "legend")

        # graph actualization & initialisation of parameters
        self.graph = G
        self.mu, self.sigma2 = self.update_parameters()

    def update_labels(self):  # update latent field of cell classes

        # Important quantities
        cell_class_list = hmrf.categorical_vector(
            self.graph, "class"
        )  # list of cell classes
        cell_type_list = hmrf.categorical_vector(
            self.graph, "cell_type"
        )  # list of cell types

        N = len(cell_type_list)  # Number of cell
        M = len(np.unique(cell_type_list))  # Number of cell types

        # Create matrix from cell type
        mat_cell_type = np.zeros((N, M))

        list_index = np.array([i for i in range(N)])
        mat_cell_type[list_index, cell_type_list] = 1

        # Influence of neighbors labels (compute log probability)
        log_P_neigh = np.zeros((N, self.K))

        for node in range(N):
            a, b = np.unique(
                [cell_class_list[n] for n in self.graph.neighbors(node)],
                return_counts=True,
            )
            log_P_neigh[node, a.astype(int)] = b

        log_P_neigh *= self.beta

        # Log-probability of emitting a specific latent label knowing cell's phenotype

        Mat = np.ones((N, self.K, M))

        for i in range(N):
            Mat[i, :, :] *= -0.5 * (mat_cell_type[i] - self.mu) ** 2

        list_index = np.array([i for i in range(N)])

        for j in range(self.K):
            var = self.sigma2[j]

            v = np.copy(np.diag(var))
            l = np.where(v == 0)[0]
            v[v == 0] = 1

            Ar = Mat[:, j, :][:, l]
            Ar[Ar != 0] = np.nan
            Mat[:, j, :][:, l] = Ar

            Mat[:, j, :] /= v

        Mat[np.isnan(Mat)] = -1e10

        log_P_gauss = np.sum(Mat, axis=2)

        # MAP criterion to determine new labels
        sum_prob = log_P_gauss + log_P_neigh
        new_class = np.argmax(sum_prob, axis=1)

        # Update labels in the graph (saved as nodes attribute)
        for node in sorted(self.graph.nodes):
            nx.set_node_attributes(self.graph, {node: new_class[node]}, "class")
            nx.set_node_attributes(
                self.graph, {node: self.color_list[new_class[node]]}, "color"
            )
            nx.set_node_attributes(self.graph, {node: new_class[node]}, "legend")

    def update_parameters(self):  # update parameters self.mu and self.sigma2

        cell_type_list = hmrf.categorical_vector(
            self.graph, "cell_type"
        )  # List of cell types
        N = len(cell_type_list)  # Number of cells
        M = len(np.unique(cell_type_list))  # Number of cell types

        # Create matrix (NxM) from cell type
        mat_cell_type = np.zeros((N, M))
        for i in range(N):
            mat_cell_type[i, cell_type_list[i]] = 1

        cell_class_list = hmrf.categorical_vector(self.graph, "class")  # List of labels

        classes, card_classes = np.unique(
            cell_class_list, return_counts=True
        )  # Number of cell labels

        # Little trick to allow loosing some labels
        card_classes2 = np.zeros(self.K)
        card_classes2[classes] = card_classes
        card_classes = card_classes2

        # Count frequencies of cell types in each latent class
        mu = np.zeros((self.K, M))

        # Variability inside class
        sig = [np.eye(M) for j in range(self.K)]

        for j in range(self.K):

            # Count number of cells of each type in class j
            x1, x2 = np.unique(cell_type_list[cell_class_list == j], return_counts=True)
            mu[j, x1] = x2
            mu[j, :] /= card_classes[j]
            mu[j, :] /= np.sum(mu[j, :])

            # Compute variability inside class j
            cell_type_in_j = mat_cell_type[cell_class_list == j]
            sig_j = np.sum((cell_type_in_j - mu[j]) ** 2, axis=0)
            sig_j /= card_classes[j]
            sig[j] *= sig_j

        return mu, sig

    def run(self):  # run the loop self.max_it times
        list_param = [[self.mu, self.sigma2]]

        for cpt in tqdm(range(self.max_it)):

            self.update_labels()
            self.mu, self.sigma2 = self.update_parameters()
            list_param.append([self.mu, self.sigma2])

        self.parameters = list_param  # save all values of parameters

    # ---------------------------------------------------------------------------------------------------------------

    @staticmethod
    def categorical_vector(G, category):
        # input: G (NetworkX graph)
        #        category (str): name of a node attribute
        # return: a numpy.array vector containing the values of the 'category' node attribute for each node

        cat = nx.get_node_attributes(G, category)
        type_of_data = type(cat[0])
        V = np.array(list(cat.items()), dtype=type_of_data)
        a = map(int, V[:, 0])
        a = np.array(list(a))
        ind = np.argsort(a)
        Vect = V[:, 1][ind]

        return Vect

    @staticmethod
    def get_compo_nn_properties(G, number_of_cell_types):
        # input: G (NetworkX graph)
        #        number_of_cell_types (int)
        # return: a pandas.DataFrame containing the composition of nearest neighbors phenotypes (in percent) for each node

        compo = hmrf.categorical_vector(G, "compo_nn")

        resultframe = pandas.DataFrame(
            np.concatenate([L for L in compo]).reshape(
                (len(compo), number_of_cell_types)
            )
        )

        return resultframe.fillna(0)
