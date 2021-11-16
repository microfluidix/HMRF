import networkx as nx
import numpy as np
from tqdm import tqdm


class hmrf:

    """

    Create an instance of the hmrf class.

    Parameters:
     - self.graph: networkx object containing
    """

    def __init__(self, G, gamma=5, epochs=50):

        self.graph = G
        self.epochs = epochs
        self.gamma = gamma
        self.node_attributes = np.unique(
            np.array(
                [list(self.graph.nodes[n].keys()) for n in self.graph.nodes()]
            ).flatten()
        )

    def initiate_latent_cell_type(self):

        # check that 'cell_type' is defined.
        # Ideally should be done for each node

        assert "cell_type" in self.node_attributes

        self.graph = self.graph

        cell_type_dict = nx.get_node_attributes(self.graph, "cell_type")
        nx.set_node_attributes(self.graph, cell_type_dict, "latent_cell_type")

        self.graph = self.graph

    def run(self):

        for i in tqdm(range(self.epochs)):

            self.update_latent_celltype()

    def energy(self, cell_type, new_lat, latent_cell_type_props):

        """

        This formula for the energy is obtained by calculating the
        the posterior probability in the Gibbs sampling approach with
        a neighborhood constraint (see Potts model) and with an
        observable constraint.

        """

        return self.gamma * int(cell_type == new_lat) + np.sum(
            [int(lat_neighbour == new_lat) for lat_neighbour in latent_cell_type_props]
        )

    def update_latent_celltype(self):

        """
        Single update step of the latent network.
        """

        latent_cell_type_dict = nx.get_node_attributes(self.graph, "latent_cell_type")
        cell_type_dict = nx.get_node_attributes(self.graph, "cell_type")

        G = self.graph

        for node in self.graph.nodes:

            latent_cell_type_props = [
                latent_cell_type_dict[n] for n in G.neighbors(node)
            ]
            cell_type = cell_type_dict[node]
            latent_cell_type = latent_cell_type_dict[node]

            cell_type_possibilities = list(np.unique(latent_cell_type_props))
            cell_type_possibilities.append(cell_type)

            energies = [
                self.energy(cell_type, new_lat, latent_cell_type_props)
                for new_lat in cell_type_possibilities
            ]

            cell_type_z_t = cell_type_possibilities[np.argmax(energies)]

            nx.set_node_attributes(G, {node: cell_type_z_t}, "latent_cell_type")

        self.graph = G
