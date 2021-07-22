import numpy as np
import pandas
import networkx as nx


def node_edge_properties(G):

    for node in G.nodes():

        length_list = []
        legend_dict = {}

        for neighbour in G[node]:

            length_list.append(G[node][neighbour]["length"])

            if G[node][neighbour]["legend"] in legend_dict:
                legend_dict[G[node][neighbour]["legend"]] += 1
            else:
                legend_dict[G[node][neighbour]["legend"]] = 1

        G.nodes[node]["neighbour_length"] = np.mean(length_list)
        G.nodes[node]["neighbour_legend"] = legend_dict
        G.nodes[node]["degree"] = len(G[node])

    return G
