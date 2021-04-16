import numpy as np
import pandas
import networkx as nx

def get_length_per_edge(G):
    
    pos = nx.get_node_attributes(G, 'pos')
    
    for edge in list(G.edges()):
        
        G[edge[0]][edge[1]]['length'] = np.sqrt((pos[edge[0]][0] - pos[edge[1]][0])**2 + (pos[edge[0]][1] - pos[edge[1]][1])**2 + (pos[edge[0]][2] -pos[edge[1]][2])**2)                
            
    return G

def get_legend_per_edge(G):
    
    legend = nx.get_node_attributes(G, 'legend')
    
    for edge in list(G.edges()):
                
        G[edge[0]][edge[1]]['legend'] = (legend[edge[0]],legend[edge[1]])
    
    return G


def get_edge_properties(G,
                        edge_descriptors = ['length', 'legend']):
    
    """
    edge_descriptors: ['length', 'legend']

    """
    
    if 'length' in edge_descriptors:

        assert 'pos' in list(G.nodes.data()[0].keys())
        
        G = get_length_per_edge(G)
    

    if 'legend' in edge_descriptors:

        assert 'legend' in list(G.nodes.data()[0].keys())
        
        G = get_legend_per_edge(G)
    
    return G