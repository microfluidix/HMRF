import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas

def scatter_plot_2D(G,
                    include_color:bool = False,
                    edge_color:str = 'k', 
                    save:bool=False,
                    figsize:tuple = (8,8),
                    alpha_line = 0.6,
                    dim_to_squeeze = 'z',
                    scatterpoint_size = 20,
                    legend = False):

    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')

    # Define color range proportional to number of edges adjacent to a single node
    colors = nx.get_node_attributes(G, 'color')

    if legend:

        legend = nx.get_node_attributes(G, 'legend')

    fig, ax = plt.subplots(figsize = figsize)

    # Loop on the pos dictionary to extract the x,y,z coordinates of each node

    x = []
    y = []
    nodeColor = []
    s = []
    nodelegend = []

    for key, value in pos.items():
        x.append(value[1])
        y.append(value[2])
        s.append(scatterpoint_size)
        nodeColor.append(colors[key])

        if legend:
            nodelegend.append(legend[key])

    df = pandas.DataFrame()
    df['x'] = x
    df['y'] = y
    df['s'] = s
    df['nodeColor'] = nodeColor

    if legend:
        df['legend'] = nodelegend

    groups = df.groupby('nodeColor')

    for nodeColor, group in groups:

        if legend:

            name = group.legend.unique()[0]

            ax.plot(group.x, group.y, 
              marker='o', 
              c=nodeColor,
              markeredgewidth=1.5, 
              markeredgecolor= 'k',
              linestyle='', 
              ms=scatterpoint_size, 
              label=name)

            ax.legend()

        else:

            ax.plot(group.x, group.y, 
              marker='o', 
              c=nodeColor,
              markeredgewidth=1.5, 
              markeredgecolor= 'k',
              linestyle='', 
              ms=scatterpoint_size)

    # Scatter plot
    #sc = ax.scatter(x, y, c=nodeColor, s=s, edgecolors='k', alpha=1)
    
    # Remove tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # No ticks
    ax.set_xticks([]) 
    ax.set_yticks([]) 