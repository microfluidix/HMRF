import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas


def scatter_plot_2D(
    G,
    edge_color: str = "k",
    figsize: tuple = (8, 8),
    dim_to_squeeze="z",
    scatterpoint_size=20,
    legend=False,
    lims=None,
):

    # Get node positions
    pos = nx.get_node_attributes(G, "pos")

    colors = nx.get_node_attributes(G, "color")

    assert colors != {}

    if legend:

        legend = nx.get_node_attributes(G, "legend")

    fig, ax = plt.subplots(figsize=figsize)

    # Loop on the pos dictionary to extract the x,y,z coordinates of each node

    x = []
    y = []
    nodeColor = []
    s = []
    nodelegend = []

    if dim_to_squeeze == "z":

        for key, value in pos.items():
            x.append(value[1])
            y.append(value[2])
            s.append(scatterpoint_size)
            nodeColor.append(colors[key])

            if legend:
                nodelegend.append(legend[key])

    elif dim_to_squeeze == "x":

        for key, value in pos.items():
            x.append(value[0])
            y.append(value[2])
            s.append(scatterpoint_size)
            nodeColor.append(colors[key])

            if legend:
                nodelegend.append(legend[key])

    else:

        for key, value in pos.items():
            x.append(value[0])
            y.append(value[1])
            s.append(scatterpoint_size)
            nodeColor.append(colors[key])

            if legend:
                nodelegend.append(legend[key])

    df = pandas.DataFrame()
    df["x"] = x
    df["y"] = y
    df["s"] = s
    df["nodeColor"] = nodeColor

    if legend:
        df["legend"] = nodelegend

    groups = df.groupby("nodeColor")

    for nodeColor, group in groups:

        if legend:

            name = group.legend.unique()[0]

            ax.plot(
                group.x,
                group.y,
                marker="o",
                c=nodeColor,
                markeredgewidth=1.5,
                markeredgecolor=edge_color,
                linestyle="",
                ms=scatterpoint_size,
                label=name,
            )

            ax.legend()

        else:

            ax.plot(
                group.x,
                group.y,
                marker="o",
                c=nodeColor,
                markeredgewidth=1.5,
                markeredgecolor=edge_color,
                linestyle="",
                ms=scatterpoint_size,
            )

    # Scatter plot
    # sc = ax.scatter(x, y, c=nodeColor, s=s, edgecolors='k', alpha=1)

    # Remove tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # No ticks
    ax.set_xticks([])
    ax.set_yticks([])

    if lims:

        xlims, ylims = lims

        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
