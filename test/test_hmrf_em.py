import networkx as nx
import numpy as np
import pytest
from biograph import hmrf_em


@pytest.fixture
def create_graph():
    """Create a graph with five nodes and ten edges. Each node is
    connected to two other nodes. Each node has one attribute 'cell_type',
    there are two different 'cell_type' attributes in total.

    Returns
    -------
    G : networkx.Graph
    """
    G = nx.Graph()
    G.add_nodes_from(range(5))
    G.add_edges_from([(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)])
    G.nodes[0]["cell_type"] = 0
    G.nodes[1]["cell_type"] = 0
    G.nodes[2]["cell_type"] = 1
    G.nodes[3]["cell_type"] = 1
    G.nodes[4]["cell_type"] = 1
    return G


def test_generate_hmrf_instance(create_graph):

    assert isinstance(create_graph, nx.Graph)
    assert len(create_graph.nodes) == 5
    assert len(create_graph.edges) == 5

    hmrf_instance = hmrf_em.hmrf(create_graph, K=2, beta=10, max_it=30)
    hmrf_instance.initiate_model()

    assert hmrf_instance.K == 2
    assert hmrf_instance.beta == 10
    assert hmrf_instance.max_it == 30
    assert hmrf_instance.number_of_cell_types == 2


def test_hmrf_run(create_graph):
    """Create a hmrf instance, then run the instance over 10 iterations.
    Verify that the output object is a graph.
    """

    hmrf_instance = hmrf_em.hmrf(create_graph, K=2, beta=10, max_it=30)
    hmrf_instance.initiate_model()
    hmrf_instance.run()

    assert isinstance(hmrf_instance.graph, nx.Graph)
    assert len(hmrf_instance.graph.nodes) == 5
    assert len(hmrf_instance.graph.edges) == 5
