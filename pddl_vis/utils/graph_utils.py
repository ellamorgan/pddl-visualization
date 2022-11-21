from PIL import Image
from io import BytesIO
import networkx as nx


def graph_to_img(graph):
    dot_graph = nx.nx_pydot.to_pydot(graph)
    img = Image.open(BytesIO(dot_graph.create_png()))
    return img


def colour_state(state, graph, colour='#bababa'):
    graph.nodes[state]['style'] = 'filled'
    graph.nodes[state]['fillcolor'] = colour
    return graph