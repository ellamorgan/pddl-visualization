import networkx as nx
from PIL import Image as Img
from io import BytesIO




def predict_vis_trace(batch, generator, model):

    X, targets = batch

    states = list(generator.graph.nodes())
    state_mapping = dict(zip(states, range(len(states))))
    state_graph = nx.relabel_nodes(generator.graph, state_mapping)

    # Used to remove all the action parameters
    for _, _, act in state_graph.edges(data=True):
        action = str(act['label']).split('(')[0]
        if action == 'pickup':
            act['color'] = "#7b4af7"
            act['label'] = "  pickup  "
        elif action == 'putdown':
            act['color'] = "#4a81f7"
            act['label'] = "  putdown  "
        elif action == 'move':
            act['color'] = "#83e070"
            act['label'] = "  move  "
        act['label'] = ""

    def colour_state(state, graph):
        graph = graph.copy()
        graph.nodes[state_mapping[state]]['style'] = 'filled'
        graph.nodes[state_mapping[state]]['fillcolor'] = '#bababa'
        dot_graph = nx.nx_pydot.to_pydot(graph)
        img = Img.open(BytesIO(dot_graph.create_png()))
        return img
    

    out = model(X)
    print(out.shape)