from macq.generate.pddl import StateEnumerator
import networkx as nx
from PIL import Image
from io import BytesIO

domain = 'domain.pddl'
problem = 'problem.pddl'

generator = StateEnumerator(dom=domain, prob=problem)


# Get networkx graph
graph = generator.graph

# Get states
states = list(graph.nodes())

# Relabel the states with numbers so it can be visualized
state_mapping = dict(zip(states, range(len(states))))
state_graph = nx.relabel_nodes(graph, state_mapping)

# Colour edges based on the action, and relabel (action labels are too long otherwise)
actions = []
colours = ['#72EC91', '#C07DFF', '#4D7DEE', '#FF5A5F', '#C81D25']       # Can create colour palettes here: https://coolors.co/

for _, _, act in state_graph.edges(data=True):
    action = str(act['label']).split('(')[0]
    if action in actions:
        act['color'] = colours[actions.index(action)]
    else:
        actions.append(action)
        if len(actions) > len(colours):
            print('Need more colours')
            exit()
        act['color'] = colours[len(actions) - 1]
    act['label'] = '  ' + action + '  '

# Saving the image, bypasses having to save the graph as a dot file first
dot_graph = nx.nx_pydot.to_pydot(state_graph)
img = Image.open(BytesIO(dot_graph.create_png())).convert('RGB')
img.save('state_space.jpg')
