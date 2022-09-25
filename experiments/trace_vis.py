from macq.generate.pddl import Generator
from macq.trace import Step, Trace
from macq.generate.pddl import StateEnumerator
from tarski.search.operations import progress
from pddl_vis.visualizers import GridVisualizer
import networkx as nx
from PIL import Image
from io import BytesIO



domain_file = 'data/pddl/grid.pddl'
problem_file = 'data/pddl/grid_data.pddl'


actions = ['(move node0-0 node0-1)',
        '(move node0-1 node1-1)',
        '(putdown node1-1 key0)',
        '(move node1-1 node1-0)',
        '(move node1-0 node0-0)',
        '(move node0-0 node0-1)',
        '(move node0-1 node1-1)',
        '(pickup node1-1 key0)',
        '(move node1-1 node1-0)',
        ]

generator = Generator(dom=domain_file, prob=problem_file)
vis = GridVisualizer(generator, square_width=50, div_width=1, door_width=6, key_size=15, robot_size=17)

graph_generator = StateEnumerator(dom=domain_file, prob=problem_file)

states = list(graph_generator.graph.nodes())
state_mapping = dict(zip(states, range(len(states))))
state_graph = nx.relabel_nodes(graph_generator.graph, state_mapping)

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
    #dot_graph.add_node(pydot.Node("img", label="", image='results/legend.png'))
    img = Image.open(BytesIO(dot_graph.create_png()))
    return img


state = generator.problem.init
graph_node = graph_generator.problem.init
trace = Trace()

graph_imgs = []

for i, action in enumerate(actions):
    graph_imgs.append(colour_state(state, state_graph))
    act = generator.op_dict[action]
    macq_state = generator.tarski_state_to_macq(state)
    macq_action = generator.tarski_act_to_macq(act)
    step = Step(macq_state, macq_action, i)
    state = progress(state, act)
    trace.append(step)
graph_imgs.append(colour_state(state, state_graph))
macq_state = generator.tarski_state_to_macq(state)
step = Step(macq_state, None, len(actions))
trace.append(step)

graph_path = 'results/gifs/graph.gif'
print("Saving gif of graph at " + graph_path + "\n")
graph_imgs[0].save(graph_path, save_all=True, append_images=graph_imgs[1:], duration=1000, loop=0)
state_imgs = vis.visualize_trace(trace, size=(652,731))

imgs = []
for graph, state in zip(graph_imgs, state_imgs):
    assert graph.height == state.height
    dst = Image.new('RGB', (state.width + graph.width, state.height))
    dst.paste(state, (0, 0))
    dst.paste(graph, (state.width, 0))
    imgs.append(dst)

state_graph_path = 'results/gifs/state_graph.gif'
print("Saving gif of state visualization and graph at " + state_graph_path + "\n")
imgs[0].save(state_graph_path, save_all=True, append_images=imgs[1:], duration=1000, loop=0)