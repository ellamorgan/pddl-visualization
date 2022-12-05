from macq.generate.pddl import Generator
from macq.trace import Step, Trace
from macq.generate.pddl import StateEnumerator, VanillaSampling
from tarski.search.operations import progress
from pddl_vis.visualizers import VISUALIZERS
import networkx as nx
from PIL import Image
import PIL.ImageOps
from io import BytesIO

domain = 'grid'
problem = 'grid1'

domain_file = 'data/pddl/' + domain + '/' + domain + '.pddl'
problem_file = 'data/pddl/' + domain + '/problems/' + problem + '.pddl'

generator = VanillaSampling(
    dom=domain_file, 
    prob=problem_file,
    plan_len=3,
    num_traces=1
)

vis = VISUALIZERS[domain](generator)

graph_generator = StateEnumerator(dom=domain_file, prob=problem_file)

states = list(graph_generator.graph.nodes())
state_mapping = dict(zip(states, range(len(states))))
state_graph = nx.relabel_nodes(graph_generator.graph, state_mapping)

macq_states = list(map(graph_generator.tarski_state_to_macq, states))
state_hashes = list(map(hash, map(str, macq_states)))


# Used to remove all the action parameters
for _, _, act in state_graph.edges(data=True):
    act['label'] = ""


def colour_state(state, graph, colour='#bababa'):
    graph.nodes[state]['style'] = 'filled'
    graph.nodes[state]['fillcolor'] = colour
    return graph

def graph_to_img(graph):
    dot_graph = nx.nx_pydot.to_pydot(graph)
    dot_graph.dpi = 500
    img = Image.open(BytesIO(dot_graph.create_png()))
    return img

state_imgs = []
graph_imgs = []
states = []
graph_vis = state_graph.copy()
prev_state = None
for step in generator.traces[0]:

    states.append(state_hashes.index(hash(str(step.state))))
    img = PIL.ImageOps.invert(vis.visualize_state(step))
    border = 10
    border_img = Image.new("RGB", (img.width + 2 * border, img.height + border), "White")
    border_img.paste(img, (border, border))
    state_imgs.append(border_img)
    colour_state(states[-1], graph_vis, colour='#66e055')

    if prev_state is not None:
        graph_vis[prev_state][states[-1]]['color'] = '#66e055'
        graph_vis[prev_state][states[-1]]['width'] = 30
    prev_state = states[-1]
    graph_imgs.append(graph_to_img(graph_vis))

g_h = graph_imgs[0].height
s_h = state_imgs[0].height
ratio = g_h / s_h

state_imgs = [img.resize((int(state_imgs[0].width * ratio), graph_imgs[0].height)) for img in state_imgs]

dst = Image.new('RGB', (3 * (state_imgs[0].width + graph_imgs[0].width) + 2 * border, graph_imgs[0].height), "White")

for i, (state_img, graph_img) in enumerate(zip(state_imgs, graph_imgs)):
    dst.paste(state_img, (i * (state_imgs[0].width + graph_imgs[0].width + border), 0))
    dst.paste(graph_img, (i * (state_imgs[0].width + graph_imgs[0].width + border) + state_imgs[0].width, 0))
dst.save("results/figure.jpg")