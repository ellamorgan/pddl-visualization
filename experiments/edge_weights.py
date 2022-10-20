import networkx as nx
import numpy as np
import torch
from PIL import Image as Img
from io import BytesIO
from colorsys import hsv_to_rgb

from macq.generate.pddl import StateEnumerator, VanillaSampling
from macq.trace import TraceList

from pddl_vis.visualizers import VISUALIZERS





def learn_edges(model, domain_file, problem_file, name, vis_args, img_size, n_states):

    # Create graph
    graph_generator = StateEnumerator(
        dom=domain_file, 
        prob=problem_file
    )
    states = list(graph_generator.graph.nodes())
    state_mapping = dict(zip(states, range(len(states))))
    original_state_graph = nx.relabel_nodes(graph_generator.graph, state_mapping)
    state_graph = nx.create_empty_copy(original_state_graph)


    # Generate traces
    trace_generator = VanillaSampling(
        dom=domain_file, 
        prob=problem_file,
        plan_len=100,
        num_traces=20
    )

    vis = VISUALIZERS[name](graph_generator, **vis_args).visualize_state

    # Turn traces into batches
    # The issue: don't have labels since VanillaSampling and StateEnumerator are different objects
    def process_img(img, size):
        img = img.resize(size)
        array_from_img = np.asarray(img).transpose(2, 0, 1)
        normalized = (array_from_img / 127.5) - 1
        return normalized

    state_vis = []
    for trace in trace_generator.traces:
        state_vis.append([])
        for state in trace:
            state_vis[-1].append(process_img(vis(state), img_size))
    state_vis = np.array(state_vis)

    counts = np.zeros((n_states, n_states))

    for batch in state_vis:
        x = torch.tensor(batch).float()
        logits = model(x)['logits'].detach().numpy()
        preds = np.array(list(map(np.argmax, logits)))

        for i in range(len(preds) - 1):
            counts[preds[i]][preds[i+1]] += 1

    blue_h, red_h = 212 / 360, 352 / 360

    counts /= np.sum(counts)
    counts /= np.max(counts)
    
    for i in range(n_states):
        for j in range(n_states):

            if original_state_graph.has_edge(i, j):
                if counts[i][j] < 0.1:
                    r, g, b = 110 / 255, 1, 134 / 255
                else:
                    (r, g, b) = hsv_to_rgb(blue_h, counts[i][j], 1)
            else:
                (r, g, b) = hsv_to_rgb(red_h, counts[i][j], 1)
            
            if counts[i][j] > 0:
                hex = ['{:X}'.format(int(255 * r)), '{:X}'.format(int(255 * g)), '{:X}'.format(int(255 * b))]
                colour = "#"
                for c in hex:
                    if len(c) == 1:
                        colour += '0' + c
                    else:
                        colour += c
                state_graph.add_edge(i, j, weight=counts[i][j], color=colour)

    dot_graph = nx.nx_pydot.to_pydot(state_graph)
    img = Img.open(BytesIO(dot_graph.create_png()))
    img.save("results/learned_edges.png")