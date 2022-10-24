import networkx as nx
import numpy as np
import torch
from PIL import Image as Img
from io import BytesIO
import math
from colorsys import hsv_to_rgb
import matplotlib.pyplot as plt

from macq.generate.pddl import StateEnumerator, VanillaSampling
from macq.trace import TraceList

from pddl_vis.visualizers import VISUALIZERS





def learn_edges(model, domain_file, problem_file, name, vis_args, img_size, n_states, plan_len, num_traces):

    # Create graph
    graph_generator = StateEnumerator(
        dom=domain_file, 
        prob=problem_file
    )
    states = list(graph_generator.graph.nodes())
    macq_states = list(map(graph_generator.tarski_state_to_macq, states))
    state_mapping = dict(zip(states, range(len(states))))
    original_state_graph = nx.relabel_nodes(graph_generator.graph, state_mapping)
    pos = nx.spring_layout(original_state_graph)
    state_graph = nx.create_empty_copy(original_state_graph)


    # Used to remove all the action parameters
    for _, _, act in original_state_graph.edges(data=True):
        action = str(act['label']).split('(')[0]
        if action == 'pickup':
            act['color'] = "#7b4af7"
            act['label'] = ""
        elif action == 'putdown':
            act['color'] = "#4a81f7"
            act['label'] = ""
        elif action == 'move':
            act['color'] = "#83e070"
            act['label'] = ""
        act['label'] = ""


    # Generate traces
    trace_generator = VanillaSampling(
        dom=domain_file, 
        prob=problem_file,
        plan_len=plan_len,
        num_traces=num_traces
    )
    batch_size = 100

    vis = VISUALIZERS[name](graph_generator, **vis_args).visualize_state

    # Turn traces into batches
    # The issue: don't have labels since VanillaSampling and StateEnumerator are different objects
    def process_img(img, size):
        img = img.resize(size)
        array_from_img = np.asarray(img).transpose(2, 0, 1)
        normalized = (array_from_img / 127.5) - 1
        return normalized

    # Get visualizations
    state_vis = []
    for trace in trace_generator.traces:
        state_vis.append([])
        for state in trace:
            state_vis[-1].append(process_img(vis(state), img_size))
    state_vis = np.array(state_vis)

    counts = np.zeros((n_states, n_states))

    # Put traces through model, breaking into smaller batches if batch size is smaller than trace len
    for trace in state_vis:
        trace_preds = []
        for batch_ind in range(math.ceil(len(trace) / batch_size)):
            batch = trace[batch_ind * batch_size : (batch_ind + 1) * batch_size]
            x = torch.tensor(batch).float()
            logits = model(x)['logits'].detach().numpy()
            trace_preds += list(map(np.argmax, logits))
        preds = np.array(trace_preds)
        for i in range(len(preds) - 1):
            counts[preds[i]][preds[i+1]] += 1

    blue_h, red_h = 212 / 360, 352 / 360

    counts /= np.sum(counts)
    counts /= np.max(counts)

    n_correct = 0
    sum_correct = 0
    n_incorrect = 0
    sum_incorrect = 0
    n_missing = 0

    dataset = []
    
    # Colour edges, blue if in original graph, red if not, green if missing
    for i in range(n_states):
        for j in range(n_states):

            has_edge = original_state_graph.has_edge(i, j)

            if has_edge:
                if counts[i][j] < 0.1:
                    r, g, b = 59 / 255, 245 / 255, 42 / 255
                    n_missing += 1
                else:
                    (r, g, b) = hsv_to_rgb(blue_h, counts[i][j], 1)
                    n_correct += 1
                    sum_correct += counts[i][j]
            else:
                (r, g, b) = hsv_to_rgb(red_h, counts[i][j], 1)
                if counts[i][j] >= 0.1:
                    n_incorrect += 1
                    sum_incorrect += counts[i][j]
            
            if counts[i][j] > 0 or has_edge:
                hex = ['{:X}'.format(int(255 * r)), '{:X}'.format(int(255 * g)), '{:X}'.format(int(255 * b))]
                colour = "#"
                for c in hex:
                    if len(c) == 1:
                        colour += '0' + c
                    else:
                        colour += c
                state_graph.add_edge(i, j, weight=counts[i][j], color=colour)
        
            dataset.append([i, j, counts[i][j]])
    
    # Save original graph
    dot_graph = nx.nx_pydot.to_pydot(original_state_graph)
    pos = nx.nx_pydot.pydot_layout(original_state_graph)
    img = Img.open(BytesIO(dot_graph.create_png()))
    img.save("results/original_graph.png")

    #nx.draw(original_state_graph, pos)
    #plt.show()
    #plt.savefig("orig_plt.png", format="PNG")

    # Save graph with learned edges
    nx.set_node_attributes(state_graph, pos, 'coord')
    dot_graph = nx.nx_pydot.to_pydot(state_graph)
    img = Img.open(BytesIO(dot_graph.create_png()))
    img.save("results/learned_edges.png")

    #nx.draw(state_graph, pos)
    #plt.show()
    #plt.savefig("learned_plt.png", format="PNG")

    print(f"Correct: {n_correct} Weighted: {sum_correct / max(n_correct, 1)}")
    print(f"Incorrect: {n_incorrect} Weighted: {sum_incorrect / max(n_incorrect, 1)}")
    print(f"Missing: {n_missing}")

    return dataset, macq_states