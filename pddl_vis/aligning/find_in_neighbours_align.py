import numpy as np
import networkx as nx
import torch
from macq.generate.pddl import StateEnumerator, VanillaSampling
from pddl_vis.utils import graph_to_img, colour_state
from pddl_vis.dataset import visualize_traces
from pddl_vis.aligning import get_graph_and_traces, get_trace_predictions
import matplotlib.pyplot as plt

#sort_inds = np.array(list(map(np.argsort, trace)))
#log_probs = np.array([trace[i][sort_inds[i]] for i in range(len(trace))])


def find_in_neighbours_align(model, domain_file, problem_file, n_traces, trace_len, batch_size, vis, img_size):

    # Get one long trace to break into trace_len sizes
    state_graph, traces, trace_states = get_graph_and_traces(domain_file, problem_file, n_traces=1, trace_len=n_traces * trace_len)
    
    # (1, trace_len * n_traces, *img_shape)
    data = visualize_traces(traces, vis, img_size)

    pred_logits = get_trace_predictions(model, data, batch_size)

    pred_logits = pred_logits.reshape((n_traces, trace_len, *pred_logits.shape[2:]))    # (n_traces, trace_len, n_states)

    max_neighbours = []
    top_1 = []
    for trace_logits in pred_logits:
        max_neighbours.append([])
        top_1.append([])
        curr_state = np.argmax(trace_logits[0])
        max_neighbours[-1].append(curr_state)
        top_1[-1].append(curr_state)
        for i in range(1, len(trace_logits)):
            next_states = state_graph.neighbors(curr_state)
            neighbour_probs = trace_logits[i][list(next_states)]
            curr_state = np.where(trace_logits[i] == max(neighbour_probs))[0][0]
            max_neighbours[-1].append(curr_state)
            top_1[-1].append(np.argmax(trace_logits[i]))
    max_neighbours = np.array(max_neighbours).flatten()
    top_1 = np.array(top_1).flatten()
    print(f"Max neighbours: {100 * sum(max_neighbours == np.array(trace_states).flatten()) / len(max_neighbours):.2f}\%")
    print(f"Top-1: {100 * sum(top_1 == np.array(trace_states).flatten()) / len(top_1):.2f}\%")
