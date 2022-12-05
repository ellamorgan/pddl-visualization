import numpy as np
import networkx as nx
import torch
from macq.generate.pddl import StateEnumerator, VanillaSampling
from pddl_vis.utils import graph_to_img, colour_state
from pddl_vis.dataset import visualize_trace
from pddl_vis.aligning import get_graph_and_traces, get_predictions
import matplotlib.pyplot as plt


def broken_path_align(model, domain_file, problem_file, n_data, batch_size, vis, img_size, top_n):

    state_graph, traces, trace_states = get_graph_and_traces(domain_file, problem_file, n_traces=1, trace_len=n_data)
    trace = traces[0]
    trace_states = trace_states[0]
    
    data = visualize_trace(trace, vis, img_size)

    preds, pred_logits = get_predictions(model, data, batch_size, top_n)

    found = []

    # Check if top-1 predictions in graph
    for pre, suc in zip(preds[:-1], preds[1:]):
        if state_graph.has_edge(pre[0], suc[0]):
            found.append(1)
        else:
            found.append(0)
    
    print(f"{100 * sum(found) / len(found):.2f}\% of edges found")

    broken_counter = 0
    broken_sequences = []
    for edge in found:
        if not edge:
            broken_counter += 1
        else:
            if broken_counter > 0:
                broken_sequences.append(broken_counter)
                broken_counter = 0
    
    plt.clf()
    plt.hist(broken_sequences, edgecolor='black')
    plt.xlabel('Seq len')
    plt.ylabel('Count')
    plt.savefig('results/hist_test.jpg')
