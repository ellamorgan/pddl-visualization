import numpy as np
import networkx as nx
import torch
import math
from macq.generate.pddl import StateEnumerator, VanillaSampling
from pddl_vis.utils import graph_to_img, colour_state
from pddl_vis.dataset import visualize_traces
from pddl_vis.aligning import get_graph_and_traces, get_trace_predictions
import matplotlib.pyplot as plt



def find_next(seq, score, trace_logits, state_graph, best_score, best_seq):

    next_nodes = list(state_graph.neighbors(seq[-1]))
    next_logits = trace_logits[len(seq)][next_nodes]

    sorted_logit_inds = np.argsort(-1 * next_logits)

    last_node = True if len(seq) == len(trace_logits) - 1 else False

    for ind in sorted_logit_inds:
        node = next_nodes[ind]
        prob = math.e ** next_logits[ind]

        new_score = score * prob

        if new_score >= best_score and not last_node:
            best_score, best_seq = find_next(seq + [node], new_score, trace_logits, state_graph, best_score, best_seq)
        if last_node and new_score > best_score:
            best_score = new_score
            best_seq = seq + [node]

    return best_score, best_seq



def branch_and_bound_align(state_graph, states, preds, logits, top_n):

    bnb_preds = []

    for trace_preds, trace_logits in zip(preds, logits):

        best_score = 0
        best_seq = []

        for first_node in trace_preds[0][:top_n]:

            score = math.e ** trace_logits[0][first_node]   # This doesn't assume trace_logits is sorted

            score, seq = find_next([first_node], score, trace_logits, state_graph, best_score, best_seq)

            if score > best_score:
                best_score = score
                best_seq = seq
        
        bnb_preds.append(best_seq)
    
    print(np.array(bnb_preds).shape)
    
    bnb_accuracy = 100 * np.sum(np.array(bnb_preds) == states) / states.size

    print()
    print(f"BnB accuracy:  {bnb_accuracy:.2f}%")
    print()

    return bnb_accuracy



# Try with trace of random states