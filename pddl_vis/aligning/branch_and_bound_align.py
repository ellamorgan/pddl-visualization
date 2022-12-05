import numpy as np
import networkx as nx
import torch
import math
from macq.generate.pddl import StateEnumerator, VanillaSampling
from pddl_vis.utils import graph_to_img, colour_state
from pddl_vis.dataset import visualize_traces
from pddl_vis.aligning import get_graph_and_traces, get_trace_predictions
import matplotlib.pyplot as plt


'''
input = [s1, s2, s3, s4]
Pr(s,*) = 1.0
score([n1,n2,n3,n4]) = Pr(s1,n1) * Pr(s2,n2) * Pr(s3,n3) * Pr(s4,n4)
score(null) = 0
best = null
for n1 in G, sorted decending by Pr(s1, n1):
  if score([n1, *, *, *]) < score (best):
    return best
  for n2 in succ(n1), sorted ... by Pr(s2, n2):
    if score([n1, n2, *, *]) < score (best):
      break
    for n3 in succ(n2), sorted ... by Pr(s3, n3):
      if score([n1, n2, n3, *]) < score (best):
        break
      n4 = max(over succ(n3)){Pr(s4,n4)}
      if best == null:
        best = [n1, n2, n3, n4]
        continue
'''

best_score = 0
best_seq = []


def score(seq, logits):
    if len(seq) == 0:
        return 0
    return np.prod([math.e ** logits[i] for i in seq])


def find_next(seq, state_graph, trace_logits, step, best_score, best_seq):

    next_nodes = list(state_graph.neighbors(seq[-1]))
    next_logits = trace_logits[step][next_nodes]
    sorted_inds = np.argsort(-1 * next_logits)

    states = np.array(list(range(len(trace_logits[step]))))
    next_states = states[next_nodes]
    # All connected to previous, sorted by highest probability
    sorted_states = next_states[sorted_inds]

    if step == len(trace_logits) - 1:
        curr_score = score(seq + [sorted_states[0]], trace_logits[step])
        if curr_score > best_score:
            best_score = curr_score
            best_seq = seq + [sorted_states[0]]
    else:
        for state in sorted_states:
            if score(seq + [state], trace_logits[step]) < best_score:
                break
            best_score, best_seq = find_next(
                seq + [state], state_graph, trace_logits, step + 1, best_score, best_seq)

    return best_score, best_seq


def branch_and_bound_align(state_graph, trace_states, pred_logits):

    seq_preds = []
    top_1_preds = []

    for trace_logits in pred_logits:
        # trace is (trace_len, n_states)

        top_1_preds.append(list(map(np.argmax, trace_logits)))

        best_score = 0
        best_seq = []

        # First node
        for node in np.argsort(-1 * trace_logits[0]):

            if score([node], trace_logits[0]) < best_score:
                break
            else:
                best_score, best_seq = find_next(
                    [node], state_graph, trace_logits, 1, best_score, best_seq)

        seq_preds.append(best_seq)

    top_1_accuracy = 100 * np.sum(top_1_preds == trace_states) / trace_states.size
    bnb_accuracy = 100 * np.sum(seq_preds == trace_states) / trace_states.size

    print(f"Top 1 preds accuracy: {top_1_accuracy}%")
    print(f"Branch and bound algorithm accuracy: {bnb_accuracy}%")

    return top_1_accuracy, bnb_accuracy
