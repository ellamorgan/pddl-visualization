import numpy as np
from pddl_vis.utils import graph_to_img, colour_state
from pddl_vis.dataset import visualize_trace
from pddl_vis.aligning import get_graph_and_traces, get_predictions




def find_next(ind, preds, pred_selected, state_graph, top_n):
    # Pred states is (states, top_n) and sorted by highest -> lowest probability
    for state in preds[ind + 1][:top_n]:
        if state_graph.has_edge(pred_selected[ind], state):
            pred_selected[ind + 1] = state
            break
    return pred_selected




def find_edge(ind, preds, pred_logits, pred_selected, state_graph, top_n):

    best = -1000
    curr_best = -1
    next_best = -1
    found = False

    for i, curr_state in enumerate(preds[ind][:top_n]):

        if found and pred_logits[ind][i] + pred_logits[ind + 1][0] < best:
            break

        for j, next_state in enumerate(preds[ind + 1][:top_n]):
            if found and pred_logits[ind][i] + pred_logits[ind + 1][j] < best:
                break

            if state_graph.has_edge(curr_state, next_state) and pred_logits[ind][i] + pred_logits[ind + 1][j] > best:
                best = pred_logits[ind][i] + pred_logits[ind + 1][j]
                curr_best = curr_state
                next_best = next_state
                found = True
                break
    
    if not found:
        # If not found, set first to top-1 then find_edge will get run on next two states
        pred_selected[ind] = preds[ind][0]
    else:
        pred_selected[ind] = curr_best
        pred_selected[ind + 1] = next_best
    
    return pred_selected




def greedy_align(state_graph, trace_states, trace_preds, trace_logits, top_n):

    trace_selected = []

    for preds, pred_logits in zip(trace_preds, trace_logits):

        pred_selected = [-1 for _ in range(len(preds))]
        greedy_found = 0
        top_1_in_graph = 0

        for i in range(len(preds) - 1):
            if state_graph.has_edge(preds[i, 0], preds[i + 1, 0]):
                top_1_in_graph += 1

            if pred_selected[i] == -1:
                pred_selected = find_edge(i, preds, pred_logits, pred_selected, state_graph, top_n)
            else:
                pred_selected = find_next(i, preds, pred_selected, state_graph, top_n)

            if pred_selected[i + 1] != -1:
                greedy_found += 1

        trace_selected.append(pred_selected)
        
    top_1_accuracy = 100 * np.sum(trace_preds[:, :, 0] == np.array(trace_states)) / len(trace_states)
    greedy_accuracy = 100 * np.sum(np.array(trace_selected) == np.array(trace_states)) / len(trace_states)

    top_1_in_graph = 100 * top_1_in_graph / ((trace_states.shape[1] - 1) * trace_states.shape[0])
    greedy_in_graph = 100 * greedy_found / ((trace_states.shape[1] - 1) * trace_states.shape[0])
    
    print(f"Top-1: {top_1_accuracy:.2f}% correct")
    print(f"Greedy:  {greedy_accuracy:.2f}% correct")
    
    print(f"Top-1: {top_1_in_graph:.2f}% in the graph")
    print(f"Greedy:  {greedy_in_graph:.2f}% in the graph")

    return greedy_accuracy, top_1_in_graph, greedy_in_graph