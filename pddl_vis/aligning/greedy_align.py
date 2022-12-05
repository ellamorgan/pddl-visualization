import numpy as np
from pddl_vis.utils import graph_to_img, colour_state
from pddl_vis.dataset import visualize_trace
from pddl_vis.aligning import get_graph_and_traces, get_predictions


def greedy_align(state_graph, trace_states, pred_logits, top_n):

    greedy_result = []
    top_1_result = []
    greedy_in_graph = 0
    top_1_in_graph = 0

    for preds in pred_logits:

        # pred_selected is the state number
        # preds is the logits (trace_len, n_states)

        pred_selected = [-1 for _ in range(len(preds))]

        # If previous state is unknown
        def find_edge(ind, pred_selected):
            edge_found = False

            curr_preds = np.argsort(-1 * preds[ind])
            next_preds = np.argsort(-1 * preds[ind + 1])

            for curr in curr_preds[:top_n]:
                for next_ in next_preds[:top_n]:
                    if preds[ind][curr] < preds[ind + 1][next_]:
                        break
                    if state_graph.has_edge(curr, next_):
                        pred_selected[ind] = curr
                        pred_selected[ind + 1] = next_
                        edge_found = True
                        break
                if edge_found:
                    break
            
            if not edge_found:
                pred_selected[ind] = curr_preds[0]
            
            return pred_selected

        # If previous state is known
        def find_next(ind, pred_selected):
            edge_found = False

            next_preds = np.argsort(-1 * preds[ind])

            for next_ in next_preds[:top_n]:
                if state_graph.has_edge(pred_selected[ind], next_):
                    pred_selected[ind + 1] = next_
                    edge_found = True
                    break
            
            if not edge_found:
                pred_selected[ind] = next_preds[0]
            
            return pred_selected

        top_1 = list(map(np.argmax, preds))

        pred_selected = find_edge(0, pred_selected)

        for i in range(1, len(preds) - 1):
            if pred_selected[i] == -1:
                pred_selected = find_edge(i, pred_selected)
            else:
                pred_selected = find_next(i, pred_selected)

            if state_graph.has_edge(pred_selected[i - 1], pred_selected[i]):
                greedy_in_graph += 1    
            if state_graph.has_edge(top_1[i - 1], top_1[i]):
                top_1_in_graph += 1   

        if pred_selected[-1] == -1:
            pred_selected[-1] = np.argmax(preds[-1])

        greedy_result.append(pred_selected)
        top_1_result.append(top_1)
    
    greedy_result = np.array(greedy_result)
    top_1_result = np.array(top_1_result)

    before_accuracy = 100 * np.sum(top_1_result == trace_states) / trace_states.size
    after_accuracy = 100 * np.sum(greedy_result == trace_states) / trace_states.size

    before_in_graph = 100 * top_1_in_graph / (trace_states.size - 1)
    after_in_graph = 100 * greedy_in_graph / (trace_states.size - 1)
    
    print(f"Top-1 accuracy: {before_accuracy:.2f}% correct")
    print(f"Greedy accuracy:  {after_accuracy:.2f}% correct")
    
    print(f"In graph before: {before_in_graph:.2f}% in the graph")
    print(f"In graph after:  {after_in_graph:.2f}% in the graph")


    return before_accuracy, after_accuracy, before_in_graph, after_in_graph