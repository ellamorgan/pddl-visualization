import numpy as np
from pddl_vis.utils import graph_to_img, colour_state
from pddl_vis.dataset import visualize_trace
from pddl_vis.aligning import get_graph_and_traces, get_predictions


def greedy_align(model, domain_file, problem_file, n_data, batch_size, vis, img_size, top_n):

    state_graph, traces, trace_states = get_graph_and_traces(domain_file, problem_file, n_traces=1, trace_len=n_data)
    trace = traces[0]
    trace_states = trace_states[0]
    
    data = visualize_trace(trace, vis, img_size)

    preds, _ = get_predictions(model, data, batch_size, top_n)


    pred_selected = [-1 for _ in range(len(preds))]
    state_pred = [0]

    # If previous state is unknown
    def find_edge(ind):
        edge_found = False
        for i in range(top_n):
            for j in range(i + 1):
                if state_graph.has_edge(preds[ind, i], preds[ind+1, j]):
                    pred_selected[ind] = i
                    pred_selected[ind+1] = j
                    edge_found = True
                    break
            if edge_found:
                break

    
    def find_next(ind):
        for i in range(top_n):
            if state_graph.has_edge(preds[ind, pred_selected[ind]], preds[ind+1, i]):
                pred_selected[ind+1] = i
                break


    never_found = 0
    found = 0
    top_1_in_graph = 0
    for i in range(len(preds) - 1):
        if state_graph.has_edge(preds[i, 0], preds[i+1, 0]):
            top_1_in_graph += 1

        if pred_selected[i] == -1:
            find_edge(i)
        else:
            find_next(i)

        if pred_selected[i] == -1:
            never_found += 1
            state_pred.append(preds[i, 0])
        else:
            found += 1
            state_pred.append(preds[i+1, pred_selected[i+1]])
    
    before_accuracy = 100 * sum(preds[:, 0] == np.array(trace_states)) / len(trace_states)
    after_accuracy = 100 * sum(np.array(state_pred) == np.array(trace_states)) / len(trace_states)

    before_in_graph = 100 * top_1_in_graph / (len(preds) - 1)
    after_in_graph = 100 * found / (len(preds) - 1)
    
    print(f"Before: {before_accuracy:.2f}% correct")
    print(f"After:  {after_accuracy:.2f}% correct")
    
    print(f"Before: {before_in_graph:.2f}% in the graph")
    print(f"After:  {after_in_graph:.2f}% in the graph")


    '''
    graph_imgs = []
    for top_pred, align_pred, truth in zip(preds[:, 0], state_pred, trace_states):
        curr_graph = state_graph.copy()
        curr_graph = colour_state(top_pred, curr_graph, colour='#eb343d')
        curr_graph = colour_state(align_pred, curr_graph, colour='#4c34eb')
        curr_graph = colour_state(truth, curr_graph, colour='#34eb6e')
        graph_imgs.append(graph_to_img(curr_graph))
    
    state_graph_path = 'results/gifs/trace_preds.gif'
    graph_imgs[0].save(state_graph_path, save_all=True, append_images=graph_imgs[1:], duration=500, loop=0)
    '''

    return before_accuracy, after_accuracy, before_in_graph, after_in_graph