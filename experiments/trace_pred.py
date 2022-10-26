import numpy as np
import networkx as nx
import torch
from macq.generate.pddl import StateEnumerator, VanillaSampling



def generate_data(trace, vis, img_size):

    def process_img(img, size):
        img = img.resize(size)
        array_from_img = np.asarray(img).transpose(2, 0, 1)
        normalized = (array_from_img / 127.5) - 1
        return normalized

    # Get visualizations
    state_vis = []
    for step in trace:
        state_vis.append(process_img(vis(step), img_size))
    state_vis = np.array(state_vis)

    # state_vis: (n_data, 3, img_w, img_h)
    return state_vis


def trace_pred_main(model, domain_file, problem_file, n_data, batch_size, vis, img_size):

    graph_generator = StateEnumerator(
        dom=domain_file, 
        prob=problem_file
    )
    #states = list(map(hash, map(graph_generator.tarski_state_to_macq, graph_generator.graph.nodes())))

    states = graph_generator.graph.nodes()
    macq_states = list(map(graph_generator.tarski_state_to_macq, states))
    state_hashes = list(map(hash, map(str, macq_states)))

    state_mapping = dict(zip(states, range(len(states))))
    state_graph = nx.relabel_nodes(graph_generator.graph, state_mapping)

    trace_generator = VanillaSampling(
        dom=domain_file, 
        prob=problem_file,
        plan_len=n_data,
        num_traces=1
    )

    trace_states = []
    for step in trace_generator.traces[0]:
        trace_states.append(state_hashes.index(hash(str(step.state))))
    
    for i in range(len(trace_states) - 1):
        assert state_graph.has_edge(trace_states[i], trace_states[i + 1])
    
    data = generate_data(trace_generator.traces[0], vis, img_size)

    epochs = int(len(data) // batch_size)
    if len(data) % batch_size != 0:
        epochs += 1

    trace_preds = []
    for epoch in range(epochs):
        batch = data[epoch * batch_size : (epoch + 1) * batch_size]
        x = torch.tensor(batch).float()
        logits = model(x)['logits'].detach().numpy()
        top_preds = []
        for _ in range(5):
            top_inds = list(map(np.argmax, logits))
            top_preds.append(top_inds)
            logits[list(range(len(logits))), top_inds] = 0
        top_preds = np.array(top_preds).transpose()
        trace_preds += list(top_preds)
    preds = np.array(trace_preds)       # (n_data, 5)

    no = 0
    yes = 0

    pred_selected = [-1 for _ in range(len(preds))]
    state_pred = [0]

    # If previous state is unknown
    def find_edge(ind):
        edge_found = False
        for i in range(5):
            for j in range(i + 1):
                if state_graph.has_edge(preds[ind, i], preds[ind+1, j]):
                    pred_selected[ind] = i
                    pred_selected[ind+1] = j
                    edge_found = True
                    break
            if edge_found:
                break

    
    def find_next(ind):
        for i in range(5):
            if state_graph.has_edge(preds[ind, pred_selected[ind]], preds[ind+1, i]):
                pred_selected[ind+1] = i
                break


    # While I am finding things in the graph, this doesn't mean it's correct. Get the ground truth 

    # Really bad names
    never_found = 0
    found = 0
    yes = 0
    no = 0
    for i in range(len(preds) - 1):
        if state_graph.has_edge(preds[i, 0], preds[i+1, 0]):
            yes += 1
        else:
            no += 1

        if pred_selected[i] == -1:
            find_edge(i)
        else:
            find_next(i)

        if pred_selected[i+1] == -1:
            never_found += 1
            state_pred.append(preds[i+1, 0])
        else:
            found += 1
            state_pred.append(preds[i+1, pred_selected[i+1]])
    
    print(f"Before: {100 * sum(preds[:, 0] == np.array(trace_states)) / len(trace_states):.2f}% correct")
    print(f"After:  {100 * sum(np.array(state_pred) == np.array(trace_states)) / len(trace_states):.2f}% correct")
    
    print(f"Before: {100 * yes / (len(preds) - 1):.2f}% in the graph, {100 * no / (len(preds) - 1):.2f}% not in the graph")
    print(f"After:  {100 * found / (len(preds) - 1):.2f}% in the graph, {100 * never_found / (len(preds) - 1):.2f}% not in the graph")