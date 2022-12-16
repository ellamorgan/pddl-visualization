

def edge_inference(state_graph, preds, states):
    # preds is predictions (n_traces, trace_len)
    # states same shape, true results

    inferred_graph = nx.create_empty_copy(state_graph)
    n_edges = preds.shape[0] * (preds.shape[1] - 1)

    for trace in preds:
        for pre, suc in zip(trace[:-1], trace[1:]):
            if G.has_edge(pre, suc):
                G[pre][suc]['weight'] += 1 / n_edges
            else:
                G.add_edge(pre, suc, weight = 1 / n_edges)
    
    