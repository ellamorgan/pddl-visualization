import networkx as nx
import torch
import numpy as np
from macq.generate.pddl import StateEnumerator, VanillaSampling


def get_graph_and_traces(domain_file, problem_file, n_traces, trace_len):

    generator = StateEnumerator(
        dom=domain_file, 
        prob=problem_file
    )

    states = generator.graph.nodes()
    macq_states = list(map(generator.tarski_state_to_macq, states))
    state_hashes = list(map(hash, map(str, macq_states)))

    state_mapping = dict(zip(states, range(len(states))))
    state_graph = nx.relabel_nodes(generator.graph, state_mapping)

    for _, _, act in state_graph.edges(data=True):
        act['label'] = ""

    trace_generator = VanillaSampling(
        dom=domain_file, 
        prob=problem_file,
        plan_len=trace_len,
        num_traces=n_traces
    )

    trace_states = []
    for trace in trace_generator.traces:
        trace_states.append([])
        for step in trace:
            trace_states[-1].append(state_hashes.index(hash(str(step.state))))
    
    return state_graph, trace_generator.traces, np.array(trace_states)

'''
def get_predictions(model, data, batch_size, top_n):

    epochs = int(len(data) // batch_size)
    if len(data) % batch_size != 0:
        epochs += 1

    trace_preds = []
    trace_logits = []
    for epoch in range(epochs):
        batch = data[epoch * batch_size : (epoch + 1) * batch_size]
        x = torch.tensor(batch).float()
        logits = model(x)['logits'].detach().numpy()
        for sample in logits:
            top_n_args = np.argpartition(sample, -top_n)[-top_n:]
            top_n_logits = sample[top_n_args]
            sorted_inds = np.argsort(-1 * top_n_logits)
            trace_preds.append(top_n_args[sorted_inds])
            trace_logits.append(top_n_logits[sorted_inds])
    preds = np.array(trace_preds)       # (n_data, top_n)
    pred_logits = np.array(trace_logits)
    print(preds.shape)
    print(pred_logits.shape)

    return preds, pred_logits
'''


def get_trace_predictions(model, data, batch_size):

    trace_logits = []
    trace_preds = []

    trace_epochs = int(len(data[0]) // batch_size)
    if len(data) % batch_size != 0:
        trace_epochs += 1

    for trace in data:    
        trace_logits.append([])
        trace_preds.append([])

        for epoch in range(trace_epochs):
            batch = trace[epoch * batch_size : (epoch + 1) * batch_size]
            x = torch.tensor(batch).float()
            logits = model(x)['logits'].detach().numpy()
            trace_logits[-1] += list(logits)
            trace_preds[-1] += list(map(np.argsort, -1 * logits))

    trace_logits = np.array(trace_logits)       # (n_traces, trace_len, n_states)
    trace_preds = np.array(trace_preds)

    return trace_preds, trace_logits