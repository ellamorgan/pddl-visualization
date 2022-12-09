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


def get_predictions(model, data, batch_size):

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
            trace_preds.append(np.argsort(-1 * sample))
            trace_logits.append(sample)

    preds = np.array(trace_preds)       # (n_data, top_n)
    logits = np.array(trace_logits)

    return preds, logits