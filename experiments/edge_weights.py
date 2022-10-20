import networkx as nx
import numpy as np
from typing import Callable
from torch.utils.data.dataset import Dataset

from macq.generate.pddl import StateEnumerator, VanillaSampling
from macq.trace import TraceList

from pddl_vis.visualizers import VISUALIZERS





def main(domain_file, problem_file, name, vis_args):

    # Create graph
    graph_generator = StateEnumerator(
        dom=domain_file, 
        prob=problem_file
    )
    states = list(graph_generator.graph.nodes())
    state_mapping = dict(zip(states, range(len(states))))
    original_state_graph = nx.relabel_nodes(graph_generator.graph, state_mapping)
    state_graph = nx.create_empty_copy(original_state_graph)


    # Generate traces
    trace_generator = VanillaSampling(
        dom=domain_file, 
        prob=problem_file,
        plan_len=30,
        num_traces=20
    )

    vis = VISUALIZERS[name](graph_generator, **vis_args).visualize_state

    # Turn traces into batches
    state_vis = []
    labels = []
    for trace in trace_generator.traces:
        state_vis.append([])
        labels.append([])
        for state in trace:
            state_vis[-1].append(vis(state))
            labels[-1].append(state_mapping[state])
    state_vis = np.array(state_vis)
    labels = np.array(labels)

    print(max(state_vis))
    print(state_vis.size)
    print(labels.size)


if __name__ == '__main__':

    name = "grid"
    domain_file = "data/pddl/grid.pddl"
    problem_file = "data/pddl/grid_data.pddl"
    vis_args = {
        'square_width': 50,
        'div_width': 1,
        'door_width': 6,
        'key_size': 15,
        'robot_size': 17,
    }

    main(domain_file, problem_file, name, vis_args)