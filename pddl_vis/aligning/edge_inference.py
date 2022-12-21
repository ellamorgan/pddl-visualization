import numpy as np
from copy import deepcopy

#inferred_graph = nx.create_empty_copy(state_graph)
#inferred_graph[pre][suc]['weight'] += 1 / n_edges

def edge_inference(state_graph, preds):
    # preds is predictions (n_traces, trace_len)
    # states same shape, true results

    n_nodes = state_graph.number_of_nodes()
    frequencies = np.zeros((n_nodes, n_nodes))

    for trace in preds:
        for pre, suc in zip(trace[:-1], trace[1:]):
            frequencies[pre, suc] += 1
    
    # Can go by percentiles or frequencies to figure out when to drop
    percentiles = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    sorted_freqs = np.sort(frequencies.flatten())
    cutoffs = [sorted_freqs[int(frequencies.size * percentile)] for percentile in percentiles]

    # For each node keep the most relevant?
    # Don't want to assume a certain number of outgoing edges from each node
    
    neighbour_freq = deepcopy(frequencies)

    for i in range(n_nodes):

        # Look at all the outgoing edges, only keep the most "significant"
        random_chance = np.sum(neighbour_freq[i]) / n_nodes
        neighbour_freq[i][np.where(neighbour_freq[i] < random_chance)] = 0
    
    percentile_rewards = [0 for _ in range(len(percentiles))]
    percentile_penalties = [0 for _ in range(len(percentiles))]

    neighbour_reward = 0
    neighbour_penalty = 0

    for i in range(len(frequencies)):
        for j in range(len(frequencies[0])):

            percentile_score = []

            for cutoff in cutoffs:
                if frequencies[i, j] >= cutoff:
                    percentile_score.append(frequencies[i, j] / n_nodes)
                else:
                    percentile_score.append(0)

            if state_graph.has_edge(i, j):
                percentile_rewards = [prev + curr for prev, curr in zip(percentile_rewards, percentile_score)]
                neighbour_reward += neighbour_freq[i, j] / n_nodes
            else:
                percentile_penalties = [prev + curr for prev, curr in zip(percentile_penalties, percentile_score)]
                neighbour_penalty += neighbour_freq[i, j] / n_nodes
    
    print("\nPercentile: ", end="")
    for percentile in percentiles:
        print(f"{percentile:.2f}".ljust(10), end="")
    print("\nReward:     ", end="")
    for reward in percentile_rewards:
        print(f"{reward:.2f}".ljust(10), end="")
    print("\nPenalty:    ", end="")
    for penalty in percentile_penalties:
        print(f"{penalty:.2f}".ljust(10), end="")
    
    print(f"\n\nNeighbour reward:  {neighbour_reward:.2f}")
    print(f"Neighbour penalty: {neighbour_penalty:.2f}")

    print()