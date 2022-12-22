import numpy as np



def find_next(seq, score, trace_logits, state_graph, best_score, best_seq, states, indiv_scores):

    next_nodes = list(state_graph.neighbors(seq[-1]))
    next_logits = trace_logits[len(seq)][next_nodes]

    sorted_logit_inds = np.argsort(-1 * next_logits)

    last_node = True if len(seq) == len(trace_logits) - 1 else False

    for ind in sorted_logit_inds:
        node = next_nodes[ind]
        prob = next_logits[ind]

        new_score = score * prob

        #if new_score >= best_score and not last_node:
        if not last_node:
            best_score, best_seq = find_next(seq + [node], new_score, trace_logits, state_graph, best_score, best_seq, states, indiv_scores + [prob])
        if last_node:
            if (states[0] == seq + [node]).all():
                print("\nWe found the guy, his score is", new_score)
                print("Broken down as:", indiv_scores + [prob])
        if last_node and new_score > best_score:
            best_score = new_score
            best_seq = seq + [node]
            print()
            print(best_score)
            print(best_seq)
            print(indiv_scores + [prob])

    return best_score, best_seq



def neighbours_align(state_graph, states, preds, logits, top_n):

    neighbours_preds = []

    for trace_preds, trace_logits in zip(preds, logits):

        best_score = 0
        best_seq = []

        for first_node in trace_preds[0][:top_n]:

            score = trace_logits[0][first_node]   # This doesn't assume trace_logits is sorted

            score, seq = find_next([first_node], score, trace_logits, state_graph, best_score, best_seq, states, [score])

            if score > best_score:
                best_score = score
                best_seq = seq
        
        neighbours_preds.append(best_seq)
    
    neighbours_accuracy = 100 * np.sum(np.array(neighbours_preds) == states) / states.size

    print()
    print(states)

    print()
    print(f"BnB neighbours accuracy:  {neighbours_accuracy:.2f}%")
    print()

    return neighbours_accuracy



# Try with trace of random states