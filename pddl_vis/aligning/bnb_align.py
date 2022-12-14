import numpy as np
import math









def find_next(
    seq,
    score,
    edges_found,
    best_scores,
    best_seqs,
    longest_seq_length,
    state_graph,
    trace_preds,
    trace_logits,
    top_n
):
    curr_ind = len(seq)

    if curr_ind == len(trace_preds):
        if score > best_scores[edges_found]:
            best_scores[edges_found] = score
            best_seqs[edges_found] = seq
            if edges_found > longest_seq_length:
                longest_seq_length = edges_found
        return best_scores, best_seqs, longest_seq_length
    
    for next_node in trace_preds[curr_ind][:top_n]:

        next_score = score * trace_logits[curr_ind][next_node]

        if state_graph.has_edge(seq[-1], next_node):
            next_edges_found = edges_found + 1
        else:
            next_edges_found = edges_found
        
        best_scores, best_seqs, longest_seq_length = find_next(
            seq + [next_node],
            next_score,
            next_edges_found,
            best_scores,
            best_seqs,
            longest_seq_length,
            state_graph,
            trace_preds,
            trace_logits,
            top_n
        )
    
    return best_scores, best_seqs, longest_seq_length



def bnb_align(state_graph, states, preds, logits, top_n):


    state_preds = []

    for i, (trace_preds, trace_logits) in enumerate(zip(preds, logits)):

        best_scores = [0 for _ in range(len(trace_preds))]
        best_seqs = [[] for _ in range(len(trace_preds))]

        longest_seq_length = 0

        for first_node in trace_preds[0][:top_n]:

            score = trace_logits[0][first_node]   # This doesn't assume trace_logits is sorted

            best_scores, best_seqs, longest_seq_length = find_next(
                [first_node],
                score,
                0,
                best_scores,
                best_seqs,
                longest_seq_length,
                state_graph,
                trace_preds,
                trace_logits,
                top_n
            )
        
        '''
        print()
        for j, seq in enumerate(best_seqs):
            if len(seq) == 0:
                continue
            for state in seq:
                print('{:<3}'.format(state), end='')
            print()
            for state in states[i]:
                print('{:<3}'.format(state), end='')
            print()
            print("Correct:", np.sum(seq == states[i]))
            print("Connected:", j)
            print()
        print("Longest seq length:", longest_seq_length)
        '''
        
        state_preds.append(best_seqs[longest_seq_length])

    
    accuracy = 100 * np.sum(np.array(state_preds) == states) / states.size

    print(f"Longest aligning accuracy:  {accuracy:.2f}%")

    return accuracy