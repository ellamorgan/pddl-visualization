import numpy as np
import math









def find_next(
    seq,
    score,
    edges_found,
    best_scores,
    best_seqs,
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
        return best_scores, best_seqs
    
    for next_node in trace_preds[curr_ind][:top_n]:

        next_score = score * trace_logits[curr_ind][next_node]

        if state_graph.has_edge(seq[-1], next_node):
            next_edges_found = edges_found + 1
        else:
            next_edges_found = edges_found
        
        best_scores, best_seqs = find_next(
            seq + [next_node],
            next_score,
            next_edges_found,
            best_scores,
            best_seqs,
            state_graph,
            trace_preds,
            trace_logits,
            top_n
        )
    
    return best_scores, best_seqs



def bnb_align(state_graph, states, preds, logits, top_n):


    for trace_preds, trace_logits in zip(preds, logits):

        best_scores = [0 for _ in range(len(trace_preds))]
        best_seqs = [[] for _ in range(len(trace_preds))]

        for first_node in trace_preds[0][:top_n]:

            score = trace_logits[0][first_node]   # This doesn't assume trace_logits is sorted

            best_scores, best_seqs = find_next(
                [first_node],
                score,
                0,
                best_scores,
                best_seqs,
                state_graph,
                trace_preds,
                trace_logits,
                top_n
            )
        
        print(best_scores)
        print(best_seqs)




















































'''
def _find_next(seq, score, trace_preds, trace_logits, state_graph, edge_score, best_score, best_seq, top_n):

    next_nodes = trace_preds[len(seq)]
    next_logits = trace_logits[len(seq)][next_nodes]

    last_node = True if len(seq) == len(trace_logits) - 1 else False

    for node, logit in zip(next_nodes[:top_n], next_logits[:top_n]):
        prob = math.e ** logit

        if state_graph.has_edge(seq[-1], node):
            edge_score += 1

        new_score = score * prob

        print(f"edge score: {edge_score} last node? {last_node}")

        #if new_score >= np.max(best_score) and not last_node:
        if not last_node:
            best_score, best_seq, edge_score = find_next(seq + [node], new_score, trace_preds, trace_logits, state_graph, edge_score, best_score, best_seq, top_n)
        if last_node and new_score > best_score[edge_score]:
            best_score[edge_score] = new_score
            best_seq[edge_score] = seq + [node]

    return best_score, best_seq, edge_score



def _bnb_align(state_graph, states, preds, logits, top_n):

    bnb_preds = []

    for trace_preds, trace_logits in zip(preds, logits):

        best_score = [0 for _ in range(len(trace_preds))]
        best_seq = [[] for _ in range(len(trace_preds))]

        for first_node in trace_preds[0][:top_n]:

            score = math.e ** trace_logits[0][first_node]   # This doesn't assume trace_logits is sorted

            score, seq, edge_score = find_next([first_node], score, trace_preds, trace_logits, state_graph, 0, best_score, best_seq, top_n)

            if score[edge_score] > best_score[edge_score]:
                best_score[edge_score] = score
                best_seq[edge_score] = seq
        
        bnb_preds.append(best_seq)
    
    print("\nBnB selected:")
    print(bnb_preds)
    
    #bnb_accuracy = 100 * np.sum(np.array(bnb_preds) == states) / states.size

    #print()
    #print(f"BnB accuracy:  {bnb_accuracy:.2f}%")
    #print()

    #return bnb_accuracy
'''


# Try with trace of random states