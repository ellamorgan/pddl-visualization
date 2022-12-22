from pddl_vis.utils import load_args
from pddl_vis.dataset import get_domain, visualize_traces
from pddl_vis.aligning import greedy_align, get_graph_and_traces, get_predictions, neighbours_align, top_n_align
from solo.utils.misc import make_contiguous
from solo.methods import METHODS
import numpy as np
import math


load_model_path = "trained_models/byol/1yhik0ge/byol-elevator-elevator1-5-128-1yhik0ge-ep=4.ckpt"
#load_model_path = "trained_models/byol/1k20okc5/byol-grid-grid2-5-256-1k20okc5-ep=4.ckpt"


def main():

    # Load pretrained model

    args = load_args()

    domain_file = 'data/pddl/' + args.domain + '/' + args.domain + '.pddl'
    problem_file = 'data/pddl/' + args.domain + '/problems/' + args.problem + '.pddl'

    visualizer, n_states = get_domain(
        domain = args.domain,
        domain_file=domain_file, 
        problem_file=problem_file
    )
    args.num_classes = n_states

    model = METHODS[args.method].load_from_checkpoint(load_model_path, **args.__dict__)
    make_contiguous(model)



    n_traces = 1
    trace_len = 5

    # Get one long trace to break into trace_len sizes
    state_graph, traces, states, _ = get_graph_and_traces(
        domain_file, 
        problem_file, 
        n_traces=1, 
        trace_len=5 * trace_len
    )

    traces = [traces[0][20:]]
    states = np.array([states[0][20:]])

    # (1, trace_len * n_traces, *img_shape)
    data = visualize_traces(traces, vis=visualizer.visualize_state, img_size=(args.img_h, args.img_w))

    preds, logits = get_predictions(model, data[0], batch_size=args.batch_size)

    # (n_traces, trace_len, n_states)
    preds = preds.reshape((n_traces, trace_len, *preds.shape[1:]))
    logits = logits.reshape((n_traces, trace_len, *logits.shape[1:]))
    states = states.reshape((n_traces, trace_len))


    top_1_accuracy = 100 * np.sum(preds[:, :, 0] == np.array(states)) / states.size

    top_n = 5

    # Turn logits into a probability distribution
    logits = math.e ** logits / np.sum(math.e ** logits, axis=2)[:, :, np.newaxis]

    print(np.min(logits), np.max(logits))


    greedy_accuracy, top_1_in_graph, greedy_in_graph = greedy_align(
        state_graph, 
        states, 
        preds, 
        logits, 
        top_n
    )

    neighbour_accuracy = neighbours_align(
        state_graph, 
        states, 
        preds, 
        logits, 
        top_n=n_states
    )

if __name__ == '__main__':

    main()