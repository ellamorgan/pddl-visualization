import numpy as np
import networkx as nx
import torch
from macq.generate.pddl import StateEnumerator, VanillaSampling
from PIL import Image
from io import BytesIO
 


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



def colour_state(state, state_mapping, graph, colour='#bababa'):
    graph = graph.copy()
    graph.nodes[state]['style'] = 'filled'
    graph.nodes[state]['fillcolor'] = colour
    return graph


def graph_to_img(graph):
    dot_graph = nx.nx_pydot.to_pydot(graph)
    img = Image.open(BytesIO(dot_graph.create_png()))
    return img


def trace_pred_main(model, domain_file, problem_file, n_data, batch_size, vis, img_size, top_n=5):

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

    for _, _, act in state_graph.edges(data=True):
        act['label'] = ""

    trace_generator = VanillaSampling(
        dom=domain_file, 
        prob=problem_file,
        plan_len=n_data,
        num_traces=1
    )

    trace_states = []
    trace_imgs = []
    for step in trace_generator.traces[0]:
        state_num = state_hashes.index(hash(str(step.state)))
        trace_states.append(state_num)
        trace_imgs.append(graph_to_img(colour_state(state_num, state_mapping, state_graph)))
    
    for i in range(len(trace_states) - 1):
        assert state_graph.has_edge(trace_states[i], trace_states[i + 1])
    
    data = generate_data(trace_generator.traces[0], vis, img_size)

    epochs = int(len(data) // batch_size)
    if len(data) % batch_size != 0:
        epochs += 1

    trace_preds = []
    pred_imgs = []
    for epoch in range(epochs):
        batch = data[epoch * batch_size : (epoch + 1) * batch_size]
        x = torch.tensor(batch).float()
        logits = model(x)['logits'].detach().numpy()
        trace_preds += [np.argpartition(sample, -top_n)[-top_n:] for sample in logits]
    pred_inds = np.array(trace_preds)       # (n_data, top_n)
    print(pred_inds.shape)
    print(logits.shape)
    pred_logits = np.array([logit[ind]] for logit, ind in zip(logits, pred_inds))
    print(pred_logits.shape)

    for inds, logits in zip(pred_inds, pred_logits):

        state_pred = inds[np.argmax(logits)]
        pred_imgs.append(graph_to_img(colour_state(state_pred, state_mapping, state_graph)))
    

    imgs = []
    for trace_img, pred_img in zip(trace_imgs, pred_imgs):
        assert trace_img.height == pred_img.height
        dst = Image.new('RGB', (pred_img.width + trace_img.width, pred_img.height))
        dst.paste(pred_img, (0, 0))
        dst.paste(trace_img, (pred_img.width, 0))
        imgs.append(dst)

    state_graph_path = 'results/gifs/trace_preds.gif'
    imgs[0].save(state_graph_path, save_all=True, append_images=imgs[1:], duration=1000, loop=0)