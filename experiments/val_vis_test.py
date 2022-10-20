import networkx as nx
from PIL import Image as Img
from io import BytesIO
import torch
import numpy as np


actions = ['(putdown node0-0 key0)',
            '(move node0-0 node0-1)',
            '(move node0-1 node1-1)',
            '(move node1-1 node1-0)',
            '(move node1-0 node0-0)',
            '(pickup node0-0 key0)',
            '(move node0-0 node0-1)',

            '(putdown node0-1 key0)',
            '(move node0-1 node1-1)',
            '(move node1-1 node1-0)',
            '(move node1-0 node0-0)',
            '(move node0-0 node0-1)',
            '(pickup node0-1 key0)',
            '(move node0-1 node1-1)',
            
            '(putdown node1-1 key0)',
            '(move node1-1 node1-0)',
            '(move node1-0 node0-0)',
            '(move node0-0 node0-1)',
            '(move node0-1 node1-1)',
            '(pickup node1-1 key0)',
            '(move node1-1 node1-0)',
            
            '(putdown node1-0 key0)',
            '(move node1-0 node0-0)',
            '(move node0-0 node0-1)',
            '(move node0-1 node1-1)',
            '(move node1-1 node1-0)',
            '(pickup node1-0 key0)',
            '(move node1-0 node0-0)',]


def predict_vis_trace(batch, generator, model):

    with torch.no_grad():
        x, targets = batch       

        states = list(generator.graph.nodes())
        state_mapping = dict(zip(states, range(len(states))))
        state_graph = nx.relabel_nodes(generator.graph, state_mapping)

        # Used to remove all the action parameters
        for _, _, act in state_graph.edges(data=True):
            action = str(act['label']).split('(')[0]
            if action == 'pickup':
                act['color'] = "#7b4af7"
                act['label'] = "  pickup  "
            elif action == 'putdown':
                act['color'] = "#4a81f7"
                act['label'] = "  putdown  "
            elif action == 'move':
                act['color'] = "#83e070"
                act['label'] = "  move  "
            act['label'] = ""

        def colour_predicted_state(pred_ind, graph, target_ind=-1):

            graph = graph.copy()
            graph.nodes[pred_ind]['style'] = 'filled'
            graph.nodes[pred_ind]['fillcolor'] = '#c4c4c4'

            if target_ind > 0 and pred_ind != target_ind:
                graph.nodes[pred_ind]['fillcolor'] = '#f25d50'
                graph.nodes[target_ind]['style'] = 'filled'
                graph.nodes[target_ind]['fillcolor'] = '#50f291'
            dot_graph = nx.nx_pydot.to_pydot(graph)
            img = Img.open(BytesIO(dot_graph.create_png()))
            return img
        
        # keys: logits (20), feats (512), z (128), p (3000)
        logits = model(x)['logits'].numpy()
        inputs = x.numpy()
        targets = targets.numpy()
        print(inputs.shape)
        print(np.max(inputs))
        print(np.min(inputs))

        preds = np.array(list(map(np.argmax, logits)))

        imgs = []
        for pred, target, inpt in zip(preds, targets, inputs):
            graph_img = colour_predicted_state(pred, state_graph, target)
            input_img = Img.fromarray(((inpt.transpose(1, 2, 0) + 1) * 127.5).astype('uint8'), 'RGB').resize((graph_img.height, graph_img.height))
            combined_img = Img.new('RGB', (input_img.width + graph_img.width, input_img.height))
            combined_img.paste(input_img, (0, 0))
            combined_img.paste(graph_img, (input_img.width, 0))
            imgs.append(combined_img)
        imgs[0].save("results/gifs/predicted_trace.gif", save_all=True, append_images=imgs[1:], duration=1000, loop=0)

        print(pred)
        print(targets)
        exit()