import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

# edge_weights.py generates the data - index of two nodes and edge weights
# we need to generate examples of each node, and pass them into a network that should predict the weight
# generate the visualizations, pass them through the contrastive network, pass the embeddings (do we have these?) into the predictor


class EdgePredictor(nn.Module):

    def __init__(self, embed_size, dims = [100]):

        super(EdgePredictor, self).__init__()

        assert len(dims) > 0

        self.layers = [nn.Linear(2 * embed_size, dims[0])]
        for i in range(len(dims[1:])):
            self.layers.append(nn.Linear(dims[i - 1], dims[i]))
        self.layers.append(nn.Linear(dims[-1], 1))
    
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        return x


def process_img(img, size):
        img = img.resize(size)
        array_from_img = np.asarray(img).transpose(2, 0, 1)
        normalized = (array_from_img / 127.5) - 1
        return normalized


def train_edge_network(data, model, vis, n_samples, states, img_size):

    train_split = 0.8
    val_split = 0.1

    # data: [[i, j, weight] x (n_state x n_state)]
    dataset = []
    targets = []
    
    for i, j, w in data:
        state = []
        for _ in range(n_samples):
            img1 = process_img(vis(states[i]), img_size)
            img2 = process_img(vis(states[j]), img_size)
            state.append([img1, img2])
        targets.append(w)
        dataset.append(state)
    dataset = np.array(dataset)
    targets = np.array(targets)

    inds = list(range(len(dataset)))
    random.shuffle(inds)
    assert train_split + val_split < 1
    train_inds = inds[:int(train_split * len(dataset))]
    val_inds = inds[int(train_split * len(dataset)) : int((train_split + val_split) * len(dataset))]
    test_inds = inds[int((train_split + val_split) * len(dataset)):]

    train_data = torch.tensor(dataset[train_inds]).float()
    train_target = torch.tensor(targets[train_inds]).float()
    val_data = torch.tensor(dataset[val_inds]).float()
    val_target = torch.tensor(targets[val_inds]).float()
    test_data = torch.tensor(dataset[test_inds]).float()
    test_target = torch.tensor(targets[test_inds]).float()


    # dataset: (n_states x n_states, n_samples, 3, img_w, img_h)
    # targets: (n_states x n_states)
    print(dataset.shape)
    print(targets.shape)