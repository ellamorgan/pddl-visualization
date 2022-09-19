from typing import Callable
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from macq.generate.pddl import StateEnumerator
from macq.trace import Step
import math
import torch
from pddl_vis.utils import get_combination
import numpy as np


class PDDLDataset(Dataset):

    def __init__(
        self,
        generator: StateEnumerator,
        vis: Callable,
        n_samples: int,
        img_size: tuple,
    ) -> None:

        self.states = [generator.tarski_state_to_macq(state) for state in generator.graph.nodes]
        self.data = [[vis(Step(state, None, 0), size=img_size) for _ in range(n_samples)] for state in self.states]
        self.targets = list(range(len(self.data)))
        self.n_states = len(self.data)
        self.n_samples = n_samples
        self.n_pairs = math.comb(n_samples, 2)
    
    def __getitem__(self, index):
        if index >= self.n_states * self.n_pairs:
            raise IndexError("Dataset index out of range")
        state_ind = index // self.n_pairs
        p1, p2 = get_combination(index % self.n_pairs, self.n_samples, self.n_pairs)
        return index, self.data[state_ind][p1], self.data[state_ind][p2], state_ind
    
    def __len__(self):
        return self.n_states * self.n_pairs


def prepare_dataloader(
    train_dataset: Dataset, batch_size: int, num_workers: int
) -> DataLoader:

    def collate_fn(data):
        '''
        data: list where each element is [index, img1, img2, state_ind]
        '''
        data = list(zip(*data))
        data = [torch.tensor(data[0]), [torch.tensor(np.array(img)).float() for img in data[1:-1]], torch.tensor(data[-1])]
        return data

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    return train_loader