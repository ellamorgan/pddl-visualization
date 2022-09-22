from typing import Callable, Optional, List
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
        train: bool = True,
        inds: List[int] = [],
    ) -> None:

        if len(inds) == 0:
            self.states = [generator.tarski_state_to_macq(state) for state in generator.graph.nodes]
        else:
            self.states = [generator.tarski_state_to_macq(generator.graph.nodes[i]) for i in inds]
            
        self.data = [[vis(Step(state, None, 0), size=img_size) for _ in range(n_samples)] for state in self.states]
        self.targets = list(range(len(self.data)))
        self.n_states = len(self.data)
        self.n_samples = n_samples
        self.n_pairs = math.comb(n_samples, 2)
        self.train = train
        self.inds = inds
    
    def __getitem__(self, index):
        if self.train:
            if index >= self.n_states * self.n_pairs:
                raise IndexError("Dataset index out of range")
            state_ind = index // self.n_pairs
            p1, p2 = get_combination(index % self.n_pairs, self.n_samples, self.n_pairs)
            return index, self.data[state_ind][p1], self.data[state_ind][p2], state_ind
        else:
            if index >= self.n_states * self.n_samples:
                raise IndexError("Dataset index out of range")
            state_ind = index // self.n_samples
            sample_ind = index % self.n_samples
            return self.data[state_ind][sample_ind], state_ind
    
    def __len__(self):
        return self.n_states * self.n_pairs



def prepare_dataloader(
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    test_dataset: Optional[Dataset] = None,
    batch_size: int = 64, 
    num_workers: int = 4,
) -> DataLoader:

    def train_collate(data):
        '''
        data: list where each element is [index, img1, img2, state_ind]
        '''
        data = list(zip(*data))
        data = [torch.tensor(data[0]), [torch.tensor(np.array(img)).float() for img in data[1:-1]], torch.tensor(data[-1])]
        return data
    
    def test_collate(data):
        return data

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=train_collate,
    )

    loaders = [train_loader]

    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=test_collate,
        )
        loaders.append(val_loader)
    
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=test_collate,
        )
        loaders.append(test_loader)

    return loaders