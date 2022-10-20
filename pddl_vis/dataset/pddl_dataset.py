from typing import Callable, List
from torch.utils.data.dataset import Dataset
from macq.generate.pddl import StateEnumerator
from macq.trace import Step
import math
import numpy as np
from pddl_vis.utils import get_combination


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
            self.states = [generator.tarski_state_to_macq(state) for state in generator.graph.nodes()]
        else:
            self.states = [generator.tarski_state_to_macq(generator.graph.nodes[i]) for i in inds]

        def preprocess(img):
            return (np.array(img).transpose((2, 0, 1)) / 127.5) - 1
            
        self.data = [[preprocess(vis(Step(state, None, 0), size=img_size)) for _ in range(n_samples)] for state in self.states]
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