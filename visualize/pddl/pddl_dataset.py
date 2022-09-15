from typing import Callable
from torch.utils.data.dataset import Dataset
from macq.generate.pddl import Generator
import math
from visualize.utils import get_combination


class PDDLDataset(Dataset):

    def __init__(
        self,
        generator: Generator,
        vis: Callable,
        n_samples: int,
    ) -> None:

        self.states = [generator.tarski_state_to_macq(state) for state in generator.graph.nodes]
        self.data = [[vis(state) for _ in range(len(n_samples))] for state in self.states]
        self.targets = list(range(len(self.data)))
        self.n_states = len(self.data)
        self.n_samples = n_samples
        self.n_pairs = math.comb(n_samples, 2)
    
    def __getitem__(self, index):
        if index >= self.n_states * self.n_pairs:
            raise IndexError("Dataset index out of range")
        state_ind = index // self.n_states
        p1, p2 = get_combination(index % self.n_states, self.n_samples, self.n_pairs)
        return self.data[state_ind][p1], self.data[state_ind][p2]
    
    def __len__(self):
        return self.n_states * self.n_pairs