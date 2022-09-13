from asyncio import streams
from typing import Callable
from torch.utils.data.dataset import Dataset
from macq.generate.pddl import StateEnumerator
import math





class BaseDataset(Dataset):

    def __init__(
        self,
        domain_file: str,
        problem_file: str,
        n_samples: int,
        vis: Callable,
    ) -> None:

        generator = StateEnumerator(dom=domain_file, prob=problem_file)
        self.states = [generator.tarski_state_to_macq(state) for state in generator.graph.nodes]
        self.data = [[vis(state) for _ in range(len(n_samples))] for state in self.states]
        self.targets = list(range(len(self.data)))
        self.n_states = len(self.data)
        self.n_samples = n_samples
        self.n_pairs = math.comb(n_samples, 2)
    

    def _get_combination(self, ind):
        '''
        Given the index of a combination of 2 items, returns the indices of the 2 items
        I.e. given ind in range [0, nC2), returns tuple of 2 indices in range [0, n)
        '''
        assert ind >= 0 and ind < self.n_pairs
        p1 = 0
        sub = self.n_samples - 1
        while ind - sub >= 0:
            ind -= sub
            p1 += 1
            sub -= 1
        p2 = p1 + 1 + ind

        return p1, p2
    

    def __getitem__(self, index):
        # Returns a pair of states?
        if index >= self.n_states * self.n_pairs:
            raise IndexError("Dataset index out of range")
        state_ind = index // self.n_states
        p1, p2 = self._get_combination(index % self.n_states)
        return self.data[state_ind][p1], self.data[state_ind][p2]
    
    def __len__(self):
        return self.n_states * self.n_pairs