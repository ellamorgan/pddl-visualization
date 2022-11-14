from typing import List, Tuple, Union
from torch.utils.data.dataset import Dataset
from macq.generate.pddl import StateEnumerator
import math
import numpy as np
from pddl_vis.utils import get_combination
from pddl_vis.visualizers import Visualizer


class PDDLDataset(Dataset):

    def __init__(
        self,
        visualizer: Visualizer,
        n_samples: int,
        img_size: Union[Tuple[int, int], None] = None,
        train: bool = True,
        inds: List[int] = [],
        seed : Union[int, None] = None,
    ) -> None:

        def preprocess(img):
            return (np.array(img).transpose((2, 0, 1)) / 127.5) - 1
            
        self.data = visualizer.create_dataset(n_samples=n_samples, preprocess=preprocess, img_size=img_size, seed=seed)
        self.targets = list(range(len(self.data)))
        self.n_states = len(self.data)
        self.n_samples = n_samples
        self.n_pairs = math.comb(n_samples, 2)
        self.train = train
        self.inds = inds
    
    def __getitem__(self, index):
        if self.train:
            if index >= self.n_states * self.n_pairs:
                raise IndexError(f"Dataset index {index} out of range for size {self.n_states * self.n_pairs}")
            state_ind = index // self.n_pairs
            p1, p2 = get_combination(index % self.n_pairs, self.n_samples, self.n_pairs)
            return index, self.data[state_ind][p1], self.data[state_ind][p2], state_ind
        else:
            if index >= self.n_states * self.n_samples:
                raise IndexError(f"Dataset index {index} out of range for size {self.n_states * self.n_samples}")
            state_ind = index // self.n_samples
            sample_ind = index % self.n_samples
            return self.data[state_ind][sample_ind], state_ind
    
    def __len__(self):
        if self.train:
            return self.n_states * self.n_pairs
        else:
            return self.n_states * self.n_samples