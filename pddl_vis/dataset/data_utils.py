from typing import Optional, List
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from macq.generate.pddl import StateEnumerator
import torch
from pddl_vis.visualizers import VISUALIZERS
import numpy as np


def get_domain(domain, domain_file, problem_file, vis_args = {}):

    assert domain in VISUALIZERS.keys()

    generator = StateEnumerator(dom=domain_file, prob=problem_file)
    n_states = generator.graph.number_of_nodes()
    visualizer = VISUALIZERS[domain](generator, **vis_args)

    return visualizer, n_states


def prepare_dataloader(
    train_dataset: Dataset,
    val_dataset: Optional[Dataset] = None,
    test_dataset: Optional[Dataset] = None,
    batch_size: int = 64, 
    num_workers: int = 4,
) -> List[DataLoader]:

    def train_collate(data):
        '''
        data: list where each element is [index, img1, img2, state_ind]
        '''
        data = list(zip(*data))
        data = [torch.tensor(data[0]), [torch.tensor(np.array(img)).float() for img in data[1:-1]], torch.tensor(data[-1])]
        return data
    
    def test_collate(data):
        '''
        data: list where each element is [img, state_ind]
        '''
        data = list(zip(*data))
        data = [torch.tensor(np.array(data[0])).float(), torch.tensor(data[1])]
        return data

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=train_collate,
    )

    loaders = [train_loader]

    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=test_collate,
        )
        loaders.append(val_loader)
    
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=test_collate,
        )
        loaders.append(test_loader)

    return loaders


def process_img(img):
        array_from_img = np.asarray(img).transpose(2, 0, 1)
        normalized = (array_from_img / 127.5) - 1
        return normalized


def visualize_trace(trace, vis, img_size):

    # Get visualizations
    state_vis = []
    for step in trace:
        state_vis.append(process_img(vis(step, size=img_size)))
    state_vis = np.array(state_vis)

    # state_vis: (n_data, 3, img_w, img_h)
    return state_vis