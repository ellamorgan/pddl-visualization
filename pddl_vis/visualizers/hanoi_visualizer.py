import pickle
import random
import numpy as np
import copy
from PIL.Image import Image
from PIL import Image as Img
from typing import Union, Tuple, List
from macq.trace import State, Step, Trace
from macq.generate.pddl import VanillaSampling, Generator
from .base_visualizer import Visualizer


class HanoiVisualizer(Visualizer):

    def __init__(
        self, 
        generator: Generator,
    ) -> None:

        super().__init__(generator)

        self.mnist_data = pickle.load(open("data/mnist_data.pkl", "rb"))
        tile_h, tile_w, _ = self._sample_mnist(0).shape

        atoms = generator.problem.init.as_atoms()

        disks = set()
        pegs = set()
        msg = "Please make sure the disks are labelled d1 through d(n) and are decreasing in size, and pegs are labelled as peg(n) (i.e. peg1)"

        for atom in atoms:
            if atom.predicate.name == 'smaller':
                if atom.subterms[0].name[0] == 'd':
                    assert atom.subterms[1].name[0] == 'd', msg
                    assert atom.subterms[0].name[1:].isnumeric() and atom.subterms[1].name[1:].isnumeric(), msg
                    assert int(atom.subterms[0].name[1:]) > int(atom.subterms[1].name[1:]), msg
                    disks.add(atom.subterms[0].name)
                    disks.add(atom.subterms[1].name)
                else:
                    assert atom.subterms[0].name[:3] == 'peg' and atom.subterms[1].name[0] == 'd', msg
                    assert atom.subterms[0].name[3:].isnumeric() and atom.subterms[1].name[1:].isnumeric(), msg
                    pegs.add(atom.subterms[0].name)
                    disks.add(atom.subterms[1].name)
        
        disks = sorted(list(disks))
        pegs = sorted(list(pegs))
        assert len(disks) <= 10, "Only have support for up to 10 disks"

        self.disks = disks
        self.pegs = pegs
        self.n_disks = len(disks)
        self.n_pegs = len(pegs)
        self.tile_w = tile_w
        self.tile_h = tile_h
        self.imgs = [self._sample_mnist(i) for i in range(len(disks))]
    

    def _sample_mnist(
        self, 
        num: int,
    ) -> np.ndarray:
        sample = random.choice(self.mnist_data[str(num)])
        sample = Img.fromarray(sample)
        sample = np.array(sample)[:, :, np.newaxis]
        sample = np.tile(sample, 3)
        return sample


    def visualize_state(
        self, 
        state: Union[Step, State], 
        out_path: Union[str, None] = None, 
        memory: bool = False, 
        size: Union[Tuple[int, int], None] = None, 
        lightscale: bool = False,
    ) -> Union[np.ndarray, Image]:

        if isinstance(state, Step):
            state = state.state

        state_vis = np.zeros((self.tile_h * self.n_disks, self.tile_w * self.n_pegs, 3))

        top_pos = []
        bottom_pos = []
        clear = []

        for fluent, v in state.items():
            if v:
                if fluent.name == 'on':
                    top_pos.append(fluent.objects[0].name)
                    bottom_pos.append(fluent.objects[1].name)
                elif fluent.name == 'clear':
                    clear.append(fluent.objects[0].name)

        for i, peg in enumerate(self.pegs):
            find_next = peg
            h = self.n_disks
            while True:
                if find_next in clear:
                    break
                ind = bottom_pos.index(find_next)
                find_next = top_pos[ind]
                h -= 1
                if memory:
                    img = self.imgs[int(find_next[1:]) - 1]
                else:
                    img = self._sample_mnist(int(find_next[1:]) - 1)
                state_vis[h * self.tile_h : (h + 1) * self.tile_h, i * self.tile_w : (i + 1) * self.tile_w] = img            

        img = Img.fromarray(state_vis.astype('uint8'), 'RGB')
        
        if lightscale:
            img = img.convert('L')
            
        if size is not None:
            img = img.resize(size)

        if out_path is not None:
            img.save(out_path)

        return img
    



if __name__ == '__main__':

    domain_file = "data/pddl/hanoi/hanoi.pddl"
    problem_file = "data/pddl/hanoi/problems/hanoi3.pddl"

    generator = VanillaSampling(
        dom=domain_file, 
        prob=problem_file,
        plan_len=5120,
        num_traces=1
    )