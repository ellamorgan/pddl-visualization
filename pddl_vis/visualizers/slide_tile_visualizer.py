import pickle
import random
import numpy as np
import re
from PIL.Image import Image
from PIL import Image as Img
from typing import Union, Tuple, List
from macq.trace import State, Step, Trace
from macq.generate.pddl import Generator
from .base_visualizer import Visualizer


class SlideTileVisualizer(Visualizer):

    def __init__(
        self, 
        generator: Generator,
    ) -> None:
        '''
        Expect tiles to be formatted as t#, and coordinates formatted as x# and y#, where # is an integer
        '''
        super().__init__(generator)

        self.mnist_data = pickle.load(open("data/mnist_data.pkl", "rb"))
        tile_w, tile_h, _ = self._sample_mnist(0).shape

        atoms = generator.problem.init.as_atoms()

        width, height = 0, 0

        for atom in atoms:
            if atom.predicate.name == 'position':
                msg = "Position names not formatted correctly, need to be of format x# or y#, where # is a number, starting from 1. Ex: x1, x2, x3, y1, y2"
                assert atom.subterms[0].name[1:].isnumeric(), msg
                assert atom.subterms[0].name[1:] != '0', msg
                assert atom.subterms[0].name[0] == 'x' or atom.subterms[0].name[0] == 'y', msg
                if atom.subterms[0].name[0] == 'x':
                    width += 1
                else:
                    height += 1

        assert width * height < 10, "Sorry, support for puzzles with more than 10 tiles isn't supported yet"

        self.imgs = [self._sample_mnist(i) for i in range(width * height)]
        self.width = width
        self.height = height
        self.tile_w = tile_w
        self.tile_h = tile_h
    


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

        state_vis = np.zeros((self.width * self.tile_w, self.height * self.tile_h, 3))
        
        def get_num(obj):
            return int(obj.name[1:])

        for fluent, v in state.items():
            if v:
                if fluent.name == 'at':
                    t, w, h = map(get_num, fluent.objects)
                    if memory:
                        img = self.imgs[t]
                    else:
                        img = self._sample_mnist(t)
                    state_vis[(w - 1) * self.tile_w : w * self.tile_w, (h - 1) * self.tile_h : h * self.tile_h] = img
        
        img = Img.fromarray(state_vis.astype('uint8'), 'RGB')
        
        if lightscale:
            img = img.convert('L')
            
        if size is not None:
            img = img.resize(size)

        if out_path is not None:
            img.save(out_path)

        return img
    