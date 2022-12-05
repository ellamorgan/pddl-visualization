import pickle
import random
import numpy as np
from PIL.Image import Image
from PIL import Image as Img
from typing import Union, Tuple, List
from macq.trace import State, Step, Trace
from macq.generate.pddl import VanillaSampling, Generator
from .base_visualizer import Visualizer

'''

Blocks are disappearing and I'm not sure why
(I found out why - blocks can be placed on top of themselves)

'''

'''
Randomness: sampling from MNIST, no other source presently
'''



class BlocksVisualizer(Visualizer):

    def __init__(
        self, 
        generator: Generator,
        outline_width : int = 1,
    ) -> None:

        super().__init__(generator)

        self.mnist_data = pickle.load(open("data/mnist_data.pkl", "rb"))
        block_h, block_w, _ = self._sample_mnist(0).shape

        atoms = generator.problem.init.as_atoms()

        blocks = []

        for atom in atoms:
            if atom.predicate.name == 'on' or atom.predicate.name == 'ontable':
                blocks.append(atom.subterms[0].name)

        blocks.sort()
        assert len(blocks) <= 10, "Sorry we only have 10 numbers (0-9), so you can only have 10 blocks (for now)"

        self.blocks = blocks
        self.n_blocks = len(blocks)
        self.block_h = block_h + 2 * outline_width
        self.block_w = block_w + 2 * outline_width
        self.outline_width = outline_width
        self.imgs = [self._make_block(self._sample_mnist(i)) for i in range(len(blocks))]

    
    def _make_block(self, img):
        '''
        Adds a white boarder around images
        '''
        block = 255 * np.ones((img.shape[0] + 2 * self.outline_width, img.shape[1] + 2 * self.outline_width, *img.shape[2:]))
        block[self.outline_width : -self.outline_width, self.outline_width : -self.outline_width] = img
        return block
    

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

        sep_width = int(0.5 * self.block_w)

        state_vis = np.zeros((int(self.block_h * (self.n_blocks + 1.5)), (sep_width + self.block_w) * self.n_blocks + sep_width, 3))

        table_blocks = []
        top_blocks = []
        bot_blocks = []

        for fluent, v in state.items():
            if v:
                if fluent.name == 'on':
                    block = fluent.objects[0].name
                    dest = fluent.objects[1].name
                    top_blocks.append(block)
                    bot_blocks.append(dest)

                elif fluent.name == 'ontable':
                    block = fluent.objects[0].name
                    pos = self.blocks.index(block)
                    table_blocks.append((block, pos))

                elif fluent.name == 'holding':
                    w = int((state_vis.shape[1] / 2) - (self.block_w / 2))
                    block_n = self.blocks.index(fluent.objects[0].name)
                    if memory:
                        img = self.imgs[block_n]
                    else:
                        img = self._make_block(self._sample_mnist(block_n))
                    state_vis[0 : self.block_h, w : w + self.block_w] = img

        block_pos = [[] for _ in range(self.n_blocks)]
        for block, pos in table_blocks:
            block_pos[pos].append(pos)
            curr_block = block
            while curr_block in bot_blocks:
                next_block = top_blocks[bot_blocks.index(curr_block)]
                next_pos = self.blocks.index(next_block)
                block_pos[pos].append(next_pos)
                curr_block = next_block

        for i, pos in enumerate(block_pos):
            for j, block_n in enumerate(pos):
                if memory:
                    img = self.imgs[block_n]
                else:
                    img = self._make_block(self._sample_mnist(block_n))

                h = state_vis.shape[0] - j * self.block_h
                w = sep_width + i * (self.block_w + sep_width)

                state_vis[h - self.block_h : h, w : w + self.block_w] = img

        img = Img.fromarray(state_vis.astype('uint8'), 'RGB')
        
        if lightscale:
            img = img.convert('L')
            
        if size is not None:
            img = img.resize(size)

        if out_path is not None:
            img.save(out_path)

        return img
    


    def visualize_trace(
        self,
        trace: Trace,
        out_path: Union[str, None] = None,
        duration: int = 1000,
        size: Union[Tuple[int, int], None] = None,
        memory: bool = True,
    ) -> List[Image]:

        imgs = []
        for step in trace:
            imgs.append(self.visualize_state(step, memory=memory, size=size))
            
        if out_path is not None:
            imgs[0].save(out_path, save_all=True, append_images=imgs[1:], duration=duration, loop=0)
        
        return imgs


if __name__ == '__main__':

    domain_file = "data/pddl/blocks/blocks.pddl"
    problem_file = "data/pddl/blocks/problems/blocks2.pddl"

    generator = VanillaSampling(
        dom=domain_file, 
        prob=problem_file,
        plan_len=50,
        num_traces=1
    )

    #generator = Generator(dom=domain_file, prob=problem_file)
    vis = BlocksVisualizer(generator)

    step = generator.traces[0][30]
    vis.visualize_state(step, "results/blocks_before.jpg")
    step = generator.traces[0][31]
    vis.visualize_state(step, "results/blocks_after.jpg")

    #state = generator.tarski_state_to_macq(generator.problem.init)
    #vis.visualize_state(state)

    vis.visualize_trace(generator.traces[0], out_path="results/gifs/blocks_test.gif", duration=500)