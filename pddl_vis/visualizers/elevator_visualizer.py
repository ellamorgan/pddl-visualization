import pickle
import random
import numpy as np
from PIL.Image import Image
from typing import Union, Tuple, List
from macq.trace import Step, Trace


class ElevatorVisualizer:

    def __init__(self, generator):

        self.mnist_data = pickle.load(open("data/mnist_data.pkl", "rb"))

        possible_actions = generator.op_dict.keys()
        atoms = generator.problem.init.as_atoms()

        print(possible_actions)
        print(atoms)
    

    def sample_mnist(
        self, 
        num: int, 
        key_size: Tuple[int, int],
    ) -> np.ndarray:
        sample = random.choice(self.mnist_data[str(num)])
        sample = Image.fromarray(sample)
        sample = np.array(sample.resize(key_size))[:, :, np.newaxis]
        sample = np.tile(sample, 3)
        return sample
    

    def visualize_state(
        self, 
        step: Step, 
        out_path: Union[str, None] = None, 
        memory: bool = False, 
        size: Union[Tuple[int, int], None] = None, 
        lightscale: bool = False,
        as_image: bool = False,
    ) -> Union[np.ndarray, Image]:
        return np.array()
    

    def visualize_trace(
        self,
        trace: Trace,
        out_path: Union[str, None] = None,
        duration: int = 1000,
        size: Union[Tuple[int, int], None] = None,
    ) -> List[Image]:

        imgs = []
        for step in trace:
            imgs.append(self.visualize_state(step, memory=True, size=size, as_image=True))
            
        if out_path is not None:
            imgs[0].save(out_path, save_all=True, append_images=imgs[1:], duration=duration, loop=0)
        
        return imgs