import random
import numpy as np
from PIL.Image import Image
from typing import Union, Tuple, List, Callable
from macq.trace import State, Step, Trace
from macq.generate.pddl import Generator, StateEnumerator
from abc import abstractmethod




class Visualizer:

    def __init__(
        self, 
        generator: Generator,
    ) -> None:

        self.generator = generator


    @abstractmethod
    def visualize_state(
        self, 
        state: Union[Step, State], 
        out_path: Union[str, None] = None, 
        memory: bool = False, 
        size: Union[Tuple[int, int], None] = None, 
        lightscale: bool = False,
    ) -> Union[np.ndarray, Image]:
        pass


    def create_dataset(
        self,  
        n_samples : int,
        states : Union[List[Union[Step, State]], None] = None,
        preprocess : Union[Callable, None] = None,
        inds: List[int] = [],
        img_size: Union[Tuple[int, int], None] = None,
        seed : Union[int, None] = None,
    ) -> List[List[np.ndarray]]:

        if seed is not None:
            random.seed(seed)

        if states is None:
            assert isinstance(self.generator, StateEnumerator), "The generator needs to be a StateEnumerator to create a dataset"

            states = list(self.generator.graph.nodes())

            if len(inds) == 0:
                states = [self.generator.tarski_state_to_macq(state) for state in states]
            else:
                print(len(list(self.generator.graph.nodes)))
                states = [self.generator.tarski_state_to_macq(states[i]) for i in inds]

        if preprocess is not None:
            data = [[preprocess(self.visualize_state(state, size=img_size)) for _ in range(n_samples)] for state in states]
        else:
            data = [[np.array(self.visualize_state(state, size=img_size)) for _ in range(n_samples)] for state in states]

        return data
    

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