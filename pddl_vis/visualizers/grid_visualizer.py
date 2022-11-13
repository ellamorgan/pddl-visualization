import pickle
import random
import numpy as np
from PIL.Image import Image
from PIL import Image as Img
import math
from typing import Union, Tuple, List
from macq.trace import State, Step, Trace
from macq.generate.pddl import Generator


class GridVisualizer:

    def __init__(self, 
        generator : Generator, 
        square_width : int = 50, 
        div_width : int = 1, 
        door_width : int = 6, 
        key_size : int = 15, 
        robot_size : int = 17
    ) -> None:

        self.mnist_data = pickle.load(open("data/mnist_data.pkl", "rb"))
        robot_img = Img.open("data/robot.jpg").convert('RGB')
        self.robot_img = np.array(robot_img.resize((robot_size, robot_size)))

        # Assumes
        # (shape triangle) (shape diamond) (shape square) (shape circle)
        self.key_shapes = {'triangle' : 0, 'diamond' : 1, 'square' : 2, 'circle' : 3}

        max_x = 0
        max_y = 0
        transitions = []
        stack = 0

        possible_actions = generator.op_dict.keys()
        atoms = generator.problem.init.as_atoms()

        # Get all transitions between rooms
        for action in possible_actions:

            action_name = action.split(' ')[0][1:]

            if action_name == 'move':
                l1, l2 = action.split(' ')[1:]
                l1x, l1y = l1.split('-')
                l2x, l2y = l2.split('-')
                l1x, l1y, l2x, l2y = int(l1x[4:]), int(l1y), int(l2x[4:]), int(l2y[:-1])

                if l1x > max_x:
                    max_x = l1x
                if l1y > max_y:
                    max_y = l1y

                transition = ((l1x, l1y), (l2x, l2y))
                if transition in transitions:
                    raise Exception("Found more than one of the same transitions")
                elif ((l2x, l2y), (l1x, l1y)) in transitions:
                    stack -= 1
                else:
                    stack += 1
                transitions.append(transition)
        
        # Get the shapes of all keys
        key_dict = dict()
        key_types = dict()
        for atom in atoms:
            if atom.predicate.name == 'key-shape':
                key_type = atom.subterms[1].name
                img = self._sample_mnist(self.key_shapes[key_type], key_size)
                key_dict[atom.subterms[0].name] = img
                key_types[atom.subterms[0].name] = key_type

        self.keys = key_dict
        self.key_types = key_types

        if stack != 0:
            raise Exception("All transitions don't have an opposite")
            
        self.transitions = transitions
        self.width = max_x + 1
        self.height = max_y + 1
        self.square_w = square_width
        self.div_w = div_width
        self.door_w = door_width
        self.key_size = key_size
        self.board = self._generate_board()
        self.obj_pos = dict()
    

    def _sample_mnist(self, num, key_size):
        sample = random.choice(self.mnist_data[str(num)])
        sample = Img.fromarray(sample)
        sample = np.array(sample.resize((key_size, key_size)))[:, :, np.newaxis]
        sample = np.tile(sample, 3)
        return sample

    
    def _generate_board(self):

        board = np.zeros((self.square_w * self.width + self.div_w * (self.width + 1) + self.key_size, self.square_w * self.height + self.div_w * (self.height + 1), 3))

        board[:-self.key_size, : self.div_w, :] = [255, 255, 255]
        board[:-self.key_size, -self.div_w:, :] = [255, 255, 255]
        board[: self.div_w, :, :] = [255, 255, 255]
        board[-self.div_w - self.key_size:-self.key_size, :, :] = [255, 255, 255]

        for div in range(1, self.width):
            x = self.square_w * div + self.div_w * div
            board[x : x + self.div_w, :, :] = [255, 255, 255]
        for div in range(1, self.height):
            y = self.square_w * div + self.div_w * div
            board[:-self.key_size, y : y + self.div_w, :] = [255, 255, 255]

        # Creates doors
        for (x1, y1), (x2, y2) in self.transitions:
            x_d = x2 - x1
            y_d = y2 - y1
            if x_d == 1 and y_d == 0:
                x = self.square_w * (x1 + 1) + self.div_w * (x1 + 1)
                y = self.square_w * y1 + math.floor(self.square_w / 2) + self.div_w * (y1 + 1)
                board[x : x + self.div_w, y - self.door_w : y + self.door_w, :] = 0
            elif y_d == 1 and x_d == 0:
                assert x1 == x2
                x = self.square_w * x1 + math.floor(self.square_w / 2) + self.div_w * (x1 + 1)
                y = self.square_w * (y1 + 1) + self.div_w * (y1 + 1)
                board[x - self.door_w : x + self.door_w, y : y + self.div_w, :] = 0
            elif (not (x_d == -1 and y_d == 0)) and (not (y_d == -1 and x_d == 0)):
                raise Exception("Invalid transition found, can't transition from (%d, %d) to (%d, %d)" % (x1, y1, x2, y2))
            
        
        return board
    

    def _place_obj(self, img, name, loc, state_vis, memory=True):

        counter = 0

        def rand_pos(img_len):
            return np.random.randint(0, self.square_w - img_len)

        if name in self.obj_pos and memory:
            x1, y1 = self.obj_pos[name]
        else:
            x1 = self.square_w * loc[0] + self.div_w * loc[0] + rand_pos(len(img))
            y1 = self.square_w * loc[1] + self.div_w * loc[1] + rand_pos(len(img))

            while np.sum(state_vis[x1 : x1 + len(img), y1 : y1 + len(img), :]) > 1000 and counter < 100:
                x1 = self.square_w * loc[0] + self.div_w * loc[0] + rand_pos(len(img))
                y1 = self.square_w * loc[1] + self.div_w * loc[1] + rand_pos(len(img))
                counter += 1
            
            if counter == 100:
                print("Warning: objects may be placed overtop of each other")

        if name != 'at-robot' and memory:
            self.obj_pos[name] = (x1, y1)

        state_vis[x1 : x1 + len(img), y1 : y1 + len(img), :] += img

        return state_vis


    def visualize_state(
        self, 
        state: Union[Step, State], 
        out_path: Union[str, None] = None, 
        memory: bool = False, 
        size: Union[Tuple[int, int], None] = None, 
        lightscale: bool = False,
    ) -> Union[np.ndarray, Image]:

        if isinstance(state, Step):
            action = state.action
            state = state.state

        state_vis = np.copy(self.board)
        robot_pos = ()
        
        for fluent, v in state.items():
            if v:
            
                if fluent.name == 'locked':
                    x, y = fluent.objects[0].name.split('-')
                    x, y = int(x[4:]), int(y)

                    x1 = self.square_w * x + self.div_w * x
                    x2 = self.square_w * (x + 1) + self.div_w * x
                    y1 = self.square_w * y + self.div_w * y
                    y2 = self.square_w * (y + 1) + self.div_w * y

                    state_vis[x1 : x1 + self.div_w, y1 : y1 + self.square_w + 2 * self.div_w, :] = [255, 255, 255]
                    state_vis[x2 + self.div_w : x2 + 2 * self.div_w, y1 : y1 + self.square_w + 2 * self.div_w, :] = [255, 255, 255]
                    state_vis[x1 : x1 + self.square_w + 2 * self.div_w, y1 : y1 + self.div_w, :] = [255, 255, 255]
                    state_vis[x1 : x1 + self.square_w + 2 * self.div_w, y2 + self.div_w : y2 + 2 * self.div_w, :] = [255, 255, 255]

                elif fluent.name == 'at':
                    x, y = fluent.objects[1].name.split('-')

                    img = self._sample_mnist(self.key_shapes[self.key_types[fluent.objects[0].name]], self.key_size)

                    self._place_obj(img, fluent.objects[0].name, (int(x[4:]), int(y)), state_vis, memory=memory)

                elif fluent.name == 'at-robot':
                    x, y = fluent.objects[0].name.split('-')
                    robot_pos = (int(x[4:]), int(y))
                
                elif fluent.name == 'holding':
                    img = self._sample_mnist(self.key_shapes[self.key_types[fluent.objects[0].name]], self.key_size)
                    state_vis[-len(img):, 0: len(img), :] += img
        
        if len(robot_pos) > 0:
            self._place_obj(self.robot_img, 'at-robot', robot_pos, state_vis, memory=memory)
        
        if action is not None and memory:
            if action.name == 'pickup':
                del self.obj_pos[action.obj_params[1].name]
        
        state_vis[state_vis > 255] = 255

        img_from_array = Img.fromarray(state_vis.astype('uint8'), 'RGB')

        if lightscale:
            img_from_array = img_from_array.convert('L')

        if size is not None:
            img_from_array = img_from_array.resize(size)

        if out_path is not None:
            img_from_array.save(out_path)

        return img_from_array


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