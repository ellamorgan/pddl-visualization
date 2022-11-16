import pickle
import random
import numpy as np
import math
from PIL.Image import Image
from PIL import Image as Img
from typing import Union, Tuple, List
from macq.trace import Step, Trace, State
from macq.generate.pddl import StateEnumerator, VanillaSampling
from .base_visualizer import Visualizer


class ElevatorVisualizer(Visualizer):

    def __init__(self, 
        generator : StateEnumerator, 
        person_size : int = 20, 
        div : int = 2
    ) -> None:

        self.mnist_data = pickle.load(open("data/mnist_data.pkl", "rb"))

        atoms = generator.problem.init.as_atoms()

        people_origin = dict()
        people_destin = dict()
        people = set()
        floors = set()

        msg = "Floor names not formatted correctly, need to be of format f# starting from 0, i.e. f0, f1, f2, ..."

        for atom in atoms:
            if atom.predicate.name == 'origin':
                people_origin[atom.subterms[0].name] = atom.subterms[1].name
                people.add(atom.subterms[0].name)
            elif atom.predicate.name == 'destin':
                people_destin[atom.subterms[0].name] = atom.subterms[1].name
            elif atom.predicate.name == 'above':
                f1, f2 = atom.subterms
                for f in [f1, f2]:
                    assert f.name[0] == 'f' and f.name[1:].isnumeric(), msg
                    floors.add(f.name)

        self.floors = sorted(list(floors))
        self.people = list(people)
        self.n_floors = len(self.floors)
        self.n_people = len(people_origin)
        self.people_origin = people_origin
        self.people_destin = people_destin
        self.div = div
        self.person_pos = dict()
        self.person_size = person_size

        self.board, self.squares, self.square_size = self._generate_board()
    

    def _sample_mnist(self, num, size):
        sample = random.choice(self.mnist_data[str(num)])
        sample = Img.fromarray(sample)
        sample = np.array(sample.resize((size, size)))[:, :, np.newaxis]
        sample = np.tile(sample, 3)
        return sample
    

    def _find_spot(self, floor):
        spot_found = False
        while not spot_found:
            match_found = False
            x, y = random.randrange(0, self.squares), random.randrange(0, self.squares)
            for _, v in self.person_pos.items():
                if (floor, x, y) == v:
                    match_found = True
                    break
            if not match_found:
                spot_found = True

        return (floor, x, y)
    

    def _generate_board(self):
        # Dimensions are (h, w, 3)

        squares = math.ceil(math.sqrt(self.n_people))
        square_size = squares * self.person_size

        board = np.zeros((self.n_floors * square_size + (self.n_floors + 1) * self.div, 2 * square_size + 3 * self.div, 3))

        for i in range(self.n_floors + 1):
            board[i * (square_size + self.div) : i * (square_size + self.div) + self.div, :, :] = 255
        for i in range(3):
            board[:, i * (square_size + self.div) : i * (square_size + self.div) + self.div, :] = 255

        return board, squares, square_size
    

    def _draw_board(self, people_pos, lift_floor, memory=False):

        state_vis = np.copy(self.board)

        for i in range(self.n_floors):
            if i != lift_floor:
                h = (self.square_size + self.div) * i + self.div
                w = 2 * self.div + self.square_size
                state_vis[h : h + self.square_size, w : w + self.square_size, :] = 255
        
        for i, floor in enumerate(people_pos):

            if not memory:
                free_spots = [(j, k) for j in range(self.squares) for k in range(self.squares)]

            for person in floor:

                if memory:
                    if person not in self.person_pos:
                        self.person_pos[person] = self._find_spot(i)
                    _, *pos = self.person_pos[person]
                        
                else:
                    random.shuffle(free_spots)
                    pos = free_spots.pop()

                h_pos = pos[0] * self.person_size + self.div
                w_pos = pos[1] * self.person_size + self.div

                if i == len(people_pos) - 1:
                    floor_pos = lift_floor * (self.square_size + self.div)
                    w_pos += self.square_size + self.div
                else:
                    floor_pos = i * (self.square_size + self.div)
                
                state_vis[h_pos + floor_pos : h_pos + floor_pos + self.person_size, w_pos : w_pos + self.person_size] = self._sample_mnist(self.people.index(person), self.person_size)
        
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

        # A person can be on starting floor (0), boarded (1), or served (2)
        person_status = {k : 0 for k in self.people}

        for fluent, v in state.items():
            if v:
                if fluent.name == 'lift-at':
                    lift_floor = self.floors.index(fluent.objects[0].name)
                elif fluent.name == 'boarded':
                    person_status[fluent.objects[0].name] = 1
                elif fluent.name == 'served':
                    person_status[fluent.objects[0].name] = 2
        
        # List for each floor, last for on elevator
        people_pos = [[] for _ in range(self.n_floors + 1)]
        
        for person in self.people:
            status = person_status[person]
            if status == 0:
                # At origin
                floor = self.people_origin[person]
                people_pos[self.floors.index(floor)].append(person)
            elif status == 1:
                # On elevator
                people_pos[-1].append(person)
            else:
                # At destination
                floor = self.people_destin[person]
                people_pos[self.floors.index(floor)].append(person)
        
        state_vis = self._draw_board(people_pos, lift_floor, memory)

        # action.name is the next action after this state
        if memory and action is not None:
            if action.name == 'board':
                person = action.obj_params[1].name
                self.person_pos[person] = self._find_spot(self.n_floors)                  # Last index is elevator
            elif action.name == 'depart':
                person = action.obj_params[1].name
                self.person_pos[person] = self._find_spot(self.floors.index(self.people_destin[person]))     # People can only get off at their destinations

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

    domain_file = "data/pddl/elevator/elevator.pddl"
    problem_file = "data/pddl/elevator/problems/elevator-2.pddl"

    generator = VanillaSampling(
        dom=domain_file, 
        prob=problem_file,
        plan_len=50,
        num_traces=1
    )

    vis = ElevatorVisualizer(generator)

    step = generator.traces[0][30]
    vis.visualize_state(step, "results/elevator_before.jpg")
    step = generator.traces[0][31]
    vis.visualize_state(step, "results/elevator_after.jpg")