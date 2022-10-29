import pickle
import random
import numpy as np
import math
import random
from PIL.Image import Image
from PIL import Image as Img
from typing import Union, Tuple, List
from macq.trace import Step, Trace
from macq.generate.pddl import StateEnumerator, VanillaSampling


class ElevatorVisualizer:

    def __init__(self, generator, person_size, div):

        self.mnist_data = pickle.load(open("data/mnist_data.pkl", "rb"))

        person_img = Img.open("data/stick_figure.png").convert('RGB')
        w, h = person_img.size
        scale = person_size / h
        self.person = (-1 * np.array(person_img.resize((int(w * scale), int(h * scale))))) + 255
        print("person is:", self.person.shape)

        atoms = generator.problem.init.as_atoms()

        people_origin = dict()
        people_destin = dict()
        people = set()

        floor_below = []
        floor_above = []
        n_floors = 1

        for atom in atoms:
            if atom.predicate.name == 'origin':
                people_origin[atom.subterms[0].name] = atom.subterms[1].name
                people.add(atom.subterms[0].name)
            elif atom.predicate.name == 'destin':
                people_destin[atom.subterms[0].name] = atom.subterms[1].name
            elif atom.predicate.name == 'above':
                floor_below.append(atom.subterms[0].name)
                floor_above.append(atom.subterms[1].name)
                n_floors += 1
        
        first_floor = list(set(floor_below).difference(set(floor_above)))[0]
        floors = [first_floor]
        curr_floor = first_floor
        for _ in range(n_floors):
            curr_floor = floor_above[floor_below.index(curr_floor)]
            floors.append(curr_floor)
        
        self.floors = list(floors)
        self.people = list(people)
        self.n_floors = len(self.floors)
        self.n_people = len(people_origin)
        self.people_origin = people_origin
        self.people_destin = people_destin
        self.div = div
        
        self.board, self.squares, self.square_h, self.square_w = self._generate_board()
    

    def _generate_board(self):
        # Dimensions are (h, w, 3)

        person_h = self.person.shape[0]
        person_w = self.person.shape[1]

        squares = math.ceil(math.sqrt(self.n_people))
        square_h = squares * person_h
        square_w = squares * person_w

        board = np.zeros((self.n_floors * square_h + (self.n_floors + 1) * self.div, 2 * square_w + 3 * self.div, 3))

        for i in range(self.n_floors + 1):
            board[i * (square_h + self.div) : i * (square_h + self.div) + self.div, :, :] = 255
        for i in range(3):
            board[:, i * (square_w + self.div) : i * (square_w + self.div) + self.div, :] = 255
        

        '''
        free_spots = [(i, j) for i in range(squares) for j in range(squares)]
        for i in range(self.n_floors):
            free_spots = [(i, j) for i in range(squares) for j in range(squares)]
            for _ in range(self.n_people):
                # Fill elevator and floor with people

                random.shuffle(free_spots)
                pos = free_spots.pop()

                h_pos = pos[0] * person_h + self.div
                w_pos = pos[1] * person_w + self.div
                floor_pos = i * (square_h + self.div)

                board[h_pos + floor_pos : h_pos + floor_pos + person_h, w_pos : w_pos + person_w] = self.person
        '''

        #img_from_array = Img.fromarray(board.astype('uint8'), 'RGB')
        #img_from_array.save("results/board.jpg")

        return board, squares, square_h, square_w
    

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
        
        state = step.state

        state_vis = np.copy(self.board)

        # A person can be on starting floor (0), boarded (1), or served (2)
        person_status = {k : 0 for k in self.people}

        for fluent, v in state.items():
            if v:
                if fluent.name == 'lift-at':
                    lift_floor = self.floors.index(fluent.objects[0].name)
                    for i in range(self.n_floors):
                        if i != lift_floor:
                            h = (self.square_h + self.div) * i + self.div
                            w = 2 * self.div + self.square_w
                            state_vis[h : h + self.square_h, w : w + self.square_w, :] = 255
                elif fluent.name == 'boarded':
                    person_status[fluent.objects[0].name] = 1
                elif fluent.name == 'served':
                    person_status[fluent.objects[0].name] = 2
        
        for person in self.people:
            status = person_status[person]
            if status == 0:
                floor = self.people_origin[person]
            elif status == 1:
                floor = lift_floor
            else:
                floor = self.people_destin[person]

    

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



if __name__ == '__main__':

    domain_file = "data/pddl/elevator.pddl"
    problem_file = "data/pddl/elevator-1.pddl"

    generator = StateEnumerator(dom=domain_file, prob=problem_file)
    states = list(generator.graph.nodes())
    print(f"There are {len(states)} states")

    trace_generator = VanillaSampling(
        dom=domain_file, 
        prob=problem_file,
        plan_len=5,
        num_traces=1
    )

    vis = ElevatorVisualizer(generator, person_size=10)

    vis._generate_board(n_people=5, n_floors=4, div=2)

    for step in trace_generator.traces[0]:
        vis.visualize_state(step)