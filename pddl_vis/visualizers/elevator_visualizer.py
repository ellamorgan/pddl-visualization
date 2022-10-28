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

    def __init__(self, generator, person_size):

        self.mnist_data = pickle.load(open("data/mnist_data.pkl", "rb"))

        person_img = Img.open("data/stick_figure.png").convert('RGB')
        w, h = person_img.size
        scale = person_size / h
        self.person = (-1 * np.array(person_img.resize((int(w * scale), int(h * scale))))) + 255
        print("person is:", self.person.shape)

        possible_actions = generator.op_dict.keys()
        atoms = generator.problem.init.as_atoms()

        people_origin = dict()
        people_destin = dict()
        floors = set()

        for atom in atoms:
            if atom.predicate.name == 'origin':
                people_origin[atom.subterms[0].name] = atom.subterms[1].name
            elif atom.predicate.name == 'destin':
                people_destin[atom.subterms[0].name] = atom.subterms[1].name
            elif atom.predicate.name == 'above':
                floors.add(atom.subterms[0].name)
                floors.add(atom.subterms[1].name)
        
        self.floors = list(floors)
        self.n_floors = len(self.floors)
        self.n_people = len(people_origin)
    

    def _generate_board(self, n_people, n_floors, div):
        # Dimensions are (h, w, 3)

        person_h = self.person.shape[0]
        person_w = self.person.shape[1]

        squares = math.ceil(math.sqrt(n_people))
        square_h = squares * person_h
        square_w = squares * person_w

        board = np.zeros((n_floors * square_h + (n_floors + 1) * div, 2 * square_w + 3 * div, 3))

        for i in range(n_floors + 1):
            board[i * (square_h + div) : i * (square_h + div) + div, :, :] = 255
        for i in range(3):
            board[:, i * (square_w + div) : i * (square_w + div) + div, :] = 255
        

        free_spots = [(i, j) for i in range(squares) for j in range(squares)]
        
        for i in range(n_floors):
            free_spots = [(i, j) for i in range(squares) for j in range(squares)]
            for _ in range(n_people):
                # Fill elevator and floor with people

                random.shuffle(free_spots)
                pos = free_spots.pop()

                h_pos = pos[0] * person_h + div
                w_pos = pos[1] * person_w + div
                floor_pos = i * (square_h + div)

                board[h_pos + floor_pos : h_pos + floor_pos + person_h, w_pos : w_pos + person_w] = self.person

        img_from_array = Img.fromarray(board.astype('uint8'), 'RGB')
        img_from_array.save("results/board.jpg")
    

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

        print("\n\n")

        # A person can be on starting floor, served, or boarded

        for fluent, v in state.items():
            if v:
                if fluent.name == 'lift-at':
                    print(f"lift at {fluent.objects[0].name}")
                elif fluent.name == 'boarded':
                    print(f"person {fluent.objects[0].name} has boarded")
                else:
                    print(fluent.name)
            else:
                print(fluent.name)
    

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