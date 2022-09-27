from pddl_vis.visualizers import SlideTileVisualizer
from macq.generate.pddl import StateEnumerator

domain_file = 'data/pddl/slidetile.pddl'
problem_file = 'data/pddl/slidetile_data.pddl'



generator = StateEnumerator(dom=domain_file, prob=problem_file, num_nodes=200)

vis = SlideTileVisualizer(generator)

state = generator.tarski_state_to_macq(generator.problem.init)
vis.visualize_state(state)