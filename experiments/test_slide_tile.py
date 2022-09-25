from pddl_vis.visualizers import SlideTileVisualizer
from macq.generate.pddl import StateEnumerator
import time

domain_file = 'data/pddl/slidetile.pddl'
problem_file = 'data/pddl/slidetile_data.pddl'

start = time.time()

generator = StateEnumerator(dom=domain_file, prob=problem_file)

end = time.time()

print("generator done after %ds" % (end - start))

#vis = SlideTileVisualizer(generator)