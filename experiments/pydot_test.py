import networkx as nx
from macq.generate.pddl import StateEnumerator as SE
import pydot



se = SE(dom='data/pddl/grid.pddl', prob='data/pddl/grid_data.pddl')

# dot -Tpng graph.dot > graph.png
# [fillcolor="#bfbfbf", style=filled]

# Used to remove all the action parameters
for _, _, act in se.graph.edges(data=True):
    action = str(act['label']).split('(')[0]
    if action == 'pickup':
        act['color'] = "#7b4af7"
    elif action == 'putdown':
        act['color'] = "#4a81f7"
    elif action == 'move':
        act['color'] = "#83e070"
    act['label'] = ""

# Write the graph dot file
print(type(nx.nx_pydot))
nx.nx_pydot.write_dot(se.graph, "results/graph_files/source.dot")

i = 1
states = dict()

with open("results/graph_files/source.dot", 'r') as f, open("results/graph_files/display.dot", 'w') as out:
    for line in f:
        if "Model" in line and "->" not in line:
            split_line = line.split('"')
            states[split_line[1]] = str(i)
            out.write("\t" + '"%d"' % (i) + '"'.join(split_line[2:]))
            i += 1
        elif "->" in line:
            split_line = line.split('"')
            s1, s2 = split_line[1], split_line[3]

            out.write("\t" + '"%s" -> "%s"'  % (states[s1], states[s2]) + '"'.join(split_line[4:]))
        else:
            out.write(line)

(graph,) = pydot.graph_from_dot_file('results/graph_files/display.dot')
print(type(graph))
graph.write_png('results/graph.png')

'''
# Make the nodes legible (ignore state details)
with open("graph.dot", 'r') as f:
    graph_string = f.read()
print(graph_string)
graph_string = graph_string.replace('strict digraph  {', 'strict digraph  {\nnode [shape=point];')
with open('graph.dot', 'w') as f:
    f.write(graph_string)
'''

'''
print(str(se.graph) + "\n")

print("\t{ Example state from node }\n")
node = list(se.graph.nodes)[0]
for fluent in node.as_atoms():
    print(str(fluent))
print()
'''