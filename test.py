
import networkx as nx
from macq.generate.pddl import StateEnumerator as SE

se = SE(dom='grid.pddl', prob='grid_prob.pddl')

print(type(se))
print(type(se.graph))
print(type(se.graph.edges(data=True)))

u, v, act = next(iter(se.graph.edges(data=True)))

print(type(u))      # tarski.model.Model
print(type(act['label']))

nodes = [node for node in se.graph.nodes]

print(len(nodes))
print(type(nodes[0]))

exit()

for atom in nodes[0]:
    print(atom)
    print(type(atom))

# Used to remove all the action parameters
for _, _, act in se.graph.edges(data=True):
    act['label'] = str(act['label']).split('(')[0]

# Write the graph dot file
nx.nx_pydot.write_dot(se.graph, "graph.dot")

# Make the nodes legible (ignore state details)
with open("graph.dot", 'r') as f:
    graph_string = f.read()
graph_string = graph_string.replace('strict digraph  {', 'strict digraph  {\nnode [shape=point];')
with open('graph.dot', 'w') as f:
    f.write(graph_string)

print(str(se.graph) + "\n")

print("\t{ Example state from node }\n")
node = list(se.graph.nodes)[0]
for fluent in node.as_atoms():
    print(str(fluent))
print()
