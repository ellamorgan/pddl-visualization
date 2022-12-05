from macq.generate.pddl import StateEnumerator
import networkx as nx
from PIL import Image
Image.MAX_IMAGE_PIXELS = 1000000000
from io import BytesIO

domain = 'hanoi'

for problem in ['hanoi1', 'hanoi2', 'hanoi3']:

    domain_file = 'data/pddl/' + domain + '/' + domain + '.pddl'
    problem_file = 'data/pddl/' + domain + '/problems/' + problem + '.pddl'
    img_path = 'results/state_spaces/' + domain + '_' + problem + '.jpg'

    generator = StateEnumerator(dom = domain_file, prob = problem_file)

    states = list(generator.graph.nodes())
    print(f'There are {len(states)} states')
    state_mapping = dict(zip(states, range(len(states))))
    state_graph = nx.relabel_nodes(generator.graph, state_mapping)

    actions = []
    colours = ['#72EC91', '#C07DFF', '#4D7DEE', '#FF5A5F', '#C81D25']

    # Used to remove all the action parameters
    for _, _, act in state_graph.edges(data=True):
        action = str(act['label']).split('(')[0]

        if action in actions:
            act['color'] = colours[actions.index(action)]
        else:
            actions.append(action)
            if len(actions) > len(colours):
                print("Need more colours!")
                exit()
            act['color'] = colours[len(actions) - 1]
        act['label'] = ""


    dot_graph = nx.nx_pydot.to_pydot(state_graph)
    img = Image.open(BytesIO(dot_graph.create_png())).convert('RGB')
    img.save(img_path)