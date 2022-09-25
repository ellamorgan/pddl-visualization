from pddl_vis.visualizers import PDDLDataset, collate_fn, GridVisualizer
from macq.generate.pddl import StateEnumerator
from torch.utils.data import DataLoader


domain_file = "data/pddl/grid.pddl"
problem_file = "data/pddl/grid_data.pddl"
graph_generator = StateEnumerator(dom=domain_file, prob=problem_file)
vis = GridVisualizer(graph_generator, square_width=50, div_width=1, door_width=6, key_size=15, robot_size=17)

dataset = PDDLDataset(graph_generator, vis.visualize_state, n_samples=5, img_size=(24,24))
dataloader = DataLoader(
    dataset, 
    batch_size=7,
    collate_fn=collate_fn,
    drop_last=True,
    )

batch = next(iter(dataloader))

print("batch len:", len(batch))     # Should be 4
print(batch[0].shape)               # Should be tensor of shape (batch_size)
print(batch[1].shape)               # Should be tensor of shape (batch_size, 3, img_w, img_h)
print(batch[2].shape)               # Should be tensor of shape (batch_size, 3, img_w, img_h)
print(batch[3].shape)               # Should be tensor of shape (batch_size)