from asyncio import streams
from torch.utils.data.dataset import Dataset
from macq.generate.pddl import StateEnumerator

class BaseDataset(Dataset):

    def __init__(
        self,
        domain_file: str,
        problem_file: str,
        n_samples: int,
    ) -> None:

        generator = StateEnumerator(dom=domain_file, prob=problem_file)
        self.data = [generator.tarski_state_to_macq(state) for state in generator.graph.nodes]
        self.targets = list(range(len(self.data)))