from .base_visualizer import Visualizer
from .grid_visualizer import GridVisualizer
from .slide_tile_visualizer import SlideTileVisualizer
from .hanoi_visualizer import HanoiVisualizer
from .elevator_visualizer import ElevatorVisualizer
from .blocks_visualizer import BlocksVisualizer


VISUALIZERS = {
    "grid" : GridVisualizer,
    "slide_tile" : SlideTileVisualizer,
    "hanoi" : HanoiVisualizer,
    "elevator" : ElevatorVisualizer,
    "blocks" : BlocksVisualizer
}

__all__ = ["Visualizer", "GridVisualizer", "SlideTileVisualizer", "HanoiVisualizer", "ElevatorVisualizer", "BlocksVisualizer"]