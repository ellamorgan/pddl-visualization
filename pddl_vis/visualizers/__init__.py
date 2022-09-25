from .grid_visualizer import GridVisualizer
from .slide_tile_visualizer import SlideTileVisualizer


VISUALIZERS = {
    "grid": GridVisualizer,
    "slide_tile": SlideTileVisualizer,
}

__all__ = ["GridVisualizer", "SlideTileVisualizer"]