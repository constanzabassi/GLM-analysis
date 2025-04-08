from .path_utils import setup_paths

__all__ = ['setup_paths']
# # First stage - only path setup
# from .path_utils import setup_paths

# # Function to initialize other modules after paths are set
# def init_modules():
#     """Initialize modules after path setup."""
#     # Import other modules only after paths are set up
#     from .data_utils import load_experimental_data
#     from .alignment_utils import setup_and_align_data
#     from .cell_visualizer import CellVisualizer
    
#     return {
#         'load_experimental_data': load_experimental_data,
#         'setup_and_align_data': setup_and_align_data,
#         'CellVisualizer': CellVisualizer
#     }

# # Only export setup_paths initially
# __all__ = ['setup_paths', 'init_modules']