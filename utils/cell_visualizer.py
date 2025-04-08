import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Tuple
from utils.Plotter import Plotter  # Changed to absolute import

class CellVisualizer:
    """Class for visualizing single cell activity patterns."""
    
    def __init__(self):
        self.event_frames = np.array([6., 38., 70., 131., 145.])
        self.event_labels = ['Sound 1', 'Sound 2', 'Sound 3', 'Turn', 'Reward']
        self.default_colors = sns.color_palette('husl', 8)
        # Initialize plotter with empty dictionaries for colors and labels
        self.plotter = Plotter({}, {})
    
    def plot_informative_cell(self, 
                            aligned_imaging: np.ndarray,
                            cell_id: int,
                            condition_indices: List[int],
                            title: Optional[str] = None,
                            save_path: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot activity of an informative cell across trials.
        
        Parameters:
        -----------
        aligned_imaging : np.ndarray
            Shape (trials, neurons, frames)
        cell_id : int
            Index of cell to plot
        condition_indices : list
            Indices of trials for specific condition
        """
        cell_data = aligned_imaging[condition_indices, cell_id, :]
        
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Plot individual trials
        for trial in range(len(condition_indices)):
            ax.plot(cell_data[trial], alpha=0.3, color='gray')
        
        # Plot mean with SEM
        mean_trace = np.mean(cell_data, axis=0)
        sem_trace = np.std(cell_data, axis=0) / np.sqrt(len(condition_indices))
        ax.plot(mean_trace, color='k', linewidth=2, label='Mean')
        ax.fill_between(np.arange(len(mean_trace)), 
                       mean_trace - sem_trace,
                       mean_trace + sem_trace,
                       alpha=0.2, color='k')
        
        # Add event markers
        for frame, label in zip(self.event_frames, self.event_labels):
            ax.axvline(x=frame, color='r', linestyle='--', alpha=0.5)
            # ax.text(frame, ax.get_ylim()[1], label, 
            #        rotation=45, ha='right', va='bottom')
        
        ax.set_xlabel('Frames')
        ax.set_ylabel('ΔF/F')
        if title:
            ax.set_title(title)
        
        plt.tight_layout()

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_box_aspect(1)

        self.plotter.plot_with_seconds(0, data.shape[1], 30)

        if save_path:
            plt.savefig(save_path)
        
        return fig, ax