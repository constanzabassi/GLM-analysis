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
                         all_conditions: List[tuple],
                         title_base: Optional[str] = None,
                         peak_info: Optional[float] = None,
                         frames: Optional[tuple] = None,
                         save_path: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot activity of an informative cell across all conditions in a single row.
        
        Parameters:
        -----------
        aligned_imaging : np.ndarray
            Shape (trials, neurons, frames)
        cell_id : int
            Index of cell to plot
        all_conditions : list of tuples
            List of (trials, combination, label) for each condition
        title_base : str, optional
            Base title for the plot
        peak_info : float, optional
            Peak information value for this neuron
        frames : tuple, optional
            Start and end frames to plot (start_frame, end_frame)
        save_path : str, optional
            Path to save the figure
        """
        n_conditions = len(all_conditions)
        fig, axes = plt.subplots(1, n_conditions, figsize=(3*n_conditions, 3))
        
        # Handle single condition case
        if n_conditions == 1:
            axes = [axes]
        
        for ax, (trials, _, label) in zip(axes, all_conditions):
            cell_data = aligned_imaging[trials, cell_id, :]

            # Apply frame selection if provided
            if frames is not None:
                start_frame, end_frame = frames
                cell_data = cell_data[:, start_frame:end_frame]
                
            # Plot individual trials
            for trial in range(len(trials)):
                ax.plot(cell_data[trial], alpha=0.3, color='gray')
            
            # Plot mean with SEM
            mean_trace = np.mean(cell_data, axis=0)
            sem_trace = np.std(cell_data, axis=0) / np.sqrt(len(trials))
            ax.plot(mean_trace, color='k', linewidth=2, label='Mean')
            ax.fill_between(np.arange(len(mean_trace)), 
                        mean_trace - sem_trace,
                        mean_trace + sem_trace,
                        alpha=0.2, color='k')
            
            # Add event markers
            for frame, event_label in zip(self.event_frames, self.event_labels):
                ax.axvline(x=frame, color='r', linestyle='--', alpha=0.5)
            
            # Customize each subplot
            ax.set_xlabel('Frames')
            if ax == axes[0]:  # Only add y-label to first subplot
                ax.set_ylabel('ΔF/F')
            ax.set_title(f'{label}', fontsize=10)
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_box_aspect(1)
            
            # Add time axis
            self.plotter.plot_with_seconds(0, len(mean_trace), 30)
        
        # Add overall title if provided
        if title_base:
            if peak_info is not None:
                title = f"{title_base}\nPeak Info: {peak_info:.3f} bits"
            else:
                title = title_base
            fig.suptitle(title, y=1.05, fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            # plt.savefig(f'{save_path}.pdf', bbox_inches='tight', dpi=300)
            plt.savefig(f'{save_path}.png', bbox_inches='tight', dpi=300)
            # plt.close()
        
        return fig, axes

    def plot_informative_cell_overlay(self, 
                                  aligned_imaging: np.ndarray,
                                  cell_id: int,
                                  all_conditions: List[tuple],
                                  condition_colors: List[str],
                                  title_base: Optional[str] = None,
                                  peak_info: Optional[float] = None,
                                  frames: Optional[tuple] = None,
                                  save_path: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Overlay mean ± SEM activity of a cell across all conditions on the same plot.

        Parameters:
        -----------
        aligned_imaging : np.ndarray
            Shape (trials, neurons, frames)
        cell_id : int
            Index of cell to plot
        all_conditions : list of tuples
            List of (trials, combination, label) for each condition
        condition_colors : list of str
            List of colors corresponding to each condition
        title_base : str, optional
            Title prefix
        peak_info : float, optional
            Peak info value for title
        frames : tuple, optional
            Start and end frames to plot (start_frame, end_frame)
        save_path : str, optional
            If provided, figure is saved here

        Returns:
        --------
        fig, ax : Tuple of matplotlib Figure and Axes
        """

        fig, ax = plt.subplots(figsize=(4, 3))

        for (trials, _, label), color in zip(all_conditions, condition_colors):
            cell_data = aligned_imaging[trials, cell_id, :]

            # Apply frame selection
            if frames is not None:
                start_frame, end_frame = frames
                cell_data = cell_data[:, start_frame:end_frame]

            mean_trace = np.mean(cell_data, axis=0)
            sem_trace = np.std(cell_data, axis=0) / np.sqrt(len(trials))

            ax.plot(mean_trace, color=color, linewidth=2, label=label)
            ax.fill_between(np.arange(len(mean_trace)),
                            mean_trace - sem_trace,
                            mean_trace + sem_trace,
                            alpha=0.3,
                            color=color)

        # Add event markers
        for frame, event_label in zip(self.event_frames, self.event_labels):
            ax.axvline(x=frame, color='r', linestyle='--', alpha=0.5)

        # Labels and style
        ax.set_xlabel('Frames')
        ax.set_ylabel('ΔF/F')
        title = f"{title_base or 'Cell'} {cell_id}"
        if peak_info is not None:
            title += f" | Peak info: {peak_info:.2f}"
        ax.set_title(title, fontsize=12)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_box_aspect(1)
        ax.legend(frameon=False, fontsize=9)

        # Optional time axis
        self.plotter.plot_with_seconds(0, len(mean_trace), 30)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')

        return fig, ax

    
    # def plot_informative_cell(self, 
    #                         aligned_imaging: np.ndarray,
    #                         cell_id: int,
    #                         condition_indices: List[int],
    #                         title: Optional[str] = None,
    #                         save_path: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    #     """
    #     Plot activity of an informative cell across trials.
        
    #     Parameters:
    #     -----------
    #     aligned_imaging : np.ndarray
    #         Shape (trials, neurons, frames)
    #     cell_id : int
    #         Index of cell to plot
    #     condition_indices : list
    #         Indices of trials for specific condition
    #     """
    #     cell_data = aligned_imaging[condition_indices, cell_id, :]
        
    #     n_conditions = len(all_conditions)
    #     fig, axes = plt.subplots(1, n_conditions, figsize=(3*n_conditions, 3))
    #     if n_conditions == 1:
    #         axes = [axes]

    #     for ax, (trials, _, label) in zip(axes, all_conditions):
    #         cell_data = aligned_imaging[trials, cell_id, :]
        
    #     # Plot individual trials
    #     for trial in range(len(trials)):
    #         ax.plot(cell_data[trial], alpha=0.3, color='gray')
        
    #     # Plot mean with SEM
    #     mean_trace = np.mean(cell_data, axis=0)
    #     sem_trace = np.std(cell_data, axis=0) / np.sqrt(len(trials))
    #     ax.plot(mean_trace, color='k', linewidth=2, label='Mean')
    #     ax.fill_between(np.arange(len(mean_trace)), 
    #                    mean_trace - sem_trace,
    #                    mean_trace + sem_trace,
    #                    alpha=0.2, color='k')
        
    #     # Add event markers
    #     for frame, event_label in zip(self.event_frames, self.event_labels):
    #         ax.axvline(x=frame, color='r', linestyle='--', alpha=0.5)
        
    #     # Customize each subplot
    #     ax.set_xlabel('Frames')
    #     if ax == axes[0]:  # Only add y-label to first subplot
    #         ax.set_ylabel('ΔF/F')
    #     ax.set_title(f'{label}', fontsize=10)
        
    #     ax.spines['top'].set_visible(False)
    #     ax.spines['right'].set_visible(False)
    #     ax.set_box_aspect(1)
        
    #     # Add time axis
    #     self.plotter.plot_with_seconds(0, len(mean_trace), 30)

    #     # Add overall title if provided
    #     if title_base:
    #         fig.suptitle(title_base, y=1.05, fontsize=8)

    #     if save_path:
    #         plt.savefig(save_path)
        
    #     return fig, ax