import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d
import seaborn as sns
from typing import List, Optional, Tuple
from utils.Plotter import Plotter  # Changed to absolute import

class CellVisualizer:
    """Class for visualizing single cell activity patterns."""
    
    def __init__(self,event_frames: Optional[np.ndarray] = None,
                 event_labels: Optional[List[str]] = None):
        """ Initialize the CellVisualizer with event frames and labels. 
        If not provided, default values are used.
        Parameters:
        -----------
        event_frames : np.ndarray, optional
            Array of event frames to plot vertical lines at
        event_labels : list of str, optional

            Labels for each event frame
        """
        if event_frames is None:
            event_frames = np.array([6., 38., 70., 131., 145.])
        if event_labels is None:
            event_labels = ['S1', 'S2', 'S3', 'T', 'R'] #['Sound 1', 'Sound 2', 'Sound 3', 'Turn', 'Reward']

        self.event_frames = event_frames
        self.event_labels = event_labels
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

    # def plot_informative_cell_overlay(self, 
    #                               aligned_imaging: np.ndarray,
    #                               cell_id: int,
    #                               all_conditions: List[tuple],
    #                               condition_colors: List[str],
    #                               title_base: Optional[str] = None,
    #                               peak_info: Optional[float] = None,
    #                               frames: Optional[tuple] = None,
    #                               save_path: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    #     """
    #     Overlay mean ± SEM activity of a cell across all conditions on the same plot.

    #     Parameters:
    #     -----------
    #     aligned_imaging : np.ndarray
    #         Shape (trials, neurons, frames)
    #     cell_id : int
    #         Index of cell to plot
    #     all_conditions : list of tuples
    #         List of (trials, combination, label) for each condition
    #     condition_colors : list of str
    #         List of colors corresponding to each condition
    #     title_base : str, optional
    #         Title prefix
    #     peak_info : float, optional
    #         Peak info value for title
    #     frames : tuple, optional
    #         Start and end frames to plot (start_frame, end_frame)
    #     save_path : str, optional
    #         If provided, figure is saved here

    #     Returns:
    #     --------
    #     fig, ax : Tuple of matplotlib Figure and Axes
    #     """

    #     fig, ax = plt.subplots(figsize=(4, 3))

    #     for (trials, _, label), color in zip(all_conditions, condition_colors):
    #         cell_data = aligned_imaging[trials, cell_id, :]

    #         # Apply frame selection
    #         if frames is not None:
    #             start_frame, end_frame = frames
    #             cell_data = cell_data[:, start_frame:end_frame]

    #         mean_trace = np.mean(cell_data, axis=0)
    #         sem_trace = np.std(cell_data, axis=0) / np.sqrt(len(trials))

    #         ax.plot(mean_trace, color=color, linewidth=2, label=label)
    #         ax.fill_between(np.arange(len(mean_trace)),
    #                         mean_trace - sem_trace,
    #                         mean_trace + sem_trace,
    #                         alpha=0.3,
    #                         color=color)

    #     # Add event markers
    #     for frame, event_label in zip(self.event_frames, self.event_labels):
    #         ax.axvline(x=frame, color='r', linestyle='--', alpha=0.5)

    #     # Labels and style
    #     ax.set_xlabel('Frames')
    #     ax.set_ylabel('ΔF/F')
    #     title = f"{title_base or 'Cell'} {cell_id}"
    #     if peak_info is not None:
    #         title += f" | Peak info: {peak_info:.2f}"
    #     ax.set_title(title, fontsize=12)

    #     ax.spines['top'].set_visible(False)
    #     ax.spines['right'].set_visible(False)
    #     ax.set_box_aspect(1)
    #     ax.legend(frameon=False, fontsize=9)

    #     # Optional time axis
    #     self.plotter.plot_with_seconds(0, len(mean_trace), 30)

    #     if save_path:
    #         plt.savefig(save_path, bbox_inches='tight')

    #     return fig, ax

    def plot_informative_cell_overlay_minimal_axis(self,aligned_imaging: np.ndarray,
                                  cell_id: int,
                                  all_conditions: List[tuple],
                                  condition_colors: List[str],
                                  title_base: Optional[str] = None,
                                  peak_info: Optional[float] = None,
                                  frames: Optional[tuple] = None,
                                  subplot_split: Optional[str] = None,
                                  legend: Optional[str] = None,
                                  figsize: Optional[Tuple[float, float]] = None,  # NEW
                                  orientation: str = "horizontal",  # NEW: "horizontal" or "vertical"
                                  smoothing: Optional[float] = None,
                                  shading: Optional[bool] = False,
                                  save_path: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Overlay mean ± SEM activity of a cell across all conditions, with optional subplot splitting.
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
        everything else: optional parameters for customization
        """

        condition_labels = [label for (_, _, label) in all_conditions]
        group_labels = self.split_condition_labels(condition_labels, subplot_split)

        n_groups = len(group_labels)
        if orientation == "vertical":
            fig, axs = plt.subplots(n_groups, 1, figsize=figsize or (5, 3 * n_groups), squeeze=False)
        else:  # horizontal
            fig, axs = plt.subplots(1, n_groups, figsize=figsize or (5 * n_groups, 3), squeeze=False)
        axs = axs.flatten()

        # fig, axs = plt.subplots(1, len(group_labels), figsize=(5 * len(group_labels), 3), squeeze=False)
        # axs = axs[0]

        for ax, (group_name, group_condition_labels) in zip(axs, group_labels.items()):
            traces = []  # store traces for y-scaling and bar placement

            for (trials, _, label), color in zip(all_conditions, condition_colors):
                if label not in group_condition_labels:
                    continue

                cell_data = aligned_imaging[trials, cell_id, :]
                if frames is not None:
                    start_frame, end_frame = frames
                    cell_data = cell_data[:, start_frame:end_frame]
                else:
                    start_frame = 0

                mean_trace = np.mean(cell_data, axis=0)
                sem_trace = np.std(cell_data, axis=0) / np.sqrt(len(trials))

                # Optional smoothing
                if smoothing is not None and smoothing > 0:
                    mean_trace = gaussian_filter1d(mean_trace, sigma=smoothing)
                    sem_trace = gaussian_filter1d(sem_trace, sigma=smoothing)

                traces.append(mean_trace)

                ax.plot(mean_trace, color=color, linewidth=1.2, label=label)

                # Uncomment the following line to add SEM shading
                if shading:
                    ax.fill_between(np.arange(len(mean_trace)),
                                    mean_trace - sem_trace,
                                    mean_trace + sem_trace,
                                    alpha=0.3,
                                    color=color)

            # Add event lines
            for frame, event_label in zip(self.event_frames, self.event_labels):
                if frames is not None and not (start_frame <= frame < end_frame):
                    continue
                ax.axvline(x=frame - start_frame, color='k', linestyle='--', alpha=0.5, ymin=0.05, ymax=0.95)

            # Remove frame spines and ticks
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

            # Title
            # if ax is axs[0]:  # Only set title for the first subplot
            #     ax.set_title(f"{title_base or 'Cell'} {cell_id}\n{group_name}" +
            #                 (f" | Peak info: {peak_info:.2f}" if peak_info else ""),
            #                 fontsize=10)
            ax.set_title(f"{title_base or 'Cell'} {cell_id}\n{group_name}" +
                            (f" | Peak info: {peak_info:.2f}" if peak_info else ""),
                            fontsize=10)

            # Legend if requested
            if legend:
                ax.legend(frameon=False, fontsize=9)

            #set y-limits
            if traces:
                all_y = np.concatenate(traces)
                y_min, y_max = np.min(all_y), np.max(all_y)
                y_range = y_max - y_min
                ax.set_ylim(y_min - .1 * y_range, y_max + 0.1 * y_range)

            # Add scale bar in bottom-left
            # Inside the same loop, replace the current scale bar block with this:
            if traces:
                all_y = np.concatenate(traces)
                y_min, y_max = np.min(all_y), np.max(all_y)
                y_range = y_max - y_min

                scalebar_x = 30  # in frames
                scalebar_y = 0.05  # ΔF/F

                # Anchor near bottom-left corner (absolute values)
                x_start = 0
                y_start = y_min - .1 * y_range  # very close to bottom

                # Horizontal bar
                ax.add_line(Line2D([x_start, x_start + scalebar_x],
                                [y_start, y_start], color='k', linewidth=1.5))
                # Vertical bar
                ax.add_line(Line2D([x_start, x_start],
                                [y_start, y_start + scalebar_y], color='k', linewidth=1.5))

                # Labels
                ax.text(x_start + scalebar_x / 2, y_start - 0.015 * y_range,
                        f'{scalebar_x/30:.0f} sec', ha='center', va='top', fontsize=10)
                ax.text(x_start - 1, y_start + scalebar_y / 2,
                        f'{scalebar_y:.2f}', ha='right', va='center', fontsize=10)
                


        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)

        return fig, axs

    def plot_avg_informative_cell_overlay(self,aligned_imaging: np.ndarray,
                                  results: dict,  # Additional input for neuron data
                                  cell_id: int,

                                  all_conditions: List[tuple],
                                  condition_colors: List[str],
                                  decoded_variable: str,  # Decoded variable for neuron data
                                  title_base: Optional[str] = None,
                                  peak_info: Optional[float] = None,
                                  frames: Optional[tuple] = None,
                                  subplot_split: Optional[str] = None,
                                  legend: Optional[str] = None,
                                  figsize: Optional[Tuple[float, float]] = None,  # NEW
                                  orientation: str = "horizontal",  # NEW: "horizontal" or "vertical"
                                  smoothing: Optional[float] = None,
                                  shading: Optional[bool] = False,
                                  save_path: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:

        # Set global font size and family 
        plt.rcParams.update({'font.size': 8, 'font.family': 'arial'})

        condition_labels = [label for (_, _, label) in all_conditions]
        group_labels = self.split_condition_labels(condition_labels, subplot_split)
        # print(f"Group labels: {group_labels}")

        n_groups = len(group_labels)
        if orientation == "vertical":
            fig, axs = plt.subplots(1+ n_groups,1, figsize=figsize or (10 * n_groups, 12), squeeze=False)  # 2 subplots (traces + info)
            plt.subplots_adjust(hspace=0.1)
        else:  # horizontal
            fig, axs = plt.subplots(1,1+ n_groups, figsize=figsize or (12 * n_groups, 10), squeeze=False)  # 2 subplots (traces + info)
            plt.subplots_adjust(wspace=0.1)
        axs = axs.flatten()
    

        for ax, (group_name, group_condition_labels) in zip(axs[:n_groups], group_labels.items()):
            # Subplot 1: Traces for all conditions
            for (trials, _, label), color in zip(all_conditions, condition_colors):
                if label not in group_condition_labels:
                    continue

                cell_data = aligned_imaging[trials, cell_id, :]
                if frames is not None:
                    start_frame, end_frame = frames
                    cell_data = cell_data[:, start_frame:end_frame]
                else:
                    start_frame = 0

                mean_trace = np.mean(cell_data, axis=0)
                sem_trace = np.std(cell_data, axis=0) / np.sqrt(len(trials))

                # Optional smoothing
                if smoothing is not None and smoothing > 0:
                    mean_trace = gaussian_filter1d(mean_trace, sigma=smoothing)
                    sem_trace = gaussian_filter1d(sem_trace, sigma=smoothing)

                ax.plot(mean_trace, color=color, linewidth=.8, label=label)

                # Uncomment the following line to add SEM shading
                if shading:
                    ax.fill_between(np.arange(len(mean_trace)),
                                    mean_trace - sem_trace,
                                    mean_trace + sem_trace,
                                    alpha=0.3,
                                    color=color)

            # Add event lines
            if self.event_frames is not None:
                for frame in self.event_frames:
                    ax.axvline(x=frame , color='k', linestyle=(0, (2, 2)), alpha=0.5) 
            ax.set_xticks(self.event_frames)
            ax.set_xticklabels([])

            # clean up appearance
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Title
            # ax.set_title(f"{title_base or 'Cell'} {cell_id}\n{group_name}" +
            #                 (f" | Peak info: {peak_info:.2f}" if peak_info else ""),
            #                 fontsize=10)
            # ax.set_title((f"Info: {peak_info:.2f}" if peak_info else ""), fontsize=10)
            ax.set_ylabel("Act.\n(a.u.)", fontsize=8)

            #make sure  y-limits are the same for all subplots
            if len(axs) > 1:  
                all_y = np.concatenate([ax.get_lines()[0].get_ydata() for ax in axs[:n_groups] if ax.get_lines()])
                y_min, y_max = np.min(all_y), np.max(all_y)
                y_range = y_max - y_min
                for ax in axs[:n_groups]:
                    ax.set_ylim(y_min - .1 * y_range, y_max + 0.1 * y_range)
            

        # Subplot 2: Information in bits (for the selected neuron)
        ax_info = axs[n_groups]  # Now this is a specific subplot

        # Extracting the data for the neuron using the given information
        neuron_data = results[decoded_variable]['sc_instantaneous_information_mean'][:, cell_id]

        ax_info.plot(neuron_data, color='k', linewidth=.8)  # Single trace for the neuron data
        #find peak value and plot yline at that value
        _info_peak_value = np.max(neuron_data)
        ax_info.axhline(y=_info_peak_value, color='m', linestyle='--', label=f'Peak Info: {_info_peak_value:.2f} bits')
        # ax_info.set_xlabel("Time (s)", fontsize=10)
        ax_info.set_ylabel("Info\n(bits)", fontsize=8)

        # # Add event lines
        # for frame, event_label in zip(event_frames, event_labels):
        #     if frames is not None and not (start_frame <= frame < end_frame):
        #         continue
        #     ax_info.axvline(x=frame - start_frame, color='k', linestyle='--', alpha=0.5)

        # Add event lines
        if self.event_frames is not None:
            for frame in self.event_frames:
                ax_info.axvline(x=frame , color='k', linestyle=(0, (2, 2)), alpha=0.5) 
        ax_info.set_xticks(self.event_frames)
        ax_info.set_xticklabels(self.event_labels)
        plt.xticks(rotation=45)   

        # Removing the ticks for the second subplot
        # ax_info.tick_params(left=False, labelleft=False)
        ax_info.spines['top'].set_visible(False)
        ax_info.spines['right'].set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)

        return fig, axs



    def split_condition_labels(self,all_condition_labels, split_by):
        """        Split condition labels into groups based on specified criteria.  
        """
        if split_by is None or split_by.lower() == 'all':
            return {'All': all_condition_labels}
        
        group_labels = {'Group 1': [], 'Group 2': []}

        if split_by is None:
            group_labels = {'All': all_condition_labels}
        elif split_by.lower() == 'right':
            group_labels['Right'] = [label for label in all_condition_labels if label.startswith('Right/')]
            group_labels['Left'] = [label for label in all_condition_labels if label.startswith('Left/')]
        elif split_by.lower() == 'turn':
            group_labels['Right Turn'] = [label for label in all_condition_labels if label.endswith('Right Turn')]
            group_labels['Left Turn'] = [label for label in all_condition_labels if label.endswith('Left Turn')]
        else:
            raise ValueError(f"Unrecognized split_by: {split_by}")

        return {k: v for k, v in group_labels.items() if v}  # remove empty groups

    
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