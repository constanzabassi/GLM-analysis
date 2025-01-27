import numpy as np
import os
import pickle
import scipy
from scipy import stats
import h5py
import random

class DecoderAnalyzer:
    def __init__(self, celltype_data):
        self.cell_types = ['pyr', 'som', 'pv']
        self.metrics = {
            'sc': ['sc_instantaneous_information_mean',
                  'sc_cumulative_information_mean',
                  'sc_instantaneous_fraction_correct_mean',
                  'sc_cumulative_fraction_correct_mean'],
            'pop': ['pop_instantaneous_information_mean',
                   'pop_cumulative_information_mean',
                   'pop_instantaneous_fraction_correct_mean',
                   'pop_cumulative_fraction_correct_mean']
        }
        self.celltype_data = celltype_data

    def analyze_peaks_by_celltype(self, mean_results_all, shuffled_structure,method ='range_threshold', decoder_type='sound_category', start_frame=14, end_frame=None, significance_percentile=95, threshold =None):
        """Analyze peak responses separated by cell type and flag significantly informative neurons.
         Parameters:
        - data: dict, data for different cell types
        - method: str, method to calculate significance ('shuffled_peak', 'threshold_peak', 'range_threshold')
        - threshold: float, threshold value for 'threshold_peak' and 'range_threshold' methods
        - range_start: int, start of the range for 'range_threshold' method
        - range_end: int, end of the range for 'range_threshold' method
        
        Returns:
        - results: dict, results of the analysis
        """
        peaks_by_celltype = {
            dataset: {
                celltype: {
                    'sc': {},
                    'pop': {}
                } for celltype in mean_results_all[dataset]['neuron_groups']
            } for dataset in mean_results_all
        }
        # Single cell metrics
        sc_metrics = self.metrics['sc']
        # Population metrics
        pop_metrics = self.metrics['pop']
        for dataset in mean_results_all:
            celltype_array = mean_results_all[dataset]['celltype_array']
            # Get indices for each cell type
            celltype_indices = {
                'pyr': np.where(celltype_array == 0)[0],
                'som': np.where(celltype_array == 1)[0],
                'pv': np.where(celltype_array == 2)[0]
            }
            # Process single cell metrics
            for metric in sc_metrics:
                if metric in mean_results_all[dataset][decoder_type]:
                    data = mean_results_all[dataset][decoder_type][metric]
                    if end_frame is None:
                        end_frame = len(data)
                    # Retrieve shuffled data for comparison
                    shuffled_data = shuffled_structure [dataset] #mean_results_all[dataset][f'shuffled/{decoder_type}'][metric]
                    for celltype, indices in celltype_indices.items():
                        peaks = []
                        peak_frames = []
                        significant_neurons = []  # Track significant neurons for this metric
                        for idx in indices:
                            neuron_data = data[start_frame:end_frame, idx]
                            peak_val = np.max(neuron_data)
                            peak_frame = np.argmax(neuron_data) + start_frame
                            
                            peaks.append(peak_val)
                            peak_frames.append(peak_frame)
                            # Flag the neuron as significant if the peak value exceeds the 95th percentile
                            if method == 'shuffled_peak':#threshold is None:
                                # Compute the peak value for the shuffled distribution
                                shuffled_peak = shuffled_data[peak_frame, idx, :]
                                # Compute the 95th percentile of the shuffled peak values
                                shuffled_95th_percentile = np.percentile(shuffled_peak, significance_percentile)
                                is_significant = peak_val > shuffled_95th_percentile
                                significant_neurons.append(is_significant)
                            elif method == 'combined':
                                # Calculate threshold from shuffled data
                                shuffled_dist = shuffled_data[start_frame:end_frame, idx, :]
                                threshold = np.percentile(shuffled_dist, significance_percentile, axis=1)
                                
                                # Check if any real data point exceeds its corresponding threshold
                                is_significant = np.any(neuron_data > threshold)
                                significant_neurons.append(is_significant)
                            elif method == 'range_threshold':
                                print('range_threshold')
                                is_significant = np.any(neuron_data > threshold)
                                significant_neurons.append(is_significant)
                            elif method == 'threshold_peak':
                                print('threshold_peak')
                                is_significant = peak_val > threshold
                                significant_neurons.append(is_significant)
                            else:
                                raise ValueError("Invalid method. Choose from 'shuffled_peak', 'threshold_peak', or 'range_threshold'")
                        peaks_by_celltype[dataset][celltype]['sc'][metric] = {
                            'peak_values': np.array(peaks),
                            'peak_frames': np.array(peak_frames),
                            'mean_peak': np.mean(peaks),
                            'sem_peak': np.std(peaks) / np.sqrt(len(peaks)),
                            'significant_neurons': np.array(significant_neurons)  # Store significance for single-cell
                        }
            # Process population metrics
            for metric in pop_metrics:
                if metric in mean_results_all[dataset][decoder_type]:
                    data = mean_results_all[dataset][decoder_type][metric]
                    if end_frame is None:
                        end_frame = len(data)
                    peak_val = np.max(data[start_frame:end_frame])
                    peak_frame = np.argmax(data[start_frame:end_frame]) + start_frame
                    # Retrieve shuffled data for population comparison
                    if method == 'shuffled_peak':#threshold is None:
                        shuffled_data = shuffled_structure [dataset] #mean_results_all[dataset][f'shuffled/{decoder_type}'][metric]
                        shuffled_peak = shuffled_data[peak_frame]
                        # Compute the 95th percentile of the shuffled peak values
                        shuffled_95th_percentile = np.percentile(shuffled_peak, significance_percentile)
                        # Flag the population as significant if the peak value exceeds the 95th percentile
                        is_significant = peak_val > shuffled_95th_percentile
                    elif method == 'range_threshold':
                        is_significant = np.any(data[start_frame:end_frame] > threshold)
                    elif method == 'threshold_peak':
                        is_significant = peak_val > threshold
                    for celltype, indices in celltype_indices.items():
                        peaks_by_celltype[dataset][celltype]['pop'][metric] = {
                            'peak_value': peak_val,
                            'peak_frame': peak_frame,
                            'is_significant': is_significant  # Store significance for population
                        }
        return peaks_by_celltype
    
    def format_peaks_for_cdf(self, peaks_by_celltype, metric='sc_instantaneous_information_mean', significant_only=True):
        """
        Format peaks data for CDF plotting.

        Args:
            peaks_by_celltype (dict): Nested dictionary containing peak information by dataset and cell type.
            metric (str): Metric to extract peak values for. Defaults to 'sc_instantaneous_information_mean'.
            significant_only (bool): Whether to include only significant neurons. Defaults to True.

        Returns:
            dict: Formatted data for CDF plotting by cell type.
            list: Cell labels corresponding to each peak value.
        """
        formatted_data = {
            'pyr': {'peaks': []},
            'som': {'peaks': []},
            'pv': {'peaks': []},
            'all': {'peaks': []}
        }
        all_peaks = []
        cell_labels = []

        # Collect peaks across datasets
        for dataset in peaks_by_celltype:
            for celltype in ['pyr', 'som', 'pv']:
                peak_values = peaks_by_celltype[dataset][celltype]['sc'][metric]['peak_values']
                significant_neurons = peaks_by_celltype[dataset][celltype]['sc'][metric]['significant_neurons']
                
                if significant_only:
                    # Filter peaks for significant neurons
                    peaks = [peak for peak, sig in zip(peak_values, significant_neurons) if sig]
                else:
                    # Include all peaks
                    peaks = peak_values

                # Append peaks to the formatted data
                formatted_data[celltype]['peaks'].extend(peaks)
                all_peaks.extend(peaks)
                cell_labels.extend([celltype] * len(peaks))

        # Include all peaks across cell types
        formatted_data['all']['peaks'] = all_peaks

        return formatted_data, cell_labels

    
    
    def format_peaks_for_boxplot(self,peaks_by_celltype, metric='sc_instantaneous_information_mean', significant_only=True):
        """
        Format peaks for boxplot visualization.

        Args:
            peaks_by_celltype (dict): Nested dictionary containing peak information by dataset and cell type.
            metric (str): Metric to extract peak values for. Defaults to 'sc_instantaneous_information_mean'.
            significant_only (bool): Whether to include only significant neurons. Defaults to True.

        Returns:
            np.ndarray: Array of all peak values.
            dict: Dictionary of neuron groups with indices categorized by cell type.
        """
        # Collect all peaks and indices
        all_peaks = []
        neuron_groups = {'pyr': [], 'som': [], 'pv': []}
        current_idx = 0

        # Collect peaks across datasets
        for dataset in peaks_by_celltype:
            for celltype in ['pyr', 'som', 'pv']:
                # Get peak values and significance flags
                peak_values = peaks_by_celltype[dataset][celltype]['sc'][metric]['peak_values']
                if significant_only:
                    significant_flags = peaks_by_celltype[dataset][celltype]['sc'][metric]['significant_neurons']
                    # Filter only significant peak values
                    peak_values = peak_values[significant_flags]

                # Append peak values to the main list
                all_peaks.extend(peak_values)

                # Store indices for this cell type
                indices = list(range(current_idx, current_idx + len(peak_values)))
                neuron_groups[celltype].extend(indices)
                current_idx += len(peak_values)

        return np.array(all_peaks), neuron_groups
    
    def analyze_significant_neurons(self, results_dict,shuffled_structure,method, decoder_type, start_frame, end_frame, metric = 'sc_instantaneous_information_mean',significance_percentile=95, threshold=None):   
        """Analyze significant neurons for plotting."""
        neuron_ids_by_dataset = {}
        significance_struc = {}
        peaks_by_celltype = self.analyze_peaks_by_celltype(results_dict,shuffled_structure,method, decoder_type, start_frame, end_frame, significance_percentile,threshold)

        for dataset in results_dict:
            neuron_ids_by_dataset[dataset] = {}
            significance_struc[dataset] = {}
            
            for celltype in peaks_by_celltype[dataset]:
                neuron_ids_by_dataset[dataset][celltype] = []

                # Ensure initialization of significance_struc[dataset][celltype]
                significance_struc[dataset][celltype] = {}
                # Extract significant neurons
                significant_neurons = peaks_by_celltype[dataset][celltype]['sc'][metric]['significant_neurons']
                neuron_ids_by_dataset[dataset][celltype] = np.where(significant_neurons)[0].tolist()
                # Add peak data for significant neurons
                significant_indices = neuron_ids_by_dataset[dataset][celltype]
                significance_struc[dataset][celltype]['peak_values'] = peaks_by_celltype[dataset][celltype]['sc'][metric]['peak_values'][significant_indices]
                significance_struc[dataset][celltype]['peak_frames'] = peaks_by_celltype[dataset][celltype]['sc'][metric]['peak_frames'][significant_indices]
            
        return neuron_ids_by_dataset, significance_struc
    
    def analyze_significant_neurons_by_threshold(self, results_dict, decoder_type, start_frame, end_frame, metric='sc_instantaneous_information_mean', threshold=0.5):
        """Analyze significant neurons by threshold, looping through cell types defined in self.cell_types."""
        neuron_ids_by_dataset = {}
        significance_struc = {}

        for dataset in results_dict:
            neuron_ids_by_dataset[dataset] = {}
            significance_struc[dataset] = {}

            # Get the indices for neurons of this cell type
            celltype_array = self.celltype_data[dataset]['celltype_array']
            # Get indices for each cell type
            celltype_indices = {
                'pyr': np.where(celltype_array == 0)[0],
                'som': np.where(celltype_array == 1)[0],
                'pv': np.where(celltype_array == 2)[0]
            }

            data = results_dict[dataset][decoder_type][metric]

            if end_frame is None:
                end_frame = data.shape[0]

            # Extract the data within the specified frames
            data_in_range = data[start_frame:end_frame, :]

            # Loop through cell types defined in self.cell_types
            for celltype, indices in celltype_indices.items(): #for celltype in self.cell_types:
                neuron_ids_by_dataset[dataset][celltype] = []
                significance_struc[dataset][celltype] = {}

                
                celltype_indices = indices  # Assuming this is a list/array of indices

                if len(celltype_indices) == 0:
                    continue  # Skip if no neurons of this cell type

                # Subset the data for the current cell type
                data_celltype = data_in_range[:, celltype_indices]

                # Find neurons that exceed the threshold at any point in the range
                significant_neurons = np.any(data_celltype > threshold, axis=0)
                neuron_ids = np.array(celltype_indices)[np.where(significant_neurons)[0]].tolist()
                neuron_ids_by_dataset[dataset][celltype] = neuron_ids

                # Collect additional information for the significant neurons
                significance_struc[dataset][celltype]['peak_values'] = np.max(data_celltype[:, significant_neurons], axis=0)
                significance_struc[dataset][celltype]['peak_frames'] = np.argmax(data_celltype[:, significant_neurons], axis=0) + start_frame

        return neuron_ids_by_dataset, significance_struc

    
    # def analyze_significant_neurons_by_threshold(self, results_dict, decoder_type, start_frame, end_frame, metric='sc_instantaneous_information_mean', threshold=0.5):
    #     """Analyze significant neurons that exceed a given threshold value for plotting."""
    #     neuron_ids_by_dataset = {}
    #     significance_struc = {}

    #     for dataset in results_dict:
    #         neuron_ids_by_dataset[dataset] = {}
    #         significance_struc[dataset] = {}
    #         data = results_dict[dataset][decoder_type][metric]

    #         if end_frame is None:
    #             end_frame = data.shape[0]

    #         # Extract the data within the specified frames
    #         data_in_range = data[start_frame:end_frame, :]

    #         for celltype in ['all']:  # Modify this if there are specific cell types
    #             neuron_ids_by_dataset[dataset][celltype] = []
    #             significance_struc[dataset][celltype] = {}

    #             # Find neurons that exceed the threshold at any point in the range
    #             significant_neurons = np.any(data_in_range > threshold, axis=0)
    #             neuron_ids = np.where(significant_neurons)[0].tolist()
    #             neuron_ids_by_dataset[dataset][celltype] = neuron_ids

    #             # Collect additional information for the significant neurons
    #             significance_struc[dataset][celltype]['peak_values'] = np.max(data_in_range[:, neuron_ids], axis=0)
    #             significance_struc[dataset][celltype]['peak_frames'] = np.argmax(data_in_range[:, neuron_ids], axis=0) + start_frame

    #     return neuron_ids_by_dataset, significance_struc

    
    # def plot_single_neuron_analysis(results_dict, decoder_type='sound_category', start_frame=14, end_frame=None):
    # """Comprehensive single neuron decoding visualization"""

    # # 1. Neuron Performance Heatmap
    # plt.figure(figsize=(12, 8))
    # for dataset in results_dict:
    #     data = results_dict[dataset][decoder_type]['sc_cumulative_information_mean']
    #     celltype_array = results_dict[dataset]['celltype_array']

    #     # Sort neurons by cell type and performance
    #     max_info = np.max(data[start_frame:, :], axis=0)
    #     sort_idx = np.argsort(max_info)

    #     plt.subplot(len(results_dict), 1, list(results_dict.keys()).index(dataset) + 1)
    #     sns.heatmap(data[:, sort_idx].T,
    #                 cmap='viridis',
    #                 xticklabels=20,
    #                 yticklabels=False)
    #     plt.title(f'{dataset} Single Neuron Decoding')
    # plt.tight_layout()

    # # 2. Best Neurons Analysis
    # neuron_ids_by_dataset = {}
    # fig, axes = plt.subplots(1, 2, figsize=(6, 3))

    # for cel_index, (celltype, color) in enumerate(plotter.celltypecolors.items()):
    #     all_peaks = []
    #     all_peaks_locs = []

    #     for dataset in results_dict:
    #         if dataset not in neuron_ids_by_dataset:
    #             neuron_ids_by_dataset[dataset] = {}
    #         if celltype not in neuron_ids_by_dataset[dataset]:
    #             neuron_ids_by_dataset[dataset][celltype] = []

    #         peaks_by_celltype = analyze_peaks_by_celltype(results_dict, decoder_type=decoder_type, start_frame=start_frame, end_frame=end_frame)
    #         peaks = peaks_by_celltype[dataset][celltype]['sc']['sc_instantaneous_information_mean']['peak_values']
    #         peaks_locs = peaks_by_celltype[dataset][celltype]['sc']['sc_instantaneous_information_mean']['peak_frames']
    #         significant_neurons = peaks_by_celltype[dataset][celltype]['sc']['sc_instantaneous_information_mean']['significant_neurons']

    #         if len(peaks) > 0:
    #             # Filter for significant neurons
    #             significant_peaks = [peak for peak, sig in zip(peaks, significant_neurons) if sig]
    #             significant_peaks_locs = [loc for loc, sig in zip(peaks_locs, significant_neurons) if sig]

    #             all_peaks.extend(significant_peaks)
    #             all_peaks_locs.extend(significant_peaks_locs)

    #             neuron_ids_by_dataset[dataset][celltype].extend(np.where(significant_neurons)[0].tolist())

    #     axes[0].hist(all_peaks, alpha=1.0, color=color, label=celltype, histtype='step', linewidth=2)
    #     axes[0].set_xlabel('Information (bits)')
    #     axes[0].spines['top'].set_visible(False)
    #     axes[0].spines['right'].set_visible(False)
    #     axes[0].set_box_aspect(1)

    #     axes[1].hist(all_peaks_locs, alpha=1.0, color=color, label=celltype, histtype='step', linewidth=2)
    #     axes[1].set_xlabel('Peak Frame')
    #     axes[1].spines['top'].set_visible(False)
    #     axes[1].spines['right'].set_visible(False)
    #     axes[1].set_box_aspect(1)

    # fig.suptitle('Significant Neurons Distribution by Cell Type')
    # plt.tight_layout()
    # plt.show()

    # # 3. Time Course by Cell Type
    # plt.figure(figsize=(3, 3))
    # for cel_index, (celltype, color) in enumerate(plotter.celltypecolors.items()):
    #     all_traces = []
    #     for dataset in results_dict:
    #         if end_frame is None:
    #             end_frame = len(data)

    #         traces = results_dict[dataset][decoder_type]['sc_instantaneous_information_mean'][0:end_frame, :]
    #         celltype_idx = results_dict[dataset]['celltype_array'] == cel_index

    #         if np.any(celltype_idx):
    #             mean_trace = np.mean(traces[:, celltype_idx], axis=1)
    #             all_traces.append(mean_trace)

    #     mean = np.mean(all_traces, axis=0)
    #     sem = np.std(all_traces, axis=0) / np.sqrt(len(all_traces))
    #     plt.plot(mean, color=color, label=celltype)
    #     ax = plt.gca()
    #     ax.axvline(x=start_frame, color='k', linestyle=':', alpha=0.5)
    #     plt.fill_between(range(len(mean)), mean - sem, mean + sem, alpha=0.2, color=color)
    #     ax.spines['top'].set_visible(False)
    #     ax.spines['right'].set_visible(False)
    #     ax.set_box_aspect(1)

    # plt.legend()
    # plt.title('Average Information Time Course by Cell Type')
    # plt.xlabel('Time (frames)')
    # plt.ylabel('Information (bits)')
    # plt.show()

    # return neuron_ids_by_dataset


    