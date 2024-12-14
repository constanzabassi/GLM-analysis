import numpy as np
import os
import pickle
import scipy
from scipy import stats
import h5py
import random

class DecoderAnalyzer:
    def __init__(self):
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

    def analyze_peaks_by_celltype(self, mean_results_all, shuffled_structure, decoder_type='sound_category', start_frame=14, end_frame=None, significance_percentile=95):
        """Analyze peak responses separated by cell type and flag significantly informative neurons."""
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
                            
                            # Compute the peak value for the shuffled distribution
                            shuffled_peak = shuffled_data[peak_frame, idx, :]
                            # Compute the 95th percentile of the shuffled peak values
                            shuffled_95th_percentile = np.percentile(shuffled_peak, significance_percentile)
                            # Flag the neuron as significant if the peak value exceeds the 95th percentile
                            is_significant = peak_val > shuffled_95th_percentile
                            peaks.append(peak_val)
                            peak_frames.append(peak_frame)
                            significant_neurons.append(is_significant)
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
                    peak_val = np.max(data[start_frame:end_frame])
                    peak_frame = np.argmax(data[start_frame:end_frame]) + start_frame
                    # Retrieve shuffled data for population comparison
                    shuffled_data = shuffled_structure [dataset] #mean_results_all[dataset][f'shuffled/{decoder_type}'][metric]
                    shuffled_peak = shuffled_data[peak_frame]
                    # Compute the 95th percentile of the shuffled peak values
                    shuffled_95th_percentile = np.percentile(shuffled_peak, significance_percentile)
                    # Flag the population as significant if the peak value exceeds the 95th percentile
                    is_significant = peak_val > shuffled_95th_percentile
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

    