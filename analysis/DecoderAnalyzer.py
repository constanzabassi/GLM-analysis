import numpy as np
import os
import pickle
import scipy
from scipy import stats
import h5py
import random
import re

#import stats class
from utils.general_stats import GeneralStats
from utils.Plotter import Plotter as plotter

class DecoderAnalyzer:
    def __init__(self, celltype_data, random_seed=412):
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
        self.celltype_info = celltype_data

        # Set random seed for numpy
        np.random.seed(random_seed)
        # Set random seed for Python's random module
        random.seed(random_seed)
        self.stats = GeneralStats()  # Instantiate GeneralStats

    def analyze_peaks_by_celltype(self, mean_results_all, shuffled_structure,method ='range_threshold', decoder_type='sound_category', start_frame=14, end_frame=None, significance_percentile=95, threshold =None, window = 2):
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
                        global_indices = []  # Add list for global indices

                        for idx in indices:
                            neuron_data = data[start_frame:end_frame, idx]
                            peak_val = np.max(neuron_data)
                            peak_frame = np.argmax(neuron_data) + start_frame
                            
                            peaks.append(peak_val)
                            peak_frames.append(peak_frame)
                            global_indices.append(idx)  # Store global index

                            # Flag the neuron as significant if the peak value exceeds the 95th percentile
                            if method == 'shuffled_peak':#threshold is None:
                                # Compute the peak value for the shuffled distribution
                                shuffled_peak = shuffled_data[peak_frame, idx, :]
                                # Compute the 95th percentile of the shuffled peak values
                                shuffled_95th_percentile = np.percentile(shuffled_peak, significance_percentile)
                                is_significant = peak_val > shuffled_95th_percentile
                                significant_neurons.append(is_significant)
                            elif method == 'shuffled_timepoint':
                                # For each timepoint, compare to shuffled 95th percentile
                                shuff = shuffled_data[start_frame:end_frame, idx, :]
                                shuff_95 = np.percentile(shuff, significance_percentile, axis=1)
                                sig_mask = neuron_data > shuff_95
                                # Set non-significant info to zero
                                neuron_data_sig = neuron_data * sig_mask
                                peak_val = np.max(neuron_data_sig)
                                peak_frame = np.argmax(neuron_data_sig) + start_frame
                                is_significant = np.any(sig_mask)
                                significant_neurons.append(is_significant)
                            elif method == 'shuffled_peak_window':
                                # window = 1  # or 2 for wider window
                                frame_range = np.arange(max(start_frame, peak_frame - window),
                                                        min(end_frame, peak_frame + window + 1))
                                shuffled_peaks = shuffled_data[frame_range, idx, :].max(axis=0)  # max across window
                                # Compute the 95th percentile of the shuffled peak values
                                shuffled_95th_percentile = np.percentile(shuffled_peaks, significance_percentile)
                                is_significant = peak_val > shuffled_95th_percentile
                                significant_neurons.append(is_significant)
                            elif method == 'shuffled_peak_zscore':
                                # window = 1  # number of frames on each side of peak to include
                                significance_zscore = 2.0  # you can tweak this cutoff

                                # Get a time window around the peak frame
                                frame_start = max(start_frame, peak_frame - window)
                                frame_end = min(end_frame, peak_frame + window + 1)
                                frame_range = np.arange(frame_start, frame_end)

                                # Get the max value from the shuffled distribution within this window
                                shuffled_peaks = shuffled_data[frame_range, idx, :].max(axis=0)

                                # Compute z-score of the real peak against the shuffled distribution
                                shuff_mean = np.mean(shuffled_peaks)
                                shuff_std = np.std(shuffled_peaks)

                                # Handle case where std is 0 (flat shuffle)
                                if shuff_std == 0:
                                    z_score = np.inf if peak_val > shuff_mean else -np.inf
                                else:
                                    z_score = (peak_val - shuff_mean) / shuff_std

                                is_significant = z_score > significance_zscore
                                significant_neurons.append(is_significant)

                            elif method == 'combined':
                                # Calculate threshold from shuffled data
                                shuffled_dist = shuffled_data[start_frame:end_frame, idx, :]
                                threshold = np.percentile(shuffled_dist, significance_percentile) #, axis=1
                                
                                # Check if any real data point exceeds its corresponding threshold
                                is_significant = np.any(neuron_data > threshold)
                                significant_neurons.append(is_significant)
                            elif method == 'combined_thr':
                                # Calculate threshold from shuffled data
                                shuffled_dist = shuffled_data[start_frame:end_frame, idx, :]
                                shuff_threshold = np.percentile(shuffled_dist, significance_percentile) #, axis=1
                                
                                # Check if any real data point exceeds its corresponding threshold and some external threshold
                                # Use parentheses and combine conditions properly
                                condition1 = neuron_data > shuff_threshold
                                condition2 = neuron_data > threshold
                                is_significant = np.any(condition1 & condition2)
                                significant_neurons.append(is_significant)
                            elif method == 'range_threshold':
                                # print('range_threshold')
                                is_significant = np.any(neuron_data > threshold)
                                significant_neurons.append(is_significant)
                            elif method == 'threshold_peak':
                                # print('threshold_peak')
                                is_significant = peak_val > threshold
                                significant_neurons.append(is_significant)
                            else:
                                raise ValueError("Invalid method. Choose from 'shuffled_peak', 'threshold_peak', or 'range_threshold'")
                        peaks_by_celltype[dataset][celltype]['sc'][metric] = {
                            'peak_values': np.array(peaks),
                            'peak_frames': np.array(peak_frames),
                            'mean_peak': np.mean(peaks),
                            'sem_peak': np.std(peaks) / np.sqrt(len(peaks)),
                            'significant_neurons': np.array(significant_neurons),  # Store significance for single-cell
                            'global_indices': np.array(global_indices)  # Add global indices to output
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

    def universal_shuffled_threshold(self,shuffled_structure, start_frame, end_frame, significance_percentile=95,mode='peak'):
        """
        Determines significant neurons based on a universal threshold computed from all shuffled peaks.

        Parameters:
        - data: 2D array (frames x neurons), real info values
        - shuffled_data: 3D array (frames x neurons x shuffles), shuffled info values
        - start_frame: int, analysis start frame
        - end_frame: int, analysis end frame
        - significance_percentile: float, percentile for the universal threshold (default 95)
        - mode: str, either 'peak' or 'mean' to choose the summary metric from shuffled data

        Returns:
        - significant_neurons: list of bool, True if neuron is significant
        - universal_threshold: float, the computed threshold from shuffled data
        """
        # Store thresholds for each dataset
        dataset_thresholds = {}
        
        # Calculate threshold for each dataset
        for dataset in shuffled_structure:
            # Get shuffled data for this dataset
            shuffled_data = shuffled_structure[dataset]

            if mode == 'peak':
                # Use max (peak) across frames
                summary_values = np.max(shuffled_data[start_frame:end_frame, :, :], axis=0)  # shape: (neurons x shuffles)
            elif mode == 'mean':
                # Use mean across frames
                summary_values = np.mean(shuffled_data[start_frame:end_frame, :, :], axis=0)  # shape: (neurons x shuffles)
            else:
                raise ValueError("Mode must be 'peak' or 'mean'.")

            summary_flat = summary_values.flatten()
            dataset_thresholds[dataset] = np.percentile(summary_flat, significance_percentile)
        
        # Calculate universal threshold as mean across datasets
        universal_threshold = np.mean(list(dataset_thresholds.values()))
        std_threshold = np.std(list(dataset_thresholds.values()))
        
        return {
            'thresholds': dataset_thresholds,
            'universal_threshold': universal_threshold,
            'std_threshold': std_threshold
        }

    
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
        """
        Analyze significant neurons for plotting.
        
        Returns:
        --------
        neuron_ids_by_dataset : dict
            Dictionary of significant neuron indices by dataset and celltype
        significance_struc : dict
            Dictionary containing peak values, frames, and indices for significant neurons
        all_significant_indices : list
            List of global indices for all significant neurons across all celltypes
        """
        neuron_ids_by_dataset = {}
        significance_struc = {}
        significance_all = {}
        all_significant_indices = []  # New list to store all significant indices

        # Analyze peaks by cell type
        peaks_by_celltype = self.analyze_peaks_by_celltype(results_dict,shuffled_structure,method, decoder_type, start_frame, end_frame, significance_percentile,threshold)

        for dataset in results_dict:
            neuron_ids_by_dataset[dataset] = {}
            significance_struc[dataset] = {}
            dataset_significant_indices = []  # Track significant indices per dataset
            
            for celltype in peaks_by_celltype[dataset]:
                neuron_ids_by_dataset[dataset][celltype] = []

                # Ensure initialization of significance_struc[dataset][celltype]
                significance_struc[dataset][celltype] = {}
                # Extract significant neurons
                significant_neurons = peaks_by_celltype[dataset][celltype]['sc'][metric]['significant_neurons']
                neuron_ids_by_dataset[dataset][celltype] = np.where(significant_neurons)[0].tolist()
                global_indices = peaks_by_celltype[dataset][celltype]['sc'][metric]['global_indices']

                # Store global indices of significant neurons
                significant_indices = np.where(significant_neurons)[0]
                significant_global_indices = global_indices[significant_indices]
                dataset_significant_indices.extend(significant_global_indices)  # Accumulate only for this dataset

                # Add peak data for significant neurons
                significant_indices = neuron_ids_by_dataset[dataset][celltype]
                significance_struc[dataset][celltype]['peak_values'] = peaks_by_celltype[dataset][celltype]['sc'][metric]['peak_values'][significant_indices]
                significance_struc[dataset][celltype]['peak_frames'] = peaks_by_celltype[dataset][celltype]['sc'][metric]['peak_frames'][significant_indices]
                significance_struc[dataset][celltype]['neuron_indices'] = significant_global_indices

            # Add combined significant neurons for the dataset
            significance_struc[dataset]['sig_neurons_all'] = np.sort(np.array(dataset_significant_indices))
            significance_all[dataset] = np.sort(np.array(dataset_significant_indices))
            
        return neuron_ids_by_dataset, significance_struc, significance_all
    
    def analyze_significant_neurons_by_threshold(self, results_dict, decoder_type, start_frame, end_frame, metric='sc_instantaneous_information_mean', threshold=0.5):
        """Analyze significant neurons by threshold, looping through cell types defined in self.cell_types."""
        neuron_ids_by_dataset = {}
        significance_struc = {}

        for dataset in results_dict:
            neuron_ids_by_dataset[dataset] = {}
            significance_struc[dataset] = {}

            # Get the indices for neurons of this cell type
            celltype_array = self.celltype_info[dataset]['celltype_array']
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


    def wrapper_info_plots_analysis(self, 
                                       results_dict,
                                       shuffled_structure,
                                       plotter,
                                       decoder_type='sound_category',
                                       start_frame=14,
                                       end_frame=None,
                                       metric='sc_instantaneous_information_mean',
                                       significance_percentile=95,
                                       threshold=None,
                                       method='shuffled_peaks',
                                       save_path=None):
        """
        Orchestrate single neuron analysis and visualization.
        
        Args:
            results_dict (dict): Dictionary containing decoder results
            shuffled_structure (dict): Dictionary containing shuffled data
            plotter (Plotter): Plotter instance for visualization
            decoder_type (str): Type of decoder analysis
            start_frame (int): Starting frame for analysis
            end_frame (int, optional): Ending frame for analysis
            metric (str): Metric to analyze
            significance_percentile (float): Percentile for significance
            threshold (float, optional): Threshold value
            method (str): Method for analysis ('shuffled_peaks', 'combined', etc.)
            save_path (str, optional): Base path for saving plots
        
        Returns:
            tuple: (significant_neurons_data, significance_struc, significant_neurons)
        """
        # Analyze significant neurons
        significant_neurons_data, significance_struc, significant_neurons = self.analyze_significant_neurons(
            results_dict,
            shuffled_structure,
            method,
            decoder_type,
            start_frame,
            end_frame,
            metric,
            significance_percentile,
            threshold=threshold
        )
        
        # Generate plots
        # plotter.plot_significant_neurons_distribution(
        #     significance_struc,
        #     save_path=f'{save_path}_hist.pdf'
        # )
        
        plotter.plot_time_course_by_cell_type(
            results_dict,
            decoder_type,
            start_frame=start_frame,
            end_frame=end_frame,
            metric=metric,
            significance_struc=significance_struc
        )
        
        plotter.plot_summary_heatmap(
            results_dict,
            decoder_type,
            start_frame,
            end_frame,
            metric,
            significance_struc,
            save_path=f'{save_path}_heatmap.pdf'
        )
        
        # plotter.plot_significant_neuron_percentages_by_celltype(
        #     significance_struc,
        #     self.celltype_info,
        #     save_path=f'{save_path}.pdf'
        # )
        
        # Print debug information
        print("Structure of significance_struc:")
        for dataset in significance_struc:
            for celltype in significance_struc[dataset]:
                if celltype != 'sig_neurons_all':
                    print(f"Dataset: {dataset}, Celltype: {celltype}, "
                          f"Neurons: {len(significance_struc[dataset][celltype]['neuron_indices'])}")
            print(f"Dataset: {dataset}, All sig neurons: "
                  f"{len(significance_struc[dataset]['sig_neurons_all'])}")
        
        print("\nStructure of significant_neurons:")
        if isinstance(significant_neurons, np.ndarray):
            print(f"Total significant neurons: {len(significant_neurons)}")
        else:
            print(f"Type of significant_neurons: {type(significant_neurons)}")
        
        return significant_neurons_data, significance_struc, significant_neurons

    def sort_peaks_by_information(self, peaks_by_celltype, metric='sc_instantaneous_information_mean'):
        """
        Sorts neurons by peak information value for all datasets and cell types.
        
        Returns a nested dict with sorted 'global_indices' and 'peak_values' arrays.

        ids_only is a dict with only global indices, useful for MATLAB compatibility.
        """
        sorted_peaks = {}
        sorted_peaks_ids_only = {}
        for dataset in peaks_by_celltype:
            sorted_peaks[dataset] = {}
            sorted_peaks_ids_only[dataset] = {}
            for celltype in peaks_by_celltype[dataset]:
                info_dict = peaks_by_celltype[dataset][celltype]['sc'][metric]
                peak_values = np.array(info_dict['peak_values'])
                global_indices = np.array(info_dict['global_indices'])
                # Sort descending by peak_values
                sorted_idx = np.argsort(peak_values)[::-1]
                sorted_peaks[dataset][celltype] = {
                    'global_indices': global_indices[sorted_idx],
                    'peak_values': peak_values[sorted_idx]
                }
                # Store only global indices
                sorted_peaks_ids_only[dataset][celltype] = np.array(global_indices[sorted_idx])
        return sorted_peaks_ids_only, sorted_peaks

    def flatten_for_matlab(self,sorted_peaks_ids_only):
        """
        Convert nested dict to a flat dict with tuple keys for MATLAB compatibility
        """
        mat_dict = {}
        for dataset in sorted_peaks_ids_only:
            # Replace invalid characters in dataset name
            safe_dataset = re.sub(r'\W|^(?=\d)', '_', dataset)
            for celltype in sorted_peaks_ids_only[dataset]:
                key = f"{safe_dataset}_{celltype}"
                mat_dict[key] = np.array(sorted_peaks_ids_only[dataset][celltype])
        return mat_dict
    
    def find_special_neurons(self, dataset, sig_neurons, mod_values, peaks_by_celltype,
                         mod_threshold=0.1, info_threshold=0.06):
        """Find neurons in different categories based on modulation and information."""

        special_neurons = {
            'most_modulated': {'neuron_id': None, 'mod_val': 0, 'info_val': 0, 'cell_type': None},
            'most_informative': {'neuron_id': None, 'mod_val': 0, 'info_val': 0, 'cell_type': None},
            'mod_not_info': {'neuron_id': None, 'mod_val': 0, 'info_val': 0, 'cell_type': None},
            'info_not_mod': {'neuron_id': None, 'mod_val': 0, 'info_val': 0, 'cell_type': None}
        }

        # Build info dictionary
        info_dict = {}
        for cell_type in self.cell_types:
            if cell_type in peaks_by_celltype.get(dataset, {}):
                sc_data = peaks_by_celltype[dataset][cell_type]['sc']['sc_instantaneous_information_mean']
                global_indices = sc_data['global_indices']
                peak_values = sc_data['peak_values']
                for idx, neuron_id in enumerate(global_indices):
                    info_dict[neuron_id] = {'info': peak_values[idx], 'cell_type': cell_type}

        # Handle sig_neurons
        if sig_neurons is None:
            sig_neurons = np.array(list(info_dict.keys())) if info_dict else np.array([])
        else:
            sig_neurons = np.array(sig_neurons)

        # Handle modulation values
        if mod_values is not None and len(mod_values) == len(sig_neurons):
            all_mod_values = np.abs(mod_values)
            has_modulation = True
        else:
            all_mod_values = np.zeros(len(sig_neurons))
            has_modulation = False

        # Most modulated neuron
        if has_modulation and len(all_mod_values) > 0:
            max_mod_idx = np.argmax(all_mod_values)
            max_mod_neuron = sig_neurons[max_mod_idx]
            special_neurons['most_modulated'].update({
                'neuron_id': max_mod_neuron,
                'mod_val': all_mod_values[max_mod_idx],
                'info_val': info_dict.get(max_mod_neuron, {'info': 0})['info'],
                'cell_type': info_dict.get(max_mod_neuron, {'cell_type': 'unknown'})['cell_type']
            })

        # Most informative neuron
        max_info = 0
        max_info_neuron = None
        max_info_cell_type = None
        for neuron_id, info in info_dict.items():
            if info['info'] > max_info:
                max_info = info['info']
                max_info_neuron = neuron_id
                max_info_cell_type = info['cell_type']

        if max_info_neuron is not None:
            mod_val = 0
            if has_modulation and max_info_neuron in sig_neurons:
                mod_idx = np.where(sig_neurons == max_info_neuron)[0]
                if len(mod_idx) > 0:
                    mod_val = all_mod_values[mod_idx[0]]
            special_neurons['most_informative'].update({
                'neuron_id': max_info_neuron,
                'mod_val': mod_val,
                'info_val': max_info,
                'cell_type': max_info_cell_type
            })

        # Modulated but not informative
        if has_modulation:
            for idx, neuron_id in enumerate(sig_neurons):
                mod_val = all_mod_values[idx]
                info_val = info_dict.get(neuron_id, {'info': 0})['info']
                if mod_val > mod_threshold and info_val < info_threshold:
                    if mod_val > special_neurons['mod_not_info']['mod_val']:
                        special_neurons['mod_not_info'].update({
                            'neuron_id': neuron_id,
                            'mod_val': mod_val,
                            'info_val': info_val,
                            'cell_type': info_dict.get(neuron_id, {'cell_type': 'unknown'})['cell_type']
                        })

        # Informative but not modulated
        for neuron_id, info in info_dict.items():
            if sig_neurons.size == 0 or neuron_id not in sig_neurons:
                continue
            mod_val = 0
            if has_modulation:
                mod_idx = np.where(sig_neurons == neuron_id)[0]
                if len(mod_idx) > 0:
                    mod_val = all_mod_values[mod_idx[0]]
            if info['info'] > info_threshold and mod_val < mod_threshold:
                if info['info'] > special_neurons['info_not_mod']['info_val']:
                    special_neurons['info_not_mod'].update({
                        'neuron_id': neuron_id,
                        'mod_val': mod_val,
                        'info_val': info['info'],
                        'cell_type': info['cell_type']
                    })

        return special_neurons
    
    
    
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


    