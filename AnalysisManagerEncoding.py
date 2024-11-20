import numpy as np
import os
import pickle
import scipy
from scipy import stats
import h5py
import random

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from scipy.stats import wilcoxon
import itertools

#IMPORT PLOTTING FUNCTIONS!
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
from .Plotter import Plotter as plotter

class AnalysisManagerEncoding:
    def __init__(self, data, plotter):
        self.data = data
        self.plotter = plotter  # Store the plotter module

    def calculate_bonferroni_significance(self,all_p_values, alpha=0.05):
        """
        Calculate Bonferroni corrected significance stars based on p-values.
        """
        num_tests = len(all_p_values)
        bonferroni_threshold = alpha / num_tests
        print(f"Bonferroni corrected alpha threshold: {bonferroni_threshold:.5f}")

        corrected_p_values = [p * num_tests for p in all_p_values]
        significance_stars = []

        for corrected_p in corrected_p_values:
            if corrected_p < 0.001:
                significance_stars.append('***')
            elif corrected_p < 0.01:
                significance_stars.append('**')
            elif corrected_p < 0.05:
                significance_stars.append('*')
            else:
                significance_stars.append('ns')  # Not significant
        
        return corrected_p_values, significance_stars

    def generate_scatter_plots(self, results, model_type, comparisons = [
            ('No Coupling', 'All'),
            ('No Coupling', 'No Pyr'),
            ('No Coupling', 'No Som'),
            ('No Coupling', 'No Pv'),
            ('No Pyr', 'All'),
            ('No Som', 'All'),
            ('No Pv', 'All')
        ],
        significant_neurons=None, lims = 0.6):

        """
        Generate scatter plots comparing different deviance explained models.

        Parameters:
        - results: dictionary of model results
        - celltypecolors: color mappings for cell types
        - save_results: directory to save the results
        - model_type: the type of model being analyzed
        - plotter: the plotting module containing scatter_model_dev_comparison
        - significant_neurons: optional list of significant neurons
        - lims: optional limits for the scatter plot axes
        """

        # Initialize an empty dictionary to store the mean deviance explained values across all datasets
        mean_dev_dict = {
            'No Coupling': [],
            'All': [],
            'No Pyr': [],
            'No Som': [],
            'No Pv': []
        }
        
        # Initialize a list to store all cell labels
        cell_labels = []
        
        # Loop over each dataset in the results dictionary
        for dataset_key, dataset in results.items():
            # Determine the cell labels based on significant neurons if provided
            if significant_neurons and dataset_key in significant_neurons:
                cell_ids = significant_neurons[dataset_key][0]
            else:
                cell_ids = range(len(dataset['celltype_array']))  # Default to all cells

            # Append the data from each dataset to the corresponding list in mean_dev_dict
            mean_dev_dict['No Coupling'].extend(dataset['mean_dev_behav'][cell_ids])
            mean_dev_dict['All'].extend(dataset['mean_dev'][cell_ids])
            mean_dev_dict['No Pyr'].extend(dataset['mean_dev_no_pyr'][cell_ids])
            mean_dev_dict['No Som'].extend(dataset['mean_dev_no_som'][cell_ids])
            mean_dev_dict['No Pv'].extend(dataset['mean_dev_no_pv'][cell_ids])
            
            # Map cell IDs to cell types and add to the cell_labels list
            cell_types = {
                0: 'pyr',
                1: 'som',
                2: 'pv',
            }
            cell_labels.extend([cell_types[dataset['celltype_array'][i]] for i in cell_ids])
        
        # Change directory to the results folder
        os.chdir(self.plotter.save_results)
        
        # List of comparisons
        comparisons = comparisons
        
        # Create scatter plots for each comparison
        for (label1, label2) in comparisons:
            full_data = mean_dev_dict[label2]
            partial_data = mean_dev_dict[label1]
            
            # Construct the save string to reflect the comparison being made
            if significant_neurons is not None:
                save_string = f'scatter_sigcells_{label1}_vs_{label2}_{model_type}.png'
            else:
                save_string = f'scatter_{label1}_vs_{label2}_{model_type}.png'
            
            self.plotter.scatter_model_dev_comparison(full_data, partial_data, cell_labels, label1, label2, colors = 1, plot_lims= lims, save_path = save_string)



    def calculate_coupling_index(self,mean_deviance_coupling, mean_deviance_uncoupled):
        """
        Calculate the coupling index.

        Parameters:
            mean_deviance_coupling (array_like): Deviance explained for the model with coupling predictors.
            mean_deviance_uncoupled (array_like): Deviance explained for the model without coupling predictors.

        Returns:
            np.ndarray: Coupling index for each observation.
        """
        # Ensure inputs are numpy arrays
        mean_deviance_coupling = np.array(mean_deviance_coupling)
        mean_deviance_uncoupled = np.array(mean_deviance_uncoupled)

        # Calculate the coupling index
        coupling_index = (mean_deviance_coupling - mean_deviance_uncoupled) / mean_deviance_coupling

        # Identify and handle cases with large coupling indices due to small or negative deviance values
        #extreme_indices = np.where((coupling_index > 1) & ((np.abs(mean_deviance_coupling) < threshold) | (np.abs(mean_deviance_uncoupled) < threshold)))
        #extreme_indices = np.where(mean_deviance_coupling < threshold)
        # extreme_indices = np.where(coupling_index > 1)
        # coupling_index[extreme_indices] = np.nan

        return coupling_index
    
    def paired_permutation_test(self,data1, data2, num_permutations=10000):
        """Perform a paired permutation test between two datasets."""
        observed_diff = np.nanmean(data1) - np.nanmean(data2)
        combined = np.concatenate((data1, data2))
        
        more_extreme = 0
        for _ in range(num_permutations):
            np.random.shuffle(combined)
            perm_diff = np.nanmean(combined[:len(data1)]) - np.nanmean(combined[len(data1):])
            if abs(perm_diff) >= abs(observed_diff):
                more_extreme += 1
        
        p_value = (more_extreme + 1) / (num_permutations + 1)  # Adding 1 to avoid p=0
        return observed_diff, p_value
    
    def plot_coupling_index_across_celltypes_cdf(self,results_list, model_types, threshold=0.05, comparisons=[('No Coupling', 'All')], significant_neurons=None, xlim_val = 1, recalculate_modulation=False):
        """
        Plot the CDF of coupling index across datasets for multiple models, separated by cell type.

        Parameters:
            results_list (list of dict): List of results dictionaries for each model.
            celltypecolors (dict): Dictionary of colors for each cell type.
            save_results (str): Path to save the results.
            model_types (list of str): List of model types corresponding to each results dictionary.
            color_map_dict (dict): Dictionary mapping (celltype, model) to specific colors.
            threshold (float, optional): Threshold for filtering outlier values. Default is 1.
            comparisons (list of tuple of str, optional): List of comparison labels to use, e.g., [('No Coupling', 'All'), ('No Pyr', 'All')].
            significant_neurons (dict, optional): Dictionary with dataset keys mapping to lists of significant neuron indices.
            recalculate_modulation (bool, optional): If True, recalculate modulation index using neurons present across all comparisons and models.

        Returns:
            dict: Dictionary containing coupling indices by cell type and model for each comparison.
        """
        coupling_index_by_comparison = {}
        # Initialize a set to keep track of neuron indices used across all models 
        used_neurons_set = None
        # Initialize dictionary to track used neurons for each dataset
        used_neurons = {}  # To track used neurons for each comparison and model

        # Loop over each comparison
        for comparison in comparisons:
            # Initialize a dictionary to store coupling indices by cell type and model for the current comparison
            coupling_index_by_celltype = {
                'pyr': {model: [] for model in model_types},
                'som': {model: [] for model in model_types},
                'pv': {model: [] for model in model_types}
            }
            
            # Initialize used neurons set for this comparison
            used_neurons[comparison] = {model: set() for model in model_types}

            # Loop over each model's results
            for model_idx, results in enumerate(results_list):
                model_type = model_types[model_idx]
                

                # Initialize a dictionary to store the mean deviance explained values across all datasets
                mean_dev_dict = {
                    'No Coupling': [],
                    'All': [],
                    'No Pyr': [],
                    'No Som': [],
                    'No Pv': []
                }
                
                # Initialize a list to store all cell labels
                cell_labels = []
                
                # Loop over each dataset in the results dictionary
                for dataset_key, dataset in results.items():
                    # Get significant neurons for the dataset if provided
                    sig_neurons = significant_neurons.get(dataset_key, None) if significant_neurons else None
                    
                    # Filter data based on significant neurons if applicable
                    if sig_neurons is not None:
                        sig_neurons = np.array(sig_neurons[0], dtype=np.uint16)
                    else:
                        sig_neurons = np.arange(len(dataset['celltype_array']))  # Use all neurons if not specified

                    
                    # Filter mean deviance values
                    mean_dev_dict['No Coupling'].extend(np.array(dataset['mean_dev_behav'])[sig_neurons])
                    mean_dev_dict['All'].extend(np.array(dataset['mean_dev'])[sig_neurons])
                    mean_dev_dict['No Pyr'].extend(np.array(dataset['mean_dev_no_pyr'])[sig_neurons])
                    mean_dev_dict['No Som'].extend(np.array(dataset['mean_dev_no_som'])[sig_neurons])
                    mean_dev_dict['No Pv'].extend(np.array(dataset['mean_dev_no_pv'])[sig_neurons])
                    
                    # Map cell IDs to cell types and add to the cell_labels list
                    cell_types = {
                        0: 'pyr',
                        1: 'som',
                        2: 'pv',
                    }
                    cell_labels.extend([cell_types[cell_id] for cell_id in np.array(dataset['celltype_array'])[sig_neurons]])

                
                # Extract labels for the comparison
                label1, label2 = comparison
                
                # Identify outliers in the 'All' condition
                outlier_indices_all = np.where(np.array(mean_dev_dict['All']) < threshold)[0]


                # # Collect mean deviance values for outlier detection from the first model type
                # if model_idx == 0:  # Use only the first model's data for outlier detection
                #     all_mean_dev_all = np.array(mean_dev_dict['All'])

                # # Identify outliers in the 'All' condition based on the first model type
                # outlier_indices_all = np.where(all_mean_dev_all < threshold)[0]
                
                # Prepare data for comparison
                full_data = np.array(mean_dev_dict[label2]).astype(np.float64)
                partial_data = np.array(mean_dev_dict[label1]).astype(np.float64)

                # Mask outliers
                if len(outlier_indices_all) > 0:
                    #print(outlier_indices_all)
                    print(f'model type: {model_type}, original length, {len(full_data)}')
                    full_data[outlier_indices_all] = np.nan
                    partial_data[outlier_indices_all] = np.nan

                # Calculate coupling index
                coupling_index = self.calculate_coupling_index(full_data, partial_data)

                #eliminate coupling indices that are greater than one
                bad_coupling = np.where(coupling_index>1)
                coupling_index[bad_coupling] = np.nan

                bad_coupling = np.where(coupling_index<-1)
                coupling_index[bad_coupling] = np.nan

                # Update the used neurons for each dataset (non-NaN neurons) 
                valid_neurons = ~np.isnan(coupling_index) 
                used_neurons[comparison][model_type] = np.where(valid_neurons)[0]
                
                # Update the global used neurons set
                if used_neurons_set is None:
                    used_neurons_set = set(used_neurons[comparison][model_type])
                else:
                    used_neurons_set &= set(used_neurons[comparison][model_type])


                # Separate coupling index by cell type
                for idx, cell_label in enumerate(cell_labels):
                    coupling_index_by_celltype[cell_label][model_type].append(coupling_index[idx])

            # Store the results for the current comparison
            coupling_index_by_comparison[comparison] = coupling_index_by_celltype

        # Recalculate modulation index using only the neurons present across all comparisons and models if required
        if recalculate_modulation:
            print("Recalculating coupling index using neurons present across all comparisons and models...")


            # Initialize used_neurons_set for each cell type (to track common neurons across all comparisons and models)
            used_neurons_set_by_celltype = {
                'pyr': None,
                'som': None,
                'pv': None
            }

            # Iterate over all comparisons
            for comparison in comparisons:
                # Iterate over all model types
                for model_type in model_types:
                    # For each cell type, find non-NaN neuron indices and keep intersection across comparisons and models
                    for cell_type in used_neurons_set_by_celltype.keys():
                        # Get the coupling index array for the current comparison, model type, and cell type
                        neuron_data = np.array(coupling_index_by_comparison[comparison][cell_type][model_type]) #np.array(coupling_index_by_celltype[cell_type][model_type])
                        
                        # Identify non-NaN neuron indices
                        valid_neurons = np.where(~np.isnan(neuron_data))[0]
                        
                        # If this is the first time, initialize the set with valid neurons
                        if used_neurons_set_by_celltype[cell_type] is None:
                            used_neurons_set_by_celltype[cell_type] = set(valid_neurons)
                        else:
                            # Take intersection of valid neurons across models and comparisons
                            used_neurons_set_by_celltype[cell_type] = used_neurons_set_by_celltype[cell_type].intersection(valid_neurons)

            # Now used_neurons_set_by_celltype will contain only the neurons common across all comparisons and model types for each cell type
            # You can now check and print the number of non-NaN neurons across all comparisons and models

            for cell_type, neuron_set in used_neurons_set_by_celltype.items():
                print(f"Cell Type: {cell_type}, Number of common non-NaN neurons across all comparisons and models: {len(neuron_set)}")

            # Update coupling_index_by_comparison with the new common neurons
            for comparison in comparisons:
                for model_type in model_types:
                    for cell_type, neuron_set in used_neurons_set_by_celltype.items():
                        # Convert neuron set back to a list of indices for easy access
                        common_neurons_indices = np.array(list(neuron_set))
                        
                        # Loop through the neuron data for the current comparison, model type, and cell type
                        neuron_data = coupling_index_by_comparison[comparison][cell_type][model_type]
                        for idx, cell_label in enumerate(neuron_data):
                            # If the index is not in the common neuron set, mark it as NaN
                            if idx not in common_neurons_indices:
                                coupling_index_by_comparison[comparison][cell_type][model_type][idx] = np.nan

                        #CODE BELOW GIVES SAME NUMBER OF NEURONS IN EACH CONDITION (ie all active comparisons have the same BUT not across conditions)           
            # for comparison in comparisons:
            #     for cell_type in ['pyr', 'som', 'pv']:
            #         for model_type in model_types:
            #             # Filter out neurons not present in the global used_neurons_set
            #             neuron_indices = used_neurons_set.intersection(used_neurons[comparison][model_type])
            #             filtered_indices = np.array(list(neuron_indices))
            #             #print(f'{comparison} {cell_type} {model_type} total neurons {len(filtered_indices)}')

            #             for idx, cell_label in enumerate(coupling_index_by_comparison[comparison][cell_type][model_type]):
            #                 if idx not in filtered_indices:
            #                     coupling_index_by_comparison[comparison][cell_type][model_type][idx] = np.nan

        for comparison in comparisons:
            label1, label2 = comparison
            # Paired permutation test for each cell type between model types
            for cell_type in self.plotter.celltypecolors.keys():
                print(f"\n{cell_type.upper()} Cell Type: {label1} vs {label2}")
                
                for model_a, model_b in zip(model_types[:-1], model_types[1:]):
                    data1 = np.array(coupling_index_by_comparison[comparison][cell_type][model_a]) # used to be coupling_index_by_celltype
                    data2 = np.array(coupling_index_by_comparison[comparison][cell_type][model_b])

                    # Perform paired permutation test
                    observed_diff, p_value = self.paired_permutation_test(data1, data2)

                    print(f"Model {model_a} vs {model_b}:")
                    print(f"Observed Difference: {observed_diff:.4f}, P-value: {p_value:.4f}")

            # Plot the CDF of coupling index for each cell type for the current comparison
            # Set global font size and family 
            plt.rcParams.update({'font.size': 14, 'font.family': 'arial'})
            plt.figure(figsize=(8, 2.67))

            for i, (celltype, _) in enumerate(self.plotter.celltypecolors.items()):
                ax = plt.subplot(1, 3, i+1)
                
                legend_elements = []

                # Plot the CDF for each model within the current cell type
                for model_type in model_types:

                    sorted_data = np.sort(coupling_index_by_comparison[comparison][celltype][model_type]) #coupling_index_by_celltype
                    print(f'model type: {model_type}, {celltype},{len(sorted_data)}, original length, {len(full_data)}')
                    x1 = np.linspace(0, 1, 100)  # Define range of x values
                    n1, _ = np.histogram(sorted_data, bins=x1)  # Histogram counts
                    p1 = n1 / np.sum(n1)  # Probability
                    cdf = np.cumsum(p1)  # Cumulative sum to get CDF
                    
                    # Use color_map_dict to assign the specific color
                    plt.plot(x1[:-1], cdf, label=f'{model_type}', color=self.plotter.color_map_dict[(celltype, model_type)], linewidth=2)

                    legend_elements.append(Line2D([0], [0], color=self.plotter.color_map_dict[(celltype, model_type)], lw=2, label=model_type))

                plt.title(f'{label1} vs {label2}') #{celltype} - 
                plt.xlabel('Coupling Index')
                if i == 0:
                    plt.ylabel('Cumulative Fraction')

                # Define the ticks you want (e.g., from 0 to 1 with increments of 0.1)
                ticks = np.arange(0, 1.1, 0.2)  # The 1.1 ensures that 1.0 is included in the ticks

                # Set the format for both x and y axis ticks to show one decimal place
                plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
                # Ensure ticks are from 0 to 1 with consistent intervals
                plt.xticks(np.arange(0, 1.2, 0.2))
                plt.yticks(np.arange(0, 1.2, 0.2))

                # Custom legend
                # Add legend with colored labels
                legend = ax.legend(handles=legend_elements, frameon=False, loc='best', handlelength=0, handletextpad=0.1)

                # Set the color of the legend text to match the corresponding model types
                for text in legend.get_texts():
                    # Extract the model type from the legend entry
                    model_type = text.get_text()
                    # Retrieve the color from color_map_dict using the current cell type and model type
                    color = self.plotter.color_map_dict.get((celltype, model_type), 'black')
                    text.set_color(color)

                # Clean up the appearance
                ax = plt.gca()
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.set_box_aspect(1)
                ax.set_xlim(0, xlim_val)
                ax.set_ylim(0, 1)

            #to save svg so that we can edit texts!
            new_rc_params = {'text.usetex': False,
            "svg.fonttype": 'none'
            }
            plt.rcParams.update(new_rc_params)

            # Save the figure for the current comparison
            plt.tight_layout()
            if significant_neurons is not None:
                save_string = f'cdf_coupling_index_comparison_{label1}_vs_{label2}_by_celltype_sigcells.png'
                save_string = f'cdf_coupling_index_comparison_{label1}_vs_{label2}_by_celltype_sigcells.svg'
                # save_string = f'cdf_coupling_index_comparison_{label1}_vs_{label2}_by_celltype_sigcells.pdf'
            else:
                save_string = f'cdf_coupling_index_comparison_{label1}_vs_{label2}_by_celltype_{model_types}.png'
                save_string = f'cdf_coupling_index_comparison_{label1}_vs_{label2}_by_celltype_{model_types}.svg'
                # save_string = f'cdf_coupling_index_comparison_{label1}_vs_{label2}_by_celltype.pdf'
            plt.savefig(os.path.join(self.plotter.save_results, save_string))
            plt.show()

        return coupling_index_by_comparison, used_neurons
    
    def bar_plot_separated_coupling_index_diff(self, coupling_index_by_celltype, comparisons, model_pairs, colors, measure_string, bar_width=0.5, save_path=None, minmax=(0, 0.1), xaxislabel = None):
        """
        Create a bar plot with error bars to compare coupling index differences across cell types within each comparison.

        Parameters:
        coupling_index_by_celltype: dict
            A nested dictionary where outer keys are comparisons, second-level keys are cell types, and inner keys are model types, with values as the coupling index.
        comparisons: list
            A list of comparisons to include in the plot.
        model_pairs: list of tuples
            A list of tuples where each tuple contains two model types to compare (e.g., [('Model_Pre', 'Model_ITI'), ...]).
        colors: dict
            A dictionary where keys are cell types and values are colors for the bars.
        measure_string: str
            The label for the y-axis.
        bar_width: float
            The width of the bars in the plot.
        save_path: str, optional
            The path to save the plot. If None, the plot is displayed instead of saved.
        minmax: tuple
            Tuple specifying the y-axis limits (min, max).
        """
        # For collecting all p-values for Bonferroni correction
        all_p_values = []

        # Set global font size and family 
        plt.rcParams.update({'font.size': 14, 'font.family': 'arial'})

        # Create a figure with subplots, one for each comparison
        fig, axs = plt.subplots(1, len(comparisons), figsize=(6, 3), sharey=True)
        
        # Iterate over comparisons to create each subplot
        for i, comparison in enumerate(comparisons):
            ax = axs[i]  # Select the subplot
            all_pos = []
            bar_heights_al = []
            # Iterate over each cell type in the current comparison
            for j, (cell_type, cell_data) in enumerate(coupling_index_by_celltype[comparison].items()):
                # Calculate positions for each model comparison along the x-axis
                positions = np.arange(len(model_pairs)) + 1 + j * (bar_width + 0.1)
                all_pos.append(positions)
                means = []
                errors = []
                
                # Iterate over each model comparison and compute differences
                for model1, model2 in model_pairs:
                    if model1 in cell_data and model2 in cell_data:
                        differences = np.array(cell_data[model1]) - np.array(cell_data[model2])
                        means.append(np.nanmean(differences))
                        errors.append(np.nanstd(differences) / np.sqrt(np.sum(~np.isnan(differences))))  # Standard error ignoring NaNs
                    else:
                        means.append(np.nan)
                        errors.append(np.nan)
                
                # Create bar plot with uncolored inside and colored outlines
                bars = ax.bar(positions, means, yerr=errors, capsize=2, 
                            edgecolor=colors[cell_type],
                            facecolor='white', linewidth=1.5, width=bar_width, ecolor=colors[cell_type])
                
                # Get heights of the current bars
                # Get heights of the current bars
                bar_heights = [bar.get_height() for bar in bars]
                bar_heights_al.append(bar_heights)

            # Statistical tests for differences from zero using Wilcoxon signed-rank test
            significance_stars = []
            for model1, model2 in model_pairs:
                p_values_for_this_comparison = []
                for cell_type, cell_data in coupling_index_by_celltype[comparison].items():
                    if model1 in cell_data and model2 in cell_data:
                        differences = np.array(cell_data[model1]) - np.array(cell_data[model2])
                        #Remove NaNs
                        differences = differences[~np.isnan(differences)]
                        if np.sum(~np.isnan(differences)) > 0:
                            # Perform Wilcoxon signed-rank test against zero
                            stat, p_value = wilcoxon(differences, alternative='greater')
                            p_values_for_this_comparison.append(p_value)
                            all_p_values.append(p_value)
                            print(f"{comparison} - {cell_type}: {model1} vs {model2} Wilcoxon p-value: {p_value:.5f}")
                            # Collect bar positions for drawing lines and stars

            if xaxislabel is None:
                ax.set_xticks(np.arange(len(model_pairs)) + 1 + bar_width)
                ax.set_xticklabels([f'{model1} - {model2}' for model1, model2 in model_pairs], rotation=45, ha='right', fontsize=10)
                ax.tick_params(axis='x', which='major', length=0)
                # Set labels and title for each subplot
                #ax.set_title(f'{comparison}', fontsize=14)
            else:
                # Use provided x-axis labels for each subplot
                if i < len(xaxislabel):  # Ensure there are enough labels for each comparison
                    ax.set_xticks(np.squeeze(all_pos))
                    ax.set_xticklabels(xaxislabel[i], rotation=45, ha='right', fontsize=10)
                    #ax.set_title(f'{comparison}', fontsize=14)
                    ax.set_title(f'{xaxislabel[i][0][:3]} coupling', fontsize=14)
                else:
                    ax.set_xticks(np.arange(len(model_pairs)) + 1 + bar_width)
                    ax.set_xticklabels([f'{model1} - {model2}' for model1, model2 in model_pairs], rotation=45, ha='right', fontsize=10)

            # Clean up the appearance
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if i == 0:
                ax.set_ylabel(f'{measure_string}', fontsize=14)

            ax.set_ylim(minmax[0], minmax[1])
            

            # Calculate Bonferroni significance
            corrected_p_values, significance_stars = self.calculate_bonferroni_significance(p_values_for_this_comparison)

            # Draw significance lines and stars
            for idx, (p, star) in enumerate(zip(corrected_p_values, significance_stars)):
                if star != 'ns':
                    bottom, top = axs[0].get_ylim()
                    top = minmax[1] - 0.1
                    top = bar_heights_al[idx][0]
                    print(top)
                    y = top + (idx * 0.02)  # Adjust y-coordinate
                    # x1 = bar_positions[idx] - bar_width / 2
                    # x2 = bar_positions[idx] + bar_width / 2
                    x1 = all_pos[idx]
                    self.plotter.add_significance_line(axs[i], x1, y=y, significance=star)
        
        # Add a global title for the figure
        #fig.suptitle(f'{measure_string} Differences Across Comparisons and Cell Types', fontsize=16)
        
        # Adjust layout
        #plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.3)

        #to save svg so that we can edit texts!
        new_rc_params = {'text.usetex': False,
        "svg.fonttype": 'none'
        }
        plt.rcParams.update(new_rc_params)


        # Save plot if save_path is provided
        if save_path: 
            os.chdir(save_path)
            plt.savefig(f'bar_diff_{model_pairs[0][0]}-{model_pairs[0][1]}_coupling_celltypesv1.svg', bbox_inches='tight')


    def bar_plot_separated_celltype_diff(self,coupling_index_by_celltype, cell_types, model_pairs, colors, measure_string, bar_width=0.3, save_path=None, minmax=(0, 0.1),comparisons=[('No Coupling', 'All')],xaxislabel = None):
        """
        Create a bar plot with error bars to compare coupling index differences across comparisons within each cell type.
        Perform statistical tests (Kruskal-Wallis) across cell types and paired permutation tests.

        Parameters:
        coupling_index_by_celltype: dict
            A nested dictionary where outer keys are comparisons, second-level keys are cell types, and inner keys are model types, with values as the coupling index.
        cell_types: list
            A list of cell types to include in the plot.
        model_pairs: list of tuples
            A list of tuples where each tuple contains two model types to compare (e.g., [('Model_Pre', 'Model_ITI'), ...]).
        colors: dict
            A dictionary where keys are cell types and values are colors for the bars.
        measure_string: str
            The label for the y-axis.
        bar_width: float
            The width of the bars in the plot.
        save_path: str, optional
            The path to save the plot. If None, the plot is displayed instead of saved.
        minmax: tuple
            Tuple specifying the y-axis limits (min, max).
        """
        # Set global font size and family 
        plt.rcParams.update({'font.size': 14, 'font.family': 'arial'})

        # Create a figure with subplots, one for each cell type
        fig, axs = plt.subplots(1, len(cell_types), figsize=(6, 3), sharey=True)
        
        # Iterate over cell types to create each subplot
        for i, cell_type in enumerate(cell_types):
            ax = axs[i]  # Select the subplot
            all_pos = []
            # Iterate over each comparison in the current cell type
            for j, comparison in enumerate(comparisons):
                cell_data = coupling_index_by_celltype[comparison][cell_type]
                
                # Calculate positions for each model comparison along the x-axis
                positions = np.arange(len(model_pairs)) + 1 + j * (bar_width + 0.1)
                all_pos.append(positions)
                means = []
                errors = []
                
                # Iterate over each model comparison and compute differences
                for model1, model2 in model_pairs:
                    if model1 in cell_data and model2 in cell_data:
                        differences = np.array(cell_data[model1]) - np.array(cell_data[model2])
                        means.append(np.nanmean(differences))
                        errors.append(np.nanstd(differences) / np.sqrt(np.sum(~np.isnan(differences))))  # Standard error ignoring NaNs
                        print(f'non nan cells total: {np.sum(~np.isnan(differences))}')
                    else:
                        means.append(np.nan)
                        errors.append(np.nan)
                
                # Create bar plot with uncolored inside and colored outlines
                bars = ax.bar(positions, means, yerr=errors, capsize=2, 
                            edgecolor=colors[cell_type],
                            facecolor='white', linewidth=1.5, width=bar_width, ecolor=colors[cell_type])


            # Statistical tests
            # Statistical tests (Kruskal-Wallis)
            # for model_idx, (model1, model2) in enumerate(model_pairs):
            #     # Collect data across cell types for this model pair comparison
            #     data_by_celltype = []
            #     cell_types_for_test = []
            #     for cell_type_test in cell_types:
            #         cell_data_test = coupling_index_by_celltype[comparison][cell_type_test]
            #         if model1 in cell_data_test and model2 in cell_data_test:
            #             differences = np.array(cell_data_test[model1]) - np.array(cell_data_test[model2])
                        
            #             # Remove NaNs
            #             differences = differences[~np.isnan(differences)]
                        
            #             if len(differences) > 0:
            #                 data_by_celltype.append(differences)
            #                 cell_types_for_test.append(cell_type_test)
                
            #     # Perform Kruskal-Wallis test across cell types if there's enough data
            #     if len(data_by_celltype) > 1 and all(len(data) > 0 for data in data_by_celltype):
            #         test_stat, p_value = kruskal(*data_by_celltype)
                    
            #         print(f"Comparison: {comparison}, Model: {model1} vs {model2}, Kruskal-Wallis p-value: {p_value:.5f}")

            #         # If significant, perform paired permutation tests between cell types
            #         if p_value < 0.05:
            #             for (cell_type1, data1), (cell_type2, data2) in itertools.combinations(zip(cell_types_for_test, data_by_celltype), 2):
            #                 perm_p_value = paired_permutation_test(data1, data2)  # Assuming this function is defined elsewhere
            #                 # Ensure that cell_type1 and cell_type2 are strings by converting them before formatting
            #                 print(f"permutation test p-value: {str(perm_p_value)}")


            # Set labels and title for each subplot
            if xaxislabel is None:
                ax.set_xticks(np.arange(len(model_pairs)) + 1 + (len(coupling_index_by_celltype) - 1) * (bar_width + 0.1) / 2)
                ax.set_xticklabels([f'{model1} - {model2}' for model1, model2 in model_pairs], rotation=45, ha='right', fontsize=10)
                ax.tick_params(axis='x', which='major', length=0)
                # Set labels and title for each subplot
                #ax.set_title(f'{comparison}', fontsize=14)
            else:
                # Use provided x-axis labels for each subplot
                if i < len(xaxislabel):  # Ensure there are enough labels for each comparison
                    ax.set_xticks(np.squeeze(all_pos))
                    ax.set_xticklabels(xaxislabel[i], rotation=45, ha='right', fontsize=10)
                    #ax.set_title(f'{comparison}', fontsize=14)
                    ax.set_title(f'{xaxislabel[i][0][:3]}', fontsize=14)
                else:
                    ax.set_xticks(np.arange(len(model_pairs)) + 1 + bar_width)
                    ax.set_xticklabels([f'{model1} - {model2}' for model1, model2 in model_pairs], rotation=45, ha='right', fontsize=10)


            # Clean up the appearance
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if i == 0:
                ax.set_ylabel(f'{measure_string}', fontsize=14)

            ax.set_ylim(minmax[0], minmax[1])
        
        # Add a global title for the figure
        #fig.suptitle(f'{measure_string} Differences Across Cell Types and Comparisons', fontsize=16)
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.3)

        #to save svg so that we can edit texts!
        new_rc_params = {'text.usetex': False,
        "svg.fonttype": 'none'
        }
        plt.rcParams.update(new_rc_params)

        # Save plot if save_path is provided
        if save_path: 
            os.chdir(save_path)
            plt.savefig(f'bar_diff_{model_pairs[0][0]}-{model_pairs[0][1]}_coupling_celltypeseparated.svg', bbox_inches='tight')

        plt.show()

    def scatter_plot_separated_celltype_mean(self,coupling_index_by_celltype, model_pairs, measure_string, minmax=(0, 0.1),markerstyles = "o",version = 1,comparisons=None):
        """
        Create a scatter plot comparing mean coupling indices between two models for each cell type.

        Parameters:
        coupling_index_by_celltype: dict
            A nested dictionary where outer keys are comparisons, second-level keys are cell types, and inner keys are model types, with values as the coupling index.
        cell_types: list
            A list of cell types to include in the plot.
        model_pairs: list of tuples
            A list of tuples where each tuple contains two model types to compare (e.g., [('Model_Pre', 'Model_ITI'), ...]).
        colors: dict
            A dictionary where keys are cell types and values are colors for the scatter points.
        measure_string: str
            The label for the axes.
        save_path: str, optional
            The path to save the plot. If None, the plot is displayed instead of saved.
        minmax: tuple
            Tuple specifying the axis limits (min, max) for both x and y axes.
        markerstyles: str or list
            Marker style(s) for different comparisons.
        version: int
            Version of the plot style (1 or 2).
        comparisons: list, optional
            List of specific comparisons to include. If None, all comparisons in `coupling_index_by_celltype` are used.

        """
        # Set global font size and family 
        plt.rcParams.update({'font.size': 14, 'font.family': 'arial'})

        # Use all comparisons if none are provided
        if comparisons is None:
            comparisons = list(coupling_index_by_celltype.keys())
        
        cell_types = self.plotter.celltypecolors.keys()
        colors = self.plotter.celltypecolors

        if version ==1: 
            # Create a figure with subplots, one for each cell type
            fig, axs = plt.subplots(1, len(cell_types), figsize=(9, 3), sharex=True, sharey=True)
            # Iterate over cell types to create each subplot
            for i, cell_type in enumerate(cell_types):
                ax = axs[i]  # Select the subplot
                handles = []  # To store scatter plot handles for the legend
                labels = []  # To store labels for the legend
                
                # Iterate over each comparison in the current cell type
                for cmp,comparison in comparison in enumerate(comparisons):
                    cell_data = coupling_index_by_celltype[comparison][cell_type]
                    
                    # Iterate over each model comparison and compute mean coupling indices
                    for model1, model2 in model_pairs:
                        if model1 in cell_data and model2 in cell_data:
                            mean_model1 = np.nanmean(cell_data[model1])  # Mean across neurons for model1
                            mean_model2 = np.nanmean(cell_data[model2])  # Mean across neurons for model2

                            # Plot the mean of model1 on the x-axis vs. model2 on the y-axis
                            scatter = ax.scatter(mean_model1, mean_model2, color=colors[cell_type], alpha=0.7, s=60, label=f'{model1} vs {model2}', edgecolor='black', marker = markerstyles[cmp])

                            # # Clean up the appearance
                            ax.spines['top'].set_visible(False)
                            ax.spines['right'].set_visible(False)
                            ax.set_box_aspect(1)

                            # Store the handle and label for legend
                            handles.append(scatter)
                            labels.append(f'{comparison[1]}')
                
                # Set labels and title for each subplot
                ax.set_title(f'{cell_type}', fontsize=14)
                if i == 0:
                    ax.set_ylabel(f'{measure_string} {model2}', fontsize=14)
                ax.set_xlabel(f'{measure_string} {model1}', fontsize=14)
                ax.set_xlim(minmax[0], minmax[1])
                ax.set_ylim(minmax[0], minmax[1])

                # Add a 45-degree diagonal line to show where x = y
                ax.plot([minmax[0], minmax[1]], [minmax[0], minmax[1]], linestyle='--', color='gray', linewidth=1)

                
                # Add the legend for this subplot
                ax.legend(handles, labels, loc= (.7,0), fontsize=10, frameon=False,handletextpad=0.1) #'lower right'

                ticks = np.arange(minmax[0], minmax[1], 0.1)
                plt.xticks(ticks)
                plt.yticks(ticks)

                # Adjust layout
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                plt.tight_layout(rect=[0, 0, .80, 1]) #rect=[0, 0, 1, 0.95]
        
        else:
            # Create a figure with subplots, one for each cell type
            fig, axs = plt.subplots(1, 1, figsize=(3, 3), sharex=True, sharey=True)
            #fig, axs = plt.subplots(1, 1, figsize=(10,10), sharex=True, sharey=True)
            
            for i, cell_type in enumerate(cell_types):
                ax = axs  # Select the subplot
                handles = []  # To store scatter plot handles for the legend
                labels = []  # To store labels for the legend
                
                # Iterate over each comparison in the current cell type
                for cmp, comparison in enumerate(comparisons):
                    cell_data = coupling_index_by_celltype[comparison][cell_type]
                    
                    # Iterate over each model comparison and compute mean coupling indices
                    for model1, model2 in model_pairs:
                        if model1 in cell_data and model2 in cell_data:
                            non_nan1 = ~np.isnan(cell_data[model1])
                            non_nan2 = ~np.isnan(cell_data[model2])
                            good_neurons = non_nan1 & non_nan2
                            print(f'Comparison {comparison}, cell type {cell_type}, neurons used: {np.sum(good_neurons)}')
                            
                            mean_model1 = np.nanmean(cell_data[model1])  # Mean across neurons for model1
                            mean_model2 = np.nanmean(cell_data[model2])  # Mean across neurons for model2

                            # Plot the mean of model1 on the x-axis vs. model2 on the y-axis
                            #scatter = ax.scatter(mean_model1, mean_model2, color=colors[cell_type], alpha=0.7, s=50, label=f'{model1} vs {model2}', edgecolor='black', marker = markerstyles[cmp])
                            scatter = ax.scatter(mean_model1, mean_model2,facecolors='none',  alpha=1, s=50, label=f'{model1} vs {model2}', edgecolor=colors[cell_type], marker = markerstyles[cmp], linewidths=1.2)
                            print(f'cell_type {cell_type}, {mean_model1},{mean_model2}')

                            # # Clean up the appearance
                            ax.spines['top'].set_visible(False)
                            ax.spines['right'].set_visible(False)
                            ax.set_box_aspect(1)

                            # Store the handle and label for legend
                            handles.append(scatter)
                            labels.append(f'{comparison[0]}')
                
                # Set labels and title for each subplot
                ax.set_title(f'{cell_type}', fontsize=14)
                if i == 0:
                    ax.set_ylabel(f'{measure_string} {model2}', fontsize=16)
                ax.set_xlabel(f'{measure_string} {model1}', fontsize=16)
                ax.set_xlim(minmax[0], minmax[1])
                ax.set_ylim(minmax[0], minmax[1])

                # Add a 45-degree diagonal line to show where x = y
                ax.plot([minmax[0], minmax[1]], [minmax[0], minmax[1]], linestyle='--', color='gray', linewidth=1)

                ticks = np.arange(minmax[0], minmax[1], 0.1)
                plt.xticks(ticks)
                plt.yticks(ticks)

                # Add the legend for this subplot
                ax.legend(handles, labels, loc= (.7,0), fontsize=10, frameon=False,handletextpad=0.1) #'lower right'

        # Add a global title for the figure
        fig.suptitle(f'{measure_string} Comparison Across Cell Types', fontsize=16)
        

        #to save svg so that we can edit texts!
        new_rc_params = {'text.usetex': False,
        "svg.fonttype": 'none'
        }
        plt.rcParams.update(new_rc_params)

        

        # Save plot if save_path is provided
        if self.plotter.save_results: 
            os.chdir(self.plotter.save_results)
            plt.savefig(f'scatter_{model1}_vs_{model2}_coupling_celltypes_version{version}.svg', bbox_inches='tight')
        
        plt.show()


    def plot_coupling_index_across_datasets(self, results, model_type, significant_neurons=None, threshold=0.05, comparisons = [
                ('No Coupling', 'All'),
                ('No Coupling', 'No Pyr'),
                ('No Coupling', 'No Som'),
                ('No Coupling', 'No Pv'),
                ('No Pyr', 'All'),
                ('No Som', 'All'),
                ('No Pv', 'All')
            ]):
        # Initialize dictionaries to store coupling indices and model improvements across datasets
        coupling_index_dir = {}
        model_improvement = {}
        
        # Initialize an empty dictionary to store the mean deviance explained values across all datasets
        mean_dev_dict = {
            'No Coupling': [],
            'All': [],
            'No Pyr': [],
            'No Som': [],
            'No Pv': []
        }
        
        # Initialize a list to store all cell labels
        cell_labels = []
        
        # Loop over each dataset in the results dictionary
        for dataset_key, dataset in results.items():
            # Determine the cell labels based on significant neurons if provided
            if significant_neurons and dataset_key in significant_neurons:
                cell_ids = significant_neurons[dataset_key][0]
            else:
                cell_ids = range(len(dataset['celltype_array']))  # Default to all cells

            # Append the data from each dataset to the corresponding list in mean_dev_dict
            mean_dev_dict['No Coupling'].extend(dataset['mean_dev_behav'][cell_ids])
            mean_dev_dict['All'].extend(dataset['mean_dev'][cell_ids])
            mean_dev_dict['No Pyr'].extend(dataset['mean_dev_no_pyr'][cell_ids])
            mean_dev_dict['No Som'].extend(dataset['mean_dev_no_som'][cell_ids])
            mean_dev_dict['No Pv'].extend(dataset['mean_dev_no_pv'][cell_ids])
            
            # Map cell IDs to cell types and add to the cell_labels list
            cell_types = {
                0: 'pyr',
                1: 'som',
                2: 'pv',
            }
            cell_labels.extend([cell_types[dataset['celltype_array'][i]] for i in cell_ids])
        
        # Change directory to the results folder
        os.chdir(self.plotter.save_results)
        
        # List of comparisons
        comparisons = comparisons

        outlier_indices_all = np.where(np.array(mean_dev_dict['All']) < threshold)
        num_cells = np.shape(mean_dev_dict['All'])[0]
        print(f"Outliers: {outlier_indices_all}")
        print(f"Percent of cells that are outliers!: {outlier_indices_all[0].size / num_cells}")
        
        # Create scatter plots and calculate coupling index for each comparison
        for (label1, label2) in comparisons:
            full_label = f'{label1} vs {label2}'

            # Convert to np.float64
            full_data = np.array(mean_dev_dict[label2]).astype(np.float64)
            partial_data = np.array(mean_dev_dict[label1]).astype(np.float64)

            # NaN outlier indices
            full_data[outlier_indices_all] = np.nan
            partial_data[outlier_indices_all] = np.nan

            # Calculate coupling index
            coupling_index = self.calculate_coupling_index(full_data, partial_data)
            coupling_index_dir[full_label] = coupling_index 

            # Calculate model improvement
            full_label_diff = f'{label2}-{label1}'
            model_improvement[full_label_diff] = full_data - partial_data
            
            # Construct the save string to reflect the comparison being made
            if significant_neurons is not None:
                save_string = f'cdf_coupling_index_sig_{label1}_vs_{label2}_{model_type}.png'
            else:
                save_string = f'cdf_coupling_index_{label1}_vs_{label2}_{model_type}.png'
            
            # Plot and save the CDF of the coupling index
            self.plotter.plot_cdf_coupling_index(coupling_index, cell_labels, colors=self.plotter.celltypecolors, title=f'{label1} vs {label2}', save_path=save_string)

        return coupling_index_dir, model_improvement