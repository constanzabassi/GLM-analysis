from xml.sax import handler

import numpy as np
import os
import pickle
import scipy
import random
import pandas as pd

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
import itertools

#IMPORT PLOTTING FUNCTIONS!
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import wilcoxon
from scipy.stats import pearsonr
from scipy.stats import sem
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as mc
from matplotlib_venn import venn2
from matplotlib.patches import Patch

#import stats class
from utils.general_stats import GeneralStats
from scipy.stats import chi2_contingency
from utils.GLMDataUtils import GLMDataUtils

from matplotlib.colors import to_hex
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from scipy.ndimage import gaussian_filter1d

# from handlers.glm_handler import DataHandlerEncoding
from handlers.DataHandlerEncoding import DataHandlerEncoding



class Plotter:
    def __init__(self, data, celltypecolors=None, save_results=None, color_map_dict = None, event_frames=None, event_labels=None, group_colors = None):
        """
        Initialize Plotter class with default colors and event frames
        
        Parameters:
        -----------
        data : dict, optional
            Data to plot
        celltypecolors : dict, optional
            Custom colors for cell types
        save_results : str, optional
            Path to save results
        color_map_dict : dict, optional
            Dictionary mapping (celltype, model) to specific colors
        event_frames : array-like, optional
            Frame indices of events to mark on plots. If None, uses default frames
        """

        self.data = data
        self.save_results = save_results
        self.color_map_dict = color_map_dict # Dictionary mapping (celltype, model) to specific colors.
        # Default event frames
        self.default_event_frames = np.array([6., 38., 70., 131., 145.])
        # Event meanings:
        # [0] (6): Sound onset/photostim
        # [3] (131): Choice
        # [4] (145): Outcome
        self.event_frames = event_frames if event_frames is not None else self.default_event_frames
        self.event_labels = event_labels if event_labels is not None else ['S1', 'S2', 'S3', 'T', 'R']

        # Default cell type colors
        self.default_colors = {
            'pyr': (0.37, 0.75, 0.49),
            'som': (0.17, 0.35, 0.8),
            'pv': (0.82, 0.04, 0.04)
        }

        self.default_group_colors = {
            'sound': (0.3, 0.2, 0.6),
            'opto': (1, 0.7, 0),
            'both': (0.3,0.8,1),
            'unmod': (0.7,0.7,0.7),
            'sound_pos': (0.3, 0.2, 0.6),
            'sound_neg': (0.3, 0.2, 0.6)
        }

        # Default variable colors (pairs for regular and shuffled) for each decoded variable
        self.default_variable_colors = {
            'sound_category': ['darkslateblue', 'mediumslateblue'],
            'choice': ['steelblue', 'lightskyblue'],
            'photostim': ['saddlebrown', 'darkorange'],
            'outcome': ['mediumvioletred', 'hotpink']
        }
        
        # Use custom colors if provided, otherwise use defaults
        self.celltypecolors = celltypecolors if celltypecolors is not None else self.default_colors
        self.group_colors = group_colors if group_colors is not None else self.default_group_colors

        self.default_cell_type_labels = {
            'pyr': 'Pyr',
            'som': 'SOM',
            'pv': 'PV'
        }
        # Use custom colors if provided, otherwise use defaults
        self.cell_type_labels = celltypecolors if celltypecolors is not None else self.default_cell_type_labels
        self.stats = GeneralStats()  # Instantiate GeneralStats
        self.glm_data_utils = GLMDataUtils() # Instantiate GLMDataUtils

    def add_significance_line(self,ax, x1, x2=None, y=None, significance='', color='black', star_height_percentage = 0.01, fontsize=7,lw=.5):
        """
        Add significance line between two bars in the plot.
        If only x1 is provided, draw only the significance star without a line.
        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            The axes to draw on
        x1, x2 : float
            x-coordinates for the line endpoints
        y : float, optional
            y-coordinate for the line. If None, uses 95% of ylim
        significance : str
            The significance marker to display
        color : str
            Color of the line and text
        """

        # Get current y-axis limits
        ylims = ax.get_ylim()
        
        # If y is not provided, set it to 95% of the y-axis range
        if y is None:
            y = ylims[1] *1.1

        if x2 is not None:  # Draw line if both x1 and x2 are provided
            # Calculate line height as small percentage of y-axis range
            line_height = y * star_height_percentage#(ylims[1] - ylims[0]) * star_height_percentage
            line_y = y 
            text_y = y + line_height

            # Draw the line
            ax.plot([x1, x1, x2, x2], [y, line_y, line_y, y], 
                    lw=lw, color=color)
            
            # Add text
            ax.text((x1 + x2) * 0.5, text_y, significance, 
                    ha='center', va='bottom', color=color, fontsize=fontsize, fontname='arial')
        else:  # Only x1 is provided, so draw only the significance star
            if y is not None:
                ax.text(x1, y, significance, 
                    ha='center', va='bottom', color=color, fontsize=fontsize, fontname='arial')
                    
    def generate_xlabels(self,cell_types_first_half, cell_types_second_half, connector='w'):
        """
        Generates combined x-axis labels for the bar plot based on two halves of cell types.

        Parameters:
        cell_types_first_half: list
            The list of cell types for the first half of the label (e.g., ['Pyr', 'SOM', 'PV']).
        cell_types_second_half: list
            The list of cell types for the second half of the label (e.g., ['Pyr', 'SOM', 'PV']).
        connector: str
            The word to connect the first and second halves (e.g., 'w' for 'with').

        Returns:
        list of lists
            A list of labels for each subplot based on combinations of the first and second half.
        """
        xlabels = []

        # Generate labels for each combination of first and second half
        for first_half in cell_types_first_half:
            labels = [f"{first_half} {connector} {second_half}" for second_half in cell_types_second_half]
            xlabels.append(labels)

        return xlabels

    def x_axis_sec_aligned(self, stim_frame, length_frames, interval=1, frame_rate=30):
        """
        Convert frame indices to seconds for x-axis ticks.
        
        Args:
            stim_frame (int): Frame number where the stimulus event occurs
            length_frames (int): Total number of frames
            interval (int): Interval for x-ticks (default is 1 second)
            frame_rate (int): Imaging frame rate (default is 30 Hz)
        
        Returns:
            xticks_in (list): Frame indices for x-ticks
            xticks_lab (list): Labels for x-ticks in seconds
        """
        frames_before = stim_frame 
        frames_after = len(np.arange(length_frames - stim_frame)) 
        
        time_before = np.arange(-frames_before, 1) / frame_rate
        time_after = np.arange(1, frames_after + 1) / frame_rate

        time_axis = np.concatenate((time_before, time_after))
         
        frame_indices = np.arange(stim_frame - frames_before, stim_frame + frames_after + 1) 
        
        x_tick_seconds = np.unique(np.floor(time_axis))
        x_tick_seconds = x_tick_seconds[x_tick_seconds % interval == 0]
        x_tick_indices = []

        # Ensure valid indices with proper array indexing
        valid_mask = np.isin(x_tick_seconds, time_axis, assume_unique=True)
        valid_indices = np.where(valid_mask)[0]  # Get array of indices directly

        # Index arrays with valid_indices
        x_tick_seconds = x_tick_seconds[valid_indices]
        x_tick_indices = [np.where(time_axis == sec)[0][0] for sec in x_tick_seconds]

        xticks_in = frame_indices[x_tick_indices] 
        xticks_lab = [str(int(sec)) for sec in x_tick_seconds]
        
        return xticks_in, xticks_lab

    def plot_with_seconds( self, stim_frame,length_frames, frame_rate=30,interval=1 ,ax=None):
        """
        Plot data with x-axis in seconds.
        
        Args:
            
            stim_frame (int): Frame number where the stimulus event occurs
            frame_rate (int): Imaging frame rate (default is 30 Hz)
            interval (int): Interval for x-ticks (default is 1 second)  
            ax = current axis
        """

        if ax is None:
            ax = plt.gca()

        length_frames = length_frames
        xticks_in, xticks_lab = self.x_axis_sec_aligned(stim_frame, length_frames, interval=interval, frame_rate=frame_rate)

        ax.set_xticks(xticks_in)
        ax.set_xticklabels(xticks_lab)
        ax.set_xlabel('Time (s)')

    #PREDICTOR PLOTTING FUNCTIONS
    # Create legend for coupling features
    def plot_feature_weights(self,server,animalID, date, model_type, model_chosen, pyr_count=3, som_count=3, pv_count=3, no_abs=1, directory = 'V:/Connie/ProcessedData/HA11-1R/2023-05-05/GLM_3nmf_iti/prepost trial cv 73 #1'):
        # Load actual response data

        handler = DataHandlerEncoding(data= None)
        feature_names = handler.load_behav_big_matrix_ids(directory = directory)
        # behav_big_matrix_ids_mat = scipy.io.loadmat(
        #     os.path.join(f'{server}/Connie/ProcessedData/{animalID}/{date}/{model_type}/prepost trial cv 73 #1', 
        #                 'behav_big_matrix_ids.mat')
        # )
        # behav_big_matrix_ids = behav_big_matrix_ids_mat['behav_big_matrix_ids']
        # feature_names = [name[0] for name in behav_big_matrix_ids[0]]  # Flatten the structure

        # Aggregate B_weights across all folds for coupling predictors
        B_weights_behavior_coupling = np.concatenate([model_chosen[fold]['B_weights'] for fold in model_chosen.keys()], axis=1)
        
        # Indices for coupling predictors
        coupling_predictors_indices = range(183, model_chosen[0]['B_weights'].shape[0])
        
        # Extract weights for other features and coupling features
        if no_abs == 1:
            other_weights = B_weights_behavior_coupling.mean(axis=1)
            coupling_weights = B_weights_behavior_coupling[coupling_predictors_indices, :].mean(axis=1)
        else:
            # Extract weights for other features and coupling features using absolute values
            other_weights = np.abs(B_weights_behavior_coupling).mean(axis=1)
            coupling_weights = np.abs(B_weights_behavior_coupling[coupling_predictors_indices, :]).mean(axis=1)
        
        # Create colors for coupling features
        coupling_colors = []
        coupling_colors.extend([self.celltypecolors['pyr']] * pyr_count)
        coupling_colors.extend([self.celltypecolors['som']] * som_count)
        coupling_colors.extend([self.celltypecolors['pv']] * pv_count)

        # Use a colormap for the other features
        cmap = plt.get_cmap('gist_ncar')
        unique_feature_names = list(set(feature_names))
        feature_colors = {name: cmap(i / len(unique_feature_names)) for i, name in enumerate(unique_feature_names)}
        other_colors = [feature_colors[name] for name in feature_names]

        # Plotting
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
        
        # Plot other features
        axes[0].bar(range(len(other_weights)), other_weights, color=other_colors)
        axes[0].axvline(x=182, linestyle='dashed', color='k', alpha=1)
        axes[0].set_xlabel('Feature Index')
        axes[0].set_ylabel('Average Coefficient')
        axes[0].set_title('All Feature Coefficients')
        
        # Create legend for other features 
        legend_elements_other = [Line2D([0], [0], color=feature_colors[name], lw=4, label=name) for name in unique_feature_names] 
        axes[0].legend(handles=legend_elements_other, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., frameon = False)

        # # Clean up the appearance
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)
        axes[0].set_box_aspect(1)

        # Plot coupling features
        axes[1].bar(range(len(coupling_weights)), coupling_weights, color=coupling_colors)
        axes[1].set_xlabel('Coupling Predictor Index')
        axes[1].set_ylabel('Average Coefficient')
        axes[1].set_title('Coupling Predictor Coefficients')
        axes[1].set_ylim(-.1,.8)

        # Create legend for coupling features
        legend_elements = [
            Line2D([0], [0], color=self.celltypecolors['pyr'], lw=4, label='Pyr'),
            Line2D([0], [0], color=self.celltypecolors['som'], lw=4, label='Som'),
            Line2D([0], [0], color=self.celltypecolors['pv'], lw=4, label='Pv')
        ]
        axes[1].legend(handles=legend_elements, frameon = False)

        # # Clean up the appearance
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)
        axes[1].set_box_aspect(1)

        # Adjust space between the subplots
        plt.subplots_adjust(wspace=1)  # Increase the width space between the plots
        
        # Save the figure
        os.chdir(self.save_results)
        plt.savefig(f'avg{no_abs}_beta_{animalID}_{date}_{model_type}.png')
        plt.show()

        return coupling_predictors_indices



    def plot_weights_heatmap(self,server,animalID, date, model_type, model_chosen, coupling_indices, cmap='coolwarm', no_abs=1,total_predictors=183, minmax=(-.2,.2), directory = 'V:/Connie/ProcessedData/HA11-1R/2023-05-05/GLM_3nmf_iti/prepost trial cv 73 #1'):
        """
        Plots a heatmap of mean weights across folds for each neuron.
        
        Parameters:
        - model_chosen: Dictionary containing the model output for each fold.
        - num_neurons: Number of neurons in the model.
        - feature_names: List of feature names corresponding to the model weights.
        - coupling_indices: Indices of coupling predictors within the weight matrix.
        - cmap: Colormap for the heatmap.
        """
        
        #find the mean across unique features 
        handler = DataHandlerEncoding(data= None)
        feature_names = handler.load_behav_big_matrix_ids(directory = directory)
        # behav_big_matrix_ids_mat = scipy.io.loadmat(
        #         os.path.join(f'{server}/Connie/ProcessedData/{animalID}/{date}/{model_type}/prepost trial cv 73 #1', 
        #                     'behav_big_matrix_ids.mat')
        #     )
        # behav_big_matrix_ids = behav_big_matrix_ids_mat['behav_big_matrix_ids']
        # feature_names = [name[0] for name in behav_big_matrix_ids[0]]
        unique_feature_names = list(set(feature_names))

        # Initialize an array to store the mean weights
        num_neurons = model_chosen[0]['B_weights'].shape[1]
        if coupling_indices is not None:
            num_coupling = len(coupling_indices) 
        else:
            num_coupling = 0   
        mean_weights = np.zeros((len(feature_names)+num_coupling, num_neurons))
        model_output = model_chosen
        
        # Calculate mean weights across folds for each neuron
        for fold in model_output.keys():
            B_weights_fold = model_output[fold]['B_weights']
            mean_weights += B_weights_fold

        # Divide by the number of folds to get the mean
        mean_weights /= len(model_output)

        # Apply absolute values if no_abs is set to 0
        if no_abs == 0:
            mean_weights = np.abs(mean_weights)

        # Extract weights for other features and coupling features
        other_weights = mean_weights[:total_predictors, :]  # Assuming non-coupling predictors are the first total_predictors rows
        if coupling_indices is not None:
            coupling_weights = mean_weights[coupling_indices, :]
            # Combine other weights and coupling weights for heatmap
            combined_weights = np.vstack([other_weights, coupling_weights])
        else:
            combined_weights = other_weights

        num_unique = len(unique_feature_names)+3
        mean_neuron_feature_unique = np.zeros((num_unique,num_neurons))
        total_beta_feature_unique = np.zeros((num_unique,num_neurons)) #sum of features!
        max_beta_feature_unique = np.zeros((num_unique,num_neurons))
        for idx,unique_f in enumerate(unique_feature_names):    
            # Find the indices where the current unique feature occurs in the feature names
            feature_indices = [i for i, name in enumerate(feature_names) if name == unique_f]

            #get the neuron weights for the unique feature
            neuron_features = combined_weights[feature_indices,:]
            mean_neuron_feature_unique[idx,:] = np.nanmean(neuron_features,axis = 0,)

            #add up all betas for the same feature!
            total_beta_feature_unique[idx,:] = np.sum(np.abs(neuron_features),axis = 0,)

            #find the max beta for each unique feature!
            max_beta_feature_unique[idx,:] = np.max(np.abs(neuron_features),axis = 0,)


        # Now calculate mean weights for coupling features grouped by cell type
        if coupling_indices is not None:
            pyr_indices = coupling_indices[:2]  # Adjust these indices as needed
            som_indices = coupling_indices[3:5]
            pv_indices = coupling_indices[6:9]

            # Calculate mean for each cell type
            mean_pyr = np.nanmean(mean_weights[pyr_indices, :], axis=0)
            mean_som = np.nanmean(mean_weights[som_indices, :], axis=0)
            mean_pv = np.nanmean(mean_weights[pv_indices, :], axis=0)

            # Calculate sums
            sum_pyr = np.sum(np.abs(mean_weights[pyr_indices, :]), axis=0)
            sum_som = np.sum(np.abs(mean_weights[som_indices, :]), axis=0)
            sum_pv = np.sum(np.abs(mean_weights[pv_indices, :]), axis=0)

            # Calculate max
            max_pyr = np.max(np.abs(mean_weights[pyr_indices, :]), axis=0)
            max_som = np.max(np.abs(mean_weights[som_indices, :]), axis=0)
            max_pv = np.max(np.abs(mean_weights[pv_indices, :]), axis=0)

            # Append these to the mean_neuron_feature_unique array
            mean_neuron_feature_unique[-3, :] = mean_pyr
            mean_neuron_feature_unique[-2, :] = mean_som
            mean_neuron_feature_unique[-1, :] = mean_pv

            total_beta_feature_unique[-3, :] = sum_pyr
            total_beta_feature_unique[-2, :] = sum_som
            total_beta_feature_unique[-1, :] = sum_pv

            max_beta_feature_unique[-3, :] = max_pyr
            max_beta_feature_unique[-2, :] = max_som
            max_beta_feature_unique[-1, :] = max_pv
        
            # Update unique_feature_names with cell type names
            updated_feature_names = unique_feature_names + ['pyr', 'som', 'pv']

            # Generate neuron labels (e.g., Neuron 1, Neuron 2, etc.)
            neuron_labels = [f'Neuron {i+1}' for i in range(num_neurons)]

            #get color palette
            palette = sns.color_palette("vlag", as_cmap=True)

            # Plot the heatmap
            plt.figure(figsize=(14, 8))
            sns.heatmap(mean_neuron_feature_unique, cmap=palette, cbar=True, yticklabels=updated_feature_names, vmin= minmax[0], vmax = minmax[1])
            plt.title('Mean Weights Across Folds')
            plt.xlabel('Neurons')
            plt.ylabel('Features')
            plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
            # Save the figure
            os.chdir(self.save_results)
            plt.savefig(f'heatmap_avg{no_abs}_uniquebeta_{animalID}_{date}_{model_type}.png')
            plt.show()
        else:
            updated_feature_names = []

        return mean_neuron_feature_unique,updated_feature_names,mean_weights,feature_names,total_beta_feature_unique,max_beta_feature_unique

    # # Example usage
    # mean_neuron_feature_unique,unique_feature_names,mean_weights = plot_weights_heatmap(animalID= animalID, date= date, model_type= model_type,
    #                      model_output=model_output_all, coupling_indices = coupling_predictors_indices, save_results=save_results, no_abs=0)

    def unique_features_heatmap_celltypes(self,mean_neuron_feature_unique,behav_features_unique,neuron_groups, minmax=(-.2,.2),model_type = None,no_abs=1,figsize = (20,8)):
        """
        Create a heatmap to show average across unique features for all neurons
        """
        fig, ax = plt.subplots(1,3, figsize = figsize)
        mpl.rcParams['pdf.fonttype'] = 42   # TrueType fonts (editable)

        #get color palette
        palette = sns.color_palette("vlag", as_cmap=True)#sns.cubehelix_palette(start=.9, rot=-.95, as_cmap=True)#'viridis'#sns.color_palette("Blues", as_cmap=True)#sns.cubehelix_palette(start=.5, rot=-.75, as_cmap=True)

        for i, (group, cel_indices) in enumerate(neuron_groups.items()):
            
            sns.heatmap(np.squeeze(mean_neuron_feature_unique[:,cel_indices]),vmin= minmax[0], vmax = minmax[1], cmap = palette , ax=ax[i], cbar=False) #model_output_all[0]['B_weights']
            ax[i].set_xlabel(group, fontsize=7)
            if i ==1 or i ==2: # delete y axis of middle and right graphs
                ax[i].set_yticks([])

        # Create colorbar for the last subplot
        cax = fig.add_axes(ax[-1])  # [left, bottom, width, height]
        cbar = plt.colorbar(ax[-1].collections[0])#, cax=cax

    
        # Graph title
        ax[1].set_title('Average Weights Across Features', fontsize=18)
            
        # Label x and y-axis
        ax[0].set_ylabel('Behavioral Features', fontsize=18)

        

        #set y labels
        # unique_feature_indices = {str(unique_f): idx for idx, unique_f in enumerate(behav_features_unique)}
        # Convert each element of the NumPy array to a regular Python string
        behav_features_unique_str = [str(label) for label in behav_features_unique]
        # Remove square brackets from the labels
        behav_features_unique_str = [label[1:-1] if label.startswith('[') and label.endswith(']') else label for label in behav_features_unique_str]
        
        ax[0].set_yticklabels(behav_features_unique)
        ax[0].tick_params(axis ='y', labelrotation =0)

        

        # Hide x-axis major ticks
        # ax.tick_params(axis='x', which='major', length=0)

        
        # # Clean up the appearance
        plt.tight_layout()
        os.chdir(self.save_results)
        plt.savefig(f'heatmap_avg{no_abs}_uniquebeta_celltypes_{model_type}.pdf', bbox_inches='tight')
        plt.show()

    # unique_features_heatmap_celltypes(mean_neuron_feature_unique,unique_feature_names,neuron_groups,save_results)



    def scatter_plot_weights_overlay(self,neuron_groups, mean_neuron_feature_unique, updated_feature_names, model_type,animalID = None, date = None,no_abs=1,minmax=(-.1,.8), figsize = (3,3)):
        """
        Create a scatter plot to show weights of unique features, overlaying for three cell types.
        
        Parameters:
        - neuron_groups: Dictionary with cell type as keys and corresponding neuron indices as values.
        - mean_neuron_feature_unique: 2D array with mean weights for unique features across neurons.
        - updated_feature_names: List of unique feature names.
        - celltypecolors: Dictionary with cell type colors.
        - save_results: Directory to save the plot.
        - animalID: Identifier for the animal.
        - date: Date of the experiment.
        - model_type: Type of model used.
        """

        # Set global font size and family 
        plt.rcParams.update({'font.size': 7, 'font.family': 'arial'})

        # Initialize the plot
        plt.figure(figsize=figsize)
        mpl.rcParams['pdf.fonttype'] = 42   # TrueType fonts (editable)

        # Loop through each cell type group and plot the corresponding weights
        for group, cell_indices in neuron_groups.items():
            # Extract the weights for the current cell type group
            group_weights = mean_neuron_feature_unique[:, cell_indices]
            
            # Calculate mean weights across neurons in the current group
            mean_group_weights = np.mean(group_weights, axis=1)

            # Calculate the standard error of the mean (SEM) across neurons
            sem_group_weights = np.std(group_weights, axis=1) / np.sqrt(group_weights.shape[1])

            # Flatten the SEM array to ensure it's 1D
            sem_group_weights = sem_group_weights.flatten()
            mean_group_weights = mean_group_weights.flatten()

            # Plot the weights as a scatter plot with error bars
            plt.errorbar(np.arange(mean_group_weights.shape[0]), mean_group_weights, yerr=sem_group_weights,
                        fmt='o', color='white', ecolor=self.celltypecolors[group], capsize=2, 
                        label=group, alpha=1, markersize=10, markeredgewidth=1, markeredgecolor=self.celltypecolors[group])
            
            # # Plot the weights as a scatter plot
            # plt.scatter(np.arange(mean_group_weights.shape[0]), mean_group_weights,color='white', 
            #             edgecolor=self.celltypecolors[group], label=group, alpha=1, s=200, linewidths= 3)
        
        # Add the unique feature names as x-tick labels
        plt.xticks(ticks=np.arange(len(updated_feature_names)), labels=updated_feature_names, rotation=90)
        
        plt.ylabel('Mean Weights')
        plt.legend(frameon = False)

        # # Clean up the appearance
        ax = plt.gca()
        ax.axhline(y=0, linestyle='dashed', color='k', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(minmax[0],minmax[1])
        ax.set_box_aspect(1)
        plt.xticks(fontsize = 6)
        # ax.set_aspect('equal', adjustable='box')
        # plt.axis('scaled')
        
        # Save the figure
        if animalID is not None:
            plt.savefig(f'{self.save_results}/scatter_overlay_weights_avg{no_abs}_{animalID}_{date}_{model_type}.pdf', bbox_inches='tight')
        else:
            plt.savefig(f'{self.save_results}/scatter_overlay_weights_avg{no_abs}_{animalID}_{date}_{model_type}.pdf', bbox_inches='tight')
        plt.show()



    def scatter_plot_weights_overlay_noerror(self,neuron_groups, mean_neuron_feature_unique, updated_feature_names, model_type,animalID = None, date = None,no_abs=1,minmax=(-.1,.8),save_string = None, figsize = (3,3)):
        """
        Create a scatter plot to show weights of unique features, overlaying for three cell types.
        
        Parameters:
        - neuron_groups: Dictionary with cell type as keys and corresponding neuron indices as values.
        - mean_neuron_feature_unique: 2D array with mean weights for unique features across neurons.
        - updated_feature_names: List of unique feature names.
        - celltypecolors: Dictionary with cell type colors.
        - save_results: Directory to save the plot.
        - animalID: Identifier for the animal.
        - date: Date of the experiment.
        - model_type: Type of model used.
        """

        # Set global font size and family 
        plt.rcParams.update({'font.size': 7, 'font.family': 'arial'})

        # Initialize the plot
        plt.figure(figsize= figsize)
        mpl.rcParams['pdf.fonttype'] = 42   # TrueType fonts (editable)

        # Loop through each cell type group and plot the corresponding weights
        for group, cell_indices in neuron_groups.items():
            # Extract the weights for the current cell type group
            group_weights = mean_neuron_feature_unique[:, cell_indices]
            
            # Calculate mean weights across neurons in the current group
            mean_group_weights = np.mean(group_weights, axis=1)

            # Calculate the standard error of the mean (SEM) across neurons
            sem_group_weights = np.std(group_weights, axis=1) / np.sqrt(group_weights.shape[1])

            # Flatten the SEM array to ensure it's 1D
            sem_group_weights = sem_group_weights.flatten()
            mean_group_weights = mean_group_weights.flatten()

            # Plot the weights as a scatter plot with error bars
            # Plot the weights as a scatter plot (hollow circles)
            plt.scatter(np.arange(mean_group_weights.shape[0]), mean_group_weights, 
                        edgecolor=self.celltypecolors[group], facecolors='none', 
                        label=group, alpha=1, s=20, linewidths= 1)
            
            # # Plot the weights as a scatter plot
            # plt.scatter(np.arange(mean_group_weights.shape[0]), mean_group_weights,color='white', 
            #             edgecolor=self.celltypecolors[group], label=group, alpha=1, s=200, linewidths= 3)
        
        # Add the unique feature names as x-tick labels
        plt.xticks(ticks=np.arange(len(updated_feature_names)), labels=updated_feature_names, rotation=90)
        
        plt.ylabel(r'|$\beta$ Weights|')
        #plt.legend(frameon = False)

        # # Clean up the appearance
        ax = plt.gca()
        ax.axhline(y=0, linestyle='dashed', color='k', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylim(minmax[0],minmax[1])
        plt.xticks(fontsize = 6)
        ax.set_box_aspect(1)
        # ax.set_aspect('equal', adjustable='box')
        # plt.axis('scaled')

        #to save svg so that we can edit texts!
        new_rc_params = {'text.usetex': False,
        "svg.fonttype": 'none'
        }
        plt.rcParams.update(new_rc_params)
        
        # Save the figure
        if animalID is not None:
            if save_string is not None:
                plt.ylabel(fr'{save_string} |$\beta$ Weights|')
                plt.savefig(f'{self.save_results}/scatter_overlay_updatedweights_avg{no_abs}_{animalID}_{date}_{model_type}_{save_string}.pdf', bbox_inches='tight')
            else:
                plt.savefig(f'{self.save_results}/scatter_overlay_updatedweights_avg{no_abs}_{animalID}_{date}_{model_type}.pdf', bbox_inches='tight')
        else:
            if save_string is not None:
                plt.ylabel(fr'{save_string} |$\beta$ Weights|')
                plt.savefig(f'{self.save_results}/scatter_overlay_updatedweights_avg{no_abs}_{model_type}_{save_string}.pdf', bbox_inches='tight')
            else:
                plt.savefig(f'{self.save_results}/scatter_overlay_updatedweights_avg{no_abs}_{model_type}.pdf', bbox_inches='tight')
        plt.show()

    # # Example usage
    # scatter_plot_weights_overlay(neuron_groups=neuron_groups, 
    #                              mean_neuron_feature_unique=mean_neuron_feature_unique, 
    #                              updated_feature_names=unique_feature_names, 
    #                              self.celltypecolors=celltypecolors, 
    #                              save_results=save_results,
    #                              animalID=animalID, 
    #                              date=date, 
    #                              model_type=model_type)

    def specified_features_heatmap(self,mean_neuron_feature_unique,specified_features,behav_features_ids,minmax=(-.2,.2)):
        """
        Create a heatmap to show average across unique features for all neurons
        """
        fig, ax = plt.subplots(1,1, figsize = (12,8))
        mpl.rcParams['pdf.fonttype'] = 42   # TrueType fonts (editable)
        colors = sns.color_palette("vlag", as_cmap=True)
        sns.heatmap(np.squeeze(mean_neuron_feature_unique[specified_features,:]),vmin= minmax[0], vmax= minmax[1], cmap = colors) #model_output_all[0]['B_weights']

        # Graph title
        ax.set_title('Average Weights Across Features', fontsize=7)
            
        # Label x and y-axis
        ax.set_ylabel('Behavioral Features', fontsize=7)
        ax.set_xlabel('Neurons', fontsize=7)

        #set y labels
        # unique_feature_indices = {str(unique_f): idx for idx, unique_f in enumerate(behav_features_unique)}
        # Convert each element of the NumPy array to a regular Python string
        behav_features_unique_str = [str(label) for label in behav_features_ids]
        # Remove square brackets from the labels
        behav_features_unique_str = [label[1:-1] if label.startswith('[') and label.endswith(']') else label for label in behav_features_unique_str]
        ax.set_yticklabels( behav_features_unique_str)
        ax.tick_params(axis ='y', labelrotation =0)

        # Label x-axis ticks
        # ax.set_xticklabels(neuron_groups.keys(), fontsize=7)

        # Hide x-axis major ticks
        ax.tick_params(axis='x', which='major', length=0)

        
        # # Clean up the appearance
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_box_aspect(1)


    def scatter_model_dev_comparison(self,mean_deviance1, mean_deviance2, cell_ids, measure_string, measure_string2, colors=None, plot_lims= None, save_path = None, plot_lims_neg = None, minmax = None):
        """
        Make scatter plot comparing deviance explained of partial vs full models.
        
        Parameters:
            mean_deviance1 (array_like): Deviance explained for full model.
            mean_deviance2 (array_like): Deviance explained for partial model.
            cell_ids (array_like): IDs of cells.
            measure_string (str) The label for the axis and title
            colors (dict, optional): Dictionary of colors for each cell type. Keys should be cell types and values should be colors.
            plot_lims (optional): limits for x and y axis
            save_path(optional): path to save plot

        Returns:
            None
        """
        # Set global font size and family 
        plt.rcParams.update({'font.size': 7, 'font.family': 'arial'})
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        mpl.rcParams['pdf.fonttype'] = 42   # TrueType fonts (editable)
        
        # Get unique cell types from cell_ids
        cell_types = np.unique(cell_ids)

        
        # Define colors for different cell types (non sequential)
        #color_dict = {cell_type: plt.cm.Dark2(i) for i, cell_type in enumerate(cell_types)} #Dark2

        if colors is None:
            # Use a sequential colormap if colors are not provided
            colormap = plt.cm.plasma
            # Define colors for different cell types by evenly spacing them in the colormap
            color_dict = {cell_type: colormap(i / len(cell_types)) for i, cell_type in enumerate(cell_types)}
        else:
            color_dict = self.celltypecolors

        
        # Assign colors to cell types
        colors = [color_dict[cell_type] for cell_type in cell_ids]

        # Draw unity line
        max_dev = max(max(mean_deviance2), max(mean_deviance1))
        if minmax is not None:
            max_dev = minmax[1]
            min_dev = minmax[0]
            plt.plot([min_dev, max_dev], [min_dev, max_dev], '--', color='black', zorder = 1, label='Unity Line')
        else:
            plt.plot([0, max_dev], [0, max_dev], '--', color='black', zorder = 1, label='Unity Line')


        # Create scatter plot
        # plt.scatter(mean_deviance2, mean_deviance1,linewidths=2, edgecolors=colors,facecolors = 'None', alpha=0.7, zorder = 2)# c = colors
        plt.scatter(mean_deviance2, mean_deviance1,c = colors, alpha=0.7, zorder = 2)# c = colors


        
        # Set labels and title
        plt.xlabel(f'{measure_string}', fontsize=7)
        plt.ylabel(f'{measure_string2}', fontsize=7)
        plt.title(f'{measure_string} vs {measure_string2}', fontsize=7)

        # Set equal limits for x and y axes
        if colors is None:
            max_lim = max(max_dev, 1.0)
        else:
            max_lim = plot_lims
        plt.xlim(0, max_lim)
        plt.ylim(0, max_lim)
        if plot_lims_neg is not None:
            plt.xlim(plot_lims_neg, max_lim)
            plt.ylim(plot_lims_neg, max_lim)

        # Add legend

        # Create legend elements with invisible markers and colored labels
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='none', label=self.cell_type_labels.values(), markerfacecolor='none', linestyle='None')
            for cell_type in color_dict.keys()
        ]

        # Add the legend to the plot
        legend = plt.legend(handles=legend_elements, frameon=False, handlelength=0, handletextpad=0.1)

        # Set the color of the legend text to match the corresponding cell type
        for text, color in zip(legend.get_texts(), color_dict.values()):
            text.set_color(color)

        # legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=cell_type, markerfacecolor=color)
        #                    for cell_type, color in color_dict.items()]
        # plt.legend(handles=legend_elements, frameon = False,handlelength=1, handletextpad=0.1)

        # # Clean up the appearance
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_box_aspect(1)
        
        #to save svg so that we can edit texts!
        new_rc_params = {'text.usetex': False,
        "svg.fonttype": 'none'
        }
        plt.rcParams.update(new_rc_params)

        # Save plot if save_path is provided
        if save_path: 
            plt.savefig(save_path, bbox_inches='tight', format = 'pdf')
        
        # Show plot
        plt.show()  


    def box_plot(self, data, neuron_groups, colors, measure_string, save_path = None, figsize = (3,3)): #plotting function for box plots to compare fraction deviance across celltypes
        """
        Create a box-and-whisker plot with significance bars.
        """
        # Set global font size and family 
        plt.rcParams.update({'font.size': 7, 'font.family': 'arial'})

        fig, ax = plt.subplots(1,1, figsize = figsize)
        mpl.rcParams['pdf.fonttype'] = 42   # TrueType fonts (editable)
        #ax = plt.axes()

        #Calculate positions for each cell type group along the x-axis
        positions = np.arange(1,len(neuron_groups)+1)
        for i, (group, cel_indices) in enumerate(neuron_groups.items()):
            flat_indices = np.ravel(cel_indices)  # Flatten the indices to 1D array
            
            #put the boxplot in the correct position
            position = positions[i]

            # Prepare data for the current group, filtering out NaNs
            filtered_data = [data[idx] for idx in flat_indices if not np.isnan(data[idx])]

            # Plot the boxplot for the current cell type group at the correct position
            bp = ax.boxplot(filtered_data, positions=[position], widths=.1, showfliers=True, patch_artist=True)

            # #plot the boxplot for the current cell type group at the correct position
            # bp = ax.boxplot([data[idx] for idx in flat_indices], positions = [position], widths= .1, showfliers = True, patch_artist=True)
            # #used to be bp = ax.boxplot(data[cel_indices], positions = [position], widths= .1, showfliers = True, patch_artist=True)
            

            for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']: #'boxes',
                plt.setp(bp[element], color = colors[group])

            for patch in bp['boxes']:
                patch.set(facecolor= (1,1,1)) 

            # Colour of the median lines
            #plt.setp(bp['medians'], color='k')
                
            #set cell type label
            ax.set_label (group)


        # for patch, c in zip(bp['boxes'],colors.values()):
        #     print(c)
        #     patch.set_facecolor(c)
            
        # for patch in bp['boxes']:
        #     patch.set_facecolor(colors[group])
        
        # Graph title
        ax.set_title(f'{measure_string} Across Cell types', fontsize=7)
            
        # Label x and y-axis
        ax.set_ylabel(f'{measure_string}', fontsize=7)
        ax.set_xlabel('Cell type', fontsize=7)

        # Label x-axis ticks
        ax.set_xticks(positions)
        ax.set_xticklabels(self.cell_type_labels.values(), fontsize=7) #neuron_groups.keys()

        # Hide x-axis major ticks
        ax.tick_params(axis='x', which='major', length=0)

        
        # # Clean up the appearance
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Save plot if save_path is provided
        if save_path: 
            plt.savefig(save_path, bbox_inches='tight')

        #find significant changes between groups
        # add_significance_stars(ax, data, neuron_groups, alpha=0.05)

    def violin_plot(self, data, neuron_groups, colors, measure_string, save_path=None):
        """
        Create a violin plot to compare distribution across cell types.
        
        Parameters:
        -----------
        data : array-like
            Data to plot
        neuron_groups : dict
            Dictionary containing indices for each cell type
        colors : dict
            Dictionary of colors for each cell type
        measure_string : str
            Label for y-axis
        save_path : str, optional
            Path to save the plot
        """
        plt.rcParams.update({'font.size': 7, 'font.family': 'arial'})
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        
        # Prepare data for violin plot
        plot_data = []
        labels = []
        color_list = []
        
        positions = np.arange(1, len(neuron_groups) + 1)
        for i, (group, cel_indices) in enumerate(neuron_groups.items()):
            flat_indices = np.ravel(cel_indices)
            filtered_data = [data[idx] for idx in flat_indices if not np.isnan(data[idx])]
            plot_data.append(filtered_data)
            labels.append(self.cell_type_labels[group])
            color_list.append(colors[group])
        
        # Create violin plot
        parts = ax.violinplot(plot_data, positions=positions, showmeans=True)
        
        # Customize violin plot appearance
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor('none')
            pc.set_edgecolor(color_list[i])
            pc.set_linewidth(2)
        
        # Customize mean lines
        parts['cmeans'].set_color('black')
        parts['cmeans'].set_linewidth(2)
        
        # Set labels and title
        ax.set_ylabel(measure_string)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        
        # Clean up appearance
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        
        plt.show()
        return ax

    def bar_plot(self, data, neuron_groups, colors, measure_string, bar_width=0.5, ylims = 0.5, save_path = None):
        """
        Create a bar plot with error bars to compare fraction deviance across cell types.
        Parameters:
        data: array-like
            The data to be plotted.
        neuron_groups: dict
            A dictionary where keys are group names and values are indices of neurons in those groups.
        colors: dict
            A dictionary where keys are group names and values are colors for the bars.
        measure_string: str
            The label for the y-axis.
        bar_width: float
            The width of the bars in the plot.
        """
        # Set global font size and family 
        plt.rcParams.update({'font.size': 7, 'font.family': 'arial'})
        mpl.rcParams['pdf.fonttype'] = 42   # TrueType fonts (editable)
        fig, ax = plt.subplots(1, 1, figsize=(2, 2))
        # Calculate positions for each cell type group along the x-axis
        positions = np.arange(len(neuron_groups)) + 1
        means = []
        errors = []
        color_list = []
        labels = []

        for group, indices in neuron_groups.items():
            flat_indices = np.ravel(indices)  # Flatten the indices to 1D array
            # Filter out NaN values from group data
            group_data = [data[idx] for idx in flat_indices if not np.isnan(data[idx])]

            # group_data = [data[idx] for idx in flat_indices]
            means.append(np.mean(group_data))
            errors.append(np.std(group_data) / np.sqrt(len(group_data)))  # Standard error
            color_list.append(colors[group])
            labels.append(self.cell_type_labels[group])

        means = np.array(means)
        errors = np.array(errors)
        # Create bar plot with uncolored inside and colored outlines
        # bars = ax.bar(positions, means, yerr=errors, capsize=5, edgecolor=[colors[group] for group in neuron_groups],
        #             facecolor='white', linewidth=2, width=bar_width, ecolor=[colors[group] for group in neuron_groups])#, ecolor=[colors[group] for group in neuron_groups]
        # Plot each bar separately with its own color
        for i, (group, mean, error) in enumerate(zip(neuron_groups, means, errors)):
            ax.bar(positions[i], mean,
                yerr=error,
                capsize=5,
                color='white',
                edgecolor=colors[group],
                linewidth=2,
                width=bar_width,
                error_kw={'ecolor': colors[group]})
    
        # Set labels and title
        ax.set_title(f'{measure_string} Across Cell types', fontsize=7)
        ax.set_ylabel(f'{measure_string}', fontsize=7)
        ax.set_xlabel('Cell type', fontsize=7)
        ax.set_xticks(positions)
        ax.set_xticklabels(self.cell_type_labels.values(), fontsize=7) #neuron_groups.keys()
        ax.tick_params(axis='x', which='major', length=0)

        if ylims:
            ax.set_ylim(0, ylims)

        # # Clean up the appearance
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Save plot if save_path is provided
        if save_path: 
            plt.savefig(save_path, bbox_inches='tight')
        
        plt.show()

    def histogram_model_dev_comparison(self, mean_deviance1, mean_deviance2, cell_ids, colors=None, save_path = None, xlims = None, bin_width = None, outlier_clip=None):
        """
        Make scatter plot comparing deviance explained of partial vs full models.
        
        Parameters:
            mean_deviance1 (array_like): Deviance explained for full model.
            mean_deviance2 (array_like): Deviance explained for partial model.
            cell_ids (array_like): IDs of cells.
            colors (dict, optional): Dictionary of colors for each cell type. Keys should be cell types and values should be colors.
            xlims (tuple, optional): Limits for the x-axis.
            bin_width (float, optional): The width of the bins in the histogram.
            outlier_clip (tuple, optional): Clip data to this range to ignore outliers.

        Returns:
            None
        """
        # Set global font size and family 
        plt.rcParams.update({'font.size': 7, 'font.family': 'arial'})
        mpl.rcParams['pdf.fonttype'] = 42   # TrueType fonts (editable)
        # Get unique cell types from cell_ids
        cell_types = np.unique(cell_ids)

        if colors is None:
        #     # Use a sequential colormap if colors are not provided
        #     colormap = plt.cm.plasma
        #     # Define colors for different cell types by evenly spacing them in the colormap
        #     color_dict = {cell_type: colormap(i / len(cell_types)) for i, cell_type in enumerate(cell_types)}
        # else:
            color_dict = self.celltypecolors
            # color_dict = {0: (0.37, 0.75, 0.49),   # pyr = 0
            #         1: (0.17, 0.35, 0.8),    # som = 1
            #         2: (0.82, 0.04, 0.04)} 

        # Create subplots for histograms
        fig, axes = plt.subplots(len(cell_types), 1, figsize=(4, 3 * len(cell_types)))
        
        # Iterate over each cell type
        for ax, cell_type in zip(axes, cell_types):
            # Select data corresponding to the current cell type
            idx = np.where(cell_ids == cell_type)[0]
            deviance_diff = mean_deviance1[idx] - mean_deviance2[idx]
            
            # Clip outliers if outlier_clip is provided
            if outlier_clip:
                deviance_diff = np.clip(deviance_diff, outlier_clip[0], outlier_clip[1])

            # Calculate the range for the bins based on the data or clipping range
            if outlier_clip:
                data_range = outlier_clip
            else:
                data_range = (min(deviance_diff), max(deviance_diff))

                # Plot histogram

            # Calculate the number of bins based on the bin width
            if bin_width:
                num_bins = int((data_range[1] - data_range[0]) / bin_width)
                num_bins = min(num_bins, 30)
            else:
                num_bins = 30

            ax.hist(deviance_diff, bins=num_bins, range=data_range , color=color_dict[cell_type], alpha= 1) #bins=20

            # ax.hist(deviance_diff, color=color_dict[cell_type], alpha= 1) #bins=20

            ax.axvline(x = 0, linestyle = 'dashed', color = 'k', alpha = 1)
            #ax.set_title(f'Cell Type: {cell_type}')
            ax.set_xlabel('Difference in Deviance Explained', fontsize=7)
            ax.set_ylabel('Frequency', fontsize=7)
            # # Clean up the appearance
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if xlims:
                ax.set_xlim(xlims)
            
        plt.tight_layout()

        # Save plot if save_path is provided
        if save_path: 
            plt.savefig(save_path, bbox_inches='tight')
        
        plt.show()

    def bar_plot_separated(self, data, neuron_groups,  measure_string, bar_width=0.5, save_path=None, minmax = (0,.5)):
        """
        Create a bar plot with error bars to compare fraction deviance across cell types.
        
        Parameters:
        data: dict
            A dictionary where keys are model comparisons and values are the data to be plotted.
        neuron_groups: dict
            A dictionary where keys are cell types and values are indices of neurons in those groups.
        colors: dict
            A dictionary where keys are group names and values are colors for the bars.
        measure_string: str
            The label for the y-axis.
        bar_width: float
            The width of the bars in the plot.
        save_path: str, optional
            The path to save the plot. If None, the plot is displayed instead of saved.
        """
        #set colors
        colors = self.celltypecolors
        # Set global font size and family 
        plt.rcParams.update({'font.size': 7, 'font.family': 'arial'})
        mpl.rcParams['pdf.fonttype'] = 42   # TrueType fonts (editable)

        # Create a figure with 3 subplots (one for each cell type)
        fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
        
        # Iterate over neuron groups to create each subplot
        for i, (cell_type, indices) in enumerate(neuron_groups.items()):
            
            ax = axs[i]  # Select the subplot
            
            # Calculate positions for each model comparison along the x-axis
            positions = np.arange(len(data)) + 1
            means = []
            errors = []
            
            # Iterate over each model comparison and compute means and errors
            for comparison, comp_data in data.items():
                group_data = np.array(comp_data)[indices]
                # means.append(np.mean(group_data))
                # errors.append(np.std(group_data) / np.sqrt(len(group_data)))  # Standard error
                # Ignore NaN values in the calculation
                means.append(np.nanmean(group_data))
                errors.append(np.nanstd(group_data) / np.sqrt(np.sum(~np.isnan(group_data))))  # Standard error ignoring NaNs
            
            # Create bar plot with uncolored inside and colored outlines
            bars = ax.bar(positions, means, yerr=errors, capsize=4, 
                        edgecolor= colors[cell_type],
                        facecolor='white', linewidth=2, width=bar_width, ecolor=colors[cell_type])

            # Set labels and title for each subplot
            ax.set_title(f'{cell_type}', fontsize=7)
            ax.set_xticks(positions)
            ax.set_xticklabels(data.keys(), rotation=45, ha='right', fontsize=7)
            ax.tick_params(axis='x', which='major', length=0)

            # Clean up the appearance
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if i == 0:
                ax.set_ylabel(f'{measure_string}', fontsize=7)

            ax.set_ylim(minmax[0],minmax[1])
        
        # Add a global title for the figure
        fig.suptitle(f'{measure_string} Across Cell Types and Models', fontsize=7)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        #plt.ylim(0,10e30)

        # Save plot if save_path is provided
        if save_path: 
            plt.savefig(save_path, bbox_inches='tight')
        
        plt.show()

    def plot_cdf_coupling_index(self, coupling_index, cell_ids, colors,title, save_path=None,xlabel = 'Coupling Index', xval = 1.1, xint = 0.2, perform_stats=False):
        """
        Create a CDF plot for coupling index across different cell types.

        Parameters:
            coupling_index (array_like): Coupling index values.
            cell_ids (array_like): IDs of cells, which determine the cell type.
            colors (dict): Dictionary of colors for each cell type. Keys are cell types and values are colors.
            save_path (str, optional): Path to save the plot.
            
        Returns:
            None
        """
        #convert to numpy array!!
        cell_ids = np.array(cell_ids)
        # Set global font size and family 
        plt.rcParams.update({'font.size': 7, 'font.family': 'arial'})

        plt.figure(figsize=(3, 3))

        # Store data by cell type for statistical testing
        data_by_celltype = {}

        # Create a mapping from cell labels to colors
        color_map = {label: colors[label] for label in np.unique(cell_ids)}
        
        # Map cell labels to colors
        color_values = [color_map[label] for label in cell_ids]

        cell_types = np.unique(cell_ids)
        
        # Plot CDF for each cell type
        for cell_type in cell_types:
            # Get indices for the current cell type
            indices = np.where(cell_ids == cell_type)[0]
            type_data = coupling_index[indices]
            data_by_celltype[cell_type] = type_data
                
            # Get coupling index values for the current cell type
            type_coupling_index = coupling_index[indices]
            
            # # Calculate and plot the CDF
            # sorted_coupling_index = np.sort(type_coupling_index)
            # cdf = np.arange(1, len(sorted_coupling_index) + 1) / len(sorted_coupling_index)
            
            # plt.plot(sorted_coupling_index, cdf, label=cell_type, color='red') #colors[cell_type]

            # Calculate CDF
            # x1 = np.linspace(np.min(type_coupling_index), np.max(type_coupling_index), 100)  # Define range of x values
            x1 = np.linspace(0, 1, 100)  # Define range of x values
            n1, _ = np.histogram(type_coupling_index, bins=x1)  # Histogram counts
            p1 = n1 / np.sum(n1)  # Probability
            cdf = np.cumsum(p1)  # Cumulative sum to get CDF
            
            plt.plot(x1[:-1], cdf, label=self.cell_type_labels[cell_type], linewidth= 2, color=color_map[cell_type])  # x1[:-1] because histogram bins include right edge
        
        

        # # Get axis
        ax = plt.gca()
        
        plt.xlabel(xlabel)
        plt.ylabel('Cumulative Fraction')

        # Define the ticks you want (e.g., from 0 to 1 with increments of 0.1)
        ticks = np.arange(0, xval, xint)  # The 1.1 ensures that 1.0 is included in the ticks

        # Set the format for both x and y axis ticks to show one decimal place
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        # Ensure ticks are from 0 to 1 with consistent intervals
        plt.xticks(np.arange(0, xval+(xval/10), xint))
        plt.yticks(np.arange(0, 1.1, 0.2))

        plt.title(title)
        # Create legend with just colored text
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='none', label=self.cell_type_labels[cell_type], 
                    markerfacecolor='none', linestyle='None')
            for cell_type in color_map.keys()
        ]
        legend = plt.legend(handles=legend_elements, frameon=False, 
                        handlelength=0, handletextpad=0.1)
        for text, color in zip(legend.get_texts(), color_map.values()):
            text.set_color(color)

        # Perform statistical comparisons if requested
        if perform_stats:
            print("\nStatistical Comparisons:")
            comparisons = list(itertools.combinations(cell_types, 2))
            for type1, type2 in comparisons:
                data1 = data_by_celltype[type1]
                data2 = data_by_celltype[type2]
                
                # Kolmogorov-Smirnov test for distribution differences
                ks_stat, p_value = scipy.stats.ks_2samp(data1, data2)
                print(f"{type1} vs {type2}:")
                print(f"KS statistic: {ks_stat:.4f}, p-value: {p_value:.4f}")
                
                # Mann-Whitney U test for median differences
                u_stat, p_value = scipy.stats.mannwhitneyu(data1, data2, alternative='two-sided')
                print(f"Mann-Whitney U statistic: {u_stat:.4f}, p-value: {p_value:.4f}\n")

        # plt.legend(frameon = False)
        # plt.axis('equal')
        
        # # Clean up the appearance
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_box_aspect(1)
        ax.set_xlim(0,xval+0.01)
        ax.set_ylim(0,1.01)

        #to save svg so that we can edit texts!
        new_rc_params = {'text.usetex': False,
        "svg.fonttype": 'none'
        }
        plt.rcParams.update(new_rc_params)

        # Save or show plot
        if save_path:
            plt.savefig(save_path, format = 'svg')
            plt.savefig(save_path.replace('.svg','.pdf'), format = 'pdf')
        else:
            plt.show()


    def plot_decoding_results(self,mean_results_all, decoder_type='choice', plot_type='pop', save_dir=None, xlim = None, ylim = None):
        """
        Plot mean decoding results across datasets.
        
        Args:
            mean_results_all (dict): Dictionary of mean results across datasets
            decoder_type (str): Type of decoding ('choice', 'sound_category', etc.)
            plot_type (str): 'pop' or 'sc' for population or single cell
            save_dir (str): Directory to save plots (optional)
        """
        # Set global font size and family 
        plt.rcParams.update({'font.size': 7, 'font.family': 'arial'})
        
        # Metrics to plot
        metrics = [
            f'{plot_type}_instantaneous_information_mean',
            f'{plot_type}_cumulative_information_mean',
            f'{plot_type}_instantaneous_fraction_correct_mean',
            f'{plot_type}_cumulative_fraction_correct_mean'
        ]

        # Get event frames from first dataset
        first_dataset = list(mean_results_all.keys())[0]
        event_frames = mean_results_all[first_dataset][decoder_type]['event_frame_mean']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 7))
        axes = axes.flat
        
        for idx, metric in enumerate(metrics):
            # Collect data across datasets
            all_data = []
            for dataset in mean_results_all.keys():
                if decoder_type in mean_results_all[dataset]:
                    data = mean_results_all[dataset][decoder_type][metric]

                    # Average across neurons for sc data
                    if plot_type == 'sc' and len(data.shape) == 2:  # frames x neurons
                        data = np.mean(data, axis=1)  # average across neurons

                    all_data.append(data)
            
            # Calculate mean and SEM across datasets
            all_data = np.array(all_data)
            mean_trace = np.mean(all_data, axis=0)
            sem_trace = np.std(all_data, axis=0) / np.sqrt(len(all_data))
            
            # Plot
            ax = axes[idx]
            x = np.arange(len(mean_trace))
            ax.plot(mean_trace, 'k-', label='Mean')
            ax.fill_between(x, mean_trace-sem_trace, mean_trace+sem_trace, 
                        alpha=0.3, color='gray', label='SEM')
            
            # Add event markers
            if xlim is None:
                xlim = (0, len(mean_trace)) # full trace
            for frame in event_frames:
                if frame < xlim[1]:
                    ax.axvline(x=frame, color='k', linestyle=':', alpha=0.5)
            
            # Formatting
            ax.set_title(metric.replace('_', ' ').title())
            ax.set_xlabel('Frames')
            if 'information' in metric:
                ax.set_ylabel('Bits')
            else:
                ax.set_ylabel('Fraction Correct')

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            ax.set_box_aspect(1)
            if xlim:
                ax.set_xlim(xlim)

            if ylim:
                ax.set_ylim(0,ylim[idx])
            
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, f'{decoder_type}_{plot_type}_decoding.png'))
        
        plt.show()




    def plot_decoding_heatmap_datasets(self,results_dict, decoder_type='sound_category', metric = 'pop_instantaneous_information_mean'):
        """Create suite of analysis plots"""
        
        # # 1. Time series plot with mean±SEM across datasets
        # plt.figure(figsize=(10,6))
        # all_data = []
        # for dataset in results_dict:
        #     data = results_dict[dataset][decoder_type]['pop_cumulative_information_mean']
        #     all_data.append(data)
        
        # mean_trace = np.mean(all_data, axis=0)
        # sem_trace = np.std(all_data, axis=0) / np.sqrt(len(all_data))
        # plt.plot(mean_trace, 'b-', label='Mean')
        # plt.fill_between(range(len(mean_trace)), 
        #                 mean_trace-sem_trace, 
        #                 mean_trace+sem_trace,
        #                 alpha=0.3)
        # plt.title(f'{decoder_type} Decoding Performance')
        # plt.xlabel('Time (frames)')
        # plt.ylabel('Information (bits)')
        
        # # 2. Compare instantaneous vs cumulative
        # # plt.figure(figsize=(10,6))
        # # for metric in ['pop_instantaneous_information_mean', 'pop_cumulative_information_mean']:
        # #     all_data = []
        # #     for dataset in results_dict:
        # #         data = results_dict[dataset][decoder_type][metric]
        # #         all_data.append(data)
        # #     mean_trace = np.mean(all_data, axis=0)
        # #     plt.plot(mean_trace, label=metric.split('_')[1])
        # # plt.legend()
        # # plt.title('Instantaneous vs Cumulative Information')
        
        # # 3. Peak information by cell type boxplot
        # peaks_by_celltype = analyze_peaks_by_celltype( results_dict, decoder_type=decoder_type, start_frame=14, end_frame = 100)
        # all_peaks, neuron_groups = format_peaks_for_boxplot(peaks_by_celltype)
        # self.box_plot(all_peaks, 
        #                 neuron_groups,
        #                 self.celltypecolors,
        #                 'Peak Information')
        
        # 4. Information timeline heatmap
        plt.figure(figsize=(12,len(results_dict)))
        data_matrix = np.array([results_dict[d][decoder_type][metric] 
                            for d in results_dict])
        sns.heatmap(data_matrix, 
                    xticklabels=20, 
                    yticklabels=list(results_dict.keys()),
                    cmap='viridis')
        plt.title('Information Evolution Across Datasets')
        # plt.xlabel('Time (frames)')
        
        plt.tight_layout()
        event_onset = self.get_event_frame_for_decoder(decoder_type)
        xticks_in, xticks_lab = self.x_axis_sec_aligned(event_onset- 0, data_matrix.shape[1], interval=10, frame_rate=30)

        plt.xticks(ticks=xticks_in, labels=xticks_lab)
        plt.xlabel('Time (s)')
        plt.show()

    def plot_neuron_performance_heatmap(self, results_dict, decoder_type, start_frame=14, end_frame=None, metric='sc_cumulative_information_mean', significance_struc=None):
        """Plot heatmap of neuron performance, using only significant neurons if they exist and sorting by peak time."""
        plt.figure(figsize=(12, len(results_dict) * 2))
        
        for dataset in results_dict:
            data = results_dict[dataset][decoder_type][metric]
            if end_frame is None:
                end_frame = len(data)
            
            # Find the peak frame for each neuron (the frame where the maximum value occurs)
            peak_frames = np.argmax(data[start_frame:end_frame, :], axis=0) + start_frame  # +start_frame to account for the offset
            max_info = np.max(data[start_frame:end_frame, :], axis=0)

            # Sort neurons first by their peak frame, then by the maximum information value
            sort_idx = np.argsort(peak_frames)  # Sort by peak frame
            # If peak frames are identical, sort by maximum information
            sort_idx = sort_idx[np.argsort(max_info[sort_idx])][::-1] #[::-1] to reverse it

            # Use significant neurons if provided
            if significance_struc is not None and dataset in significance_struc:
                sig_neurons = significance_struc[dataset]['sig_neurons_all']
                data = data[:, sig_neurons]

            plt.subplot(len(results_dict), 1, list(results_dict.keys()).index(dataset) + 1)
            sns.heatmap(data[:, sort_idx].T, cmap='viridis', xticklabels=20, yticklabels=False)
            plt.title(f'{dataset} Neuron Performance')

        plt.tight_layout()
        # xticks_in, xticks_lab = self.x_axis_sec_aligned(self.event_frames[0], len(x), interval=1, frame_rate=30) 
        # Set x-ticks to be in seconds
        # for ax in plt.gcf().axes:
        #     # Ensure we only set x-ticks for the last plot
        #     if ax == plt.gcf().axes[-1]:
        #         plt.xticks(ticks=xticks_in, labels=xticks_lab)
        #         plt.xlabel('Time (s)')
        #     else:
        #         ax.set_xticks([])


        # Adding labels for event frames if provided
        axes = plt.gca()  # get current axis  
        if self.event_frames is not None:
            
            xlim = axes.get_xlim()
            for frame in self.event_frames:
                if frame < xlim[1]:
                    axes.axvline(x=frame, color='w', linestyle=(0, (10.5,6.8)), alpha=1,lw=0.7)
            axes.set_xticks(self.event_frames)
            axes.set_xticklabels(self.event_labels)
            plt.xticks(rotation=45)
       
        plt.show()

    def plot_summary_heatmap(self, results_dict, decoder_type, start_frame=14, end_frame=None, metric='sc_cumulative_information_mean', significance_struc=None, save_path = None):
        """Plot a summary heatmap combining all datasets, normalized by each neuron's maximum value."""
        combined_data = []
        if end_frame is None:
                end_frame = len(data)

        overall_size = round(end_frame - start_frame)

        # Determine plot size based on overall_size
        if overall_size < 100:
            figsize = (3, 3)
        elif 100 <= overall_size < 200:
            figsize = (4, 3)
        elif 200 <= overall_size < 300:
            figsize = (5, 3)
        else:
            figsize = (6, 3)

        plt.figure(figsize=figsize)

        for dataset in results_dict:
            data = results_dict[dataset][decoder_type][metric] #data is frames x neurons
            event_frames = results_dict[dataset][decoder_type]['event_frame_mean']
            

            # Use significant neurons if provided
            if significance_struc is not None and dataset in significance_struc:
                sig_neurons = significance_struc[dataset]['sig_neurons_all']
                data = data[:, sig_neurons]

            # Normalize each neuron by its maximum value
            data = data[start_frame:end_frame, :]
            max_values = np.max(data, axis=0) #gives neurons
            normalized_data = data / max_values  # Normalize by Imax for each neuron
            combined_data.append(normalized_data)

        # Combine all datasets along the neuron axis
        if combined_data:
            combined_data = np.concatenate(combined_data, axis=1)
        
            # Sort neurons by their peak frame across all datasets
            peak_frames = np.argmax(combined_data, axis=0)
            sort_idx = np.argsort(peak_frames)[::-1] #[::-1] to reverse it
            combined_data = combined_data[:, sort_idx]

            # Convert x-axis from frames to seconds
            num_frames = data.shape[0]
            time_in_seconds = np.arange(num_frames) / 30.0
            # Use only full seconds for x-axis labels
            full_seconds = np.arange(0, num_frames // 30 + 1)
            full_seconds_labels = full_seconds * 30  # Convert back to frame indices for labeling

            # print(f'full sec label: {full_seconds_labels}, time in sec: {time_in_seconds}')
            
            #print(round(int(np.floor(np.shape(combined_data)[1]*.25)),0))
            print(f'Sig neurons total {np.shape(combined_data)[1]}')

            # Plot the summary heatmap
            sns.heatmap(combined_data.T, cmap='viridis', xticklabels=30, yticklabels= 100) #int(np.floor(np.shape(combined_data)[1]*.2))
            
            #plt.title('Summary Neuron Performance (Normalized)')
            plt.xlabel('Seconds')
            plt.ylabel('Neurons')
            plt.tight_layout()
            # Clean up the appearance
            ax = plt.gca()  # get current axis  
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            ax.set_xticklabels(full_seconds, rotation=0)
            #ax.set_xticklabels(rotation=0)
            # Add event markers
            
            xlim = (0, np.shape(combined_data)[0]) # full trace
            for frame in event_frames:
                if frame < xlim[1]:
                    ax.axvline(x=frame, color='w', linestyle=':', alpha=1)
            
        else:
            print("No data to plot in the summary heatmap.")

        # Save plot if save_path is provided
        if save_path: 
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()



    # def plot_neuron_performance_heatmap(self, results_dict, decoder_type, start_frame = 14, end_frame = None, metric = 'sc_cumulative_information_mean', significant_neurons=None):
    #     """Plot heatmap of neuron performance."""
    #     plt.figure(figsize=(12,len(results_dict)*2))
    #     for dataset in results_dict:
    #         data = results_dict[dataset][decoder_type][metric]
    #         if end_frame is None:
    #             end_frame = len(data)
    #         max_info = np.max(data[start_frame: end_frame,:], axis=0)
    #         sort_idx = np.argsort(max_info)

    #         plt.subplot(len(results_dict), 1, list(results_dict.keys()).index(dataset) + 1)
    #         sns.heatmap(data[:, sort_idx].T,
    #                     cmap='viridis',
    #                     xticklabels=20,
    #                     yticklabels=False) #False f'{dataset}'
    #     plt.tight_layout()
    #     plt.show()

    # def plot_neuron_performance_heatmap(self, results_dict, decoder_type, start_frame=14, end_frame=None, metric='sc_cumulative_information_mean', significant_neurons=None):
    #     """Plot heatmap of neuron performance, highlighting significant neurons."""
    #     plt.figure(figsize=(12, len(results_dict) * 2))
        
    #     for dataset in results_dict:
    #         data = results_dict[dataset][decoder_type][metric]
    #         if end_frame is None:
    #             end_frame = len(data)
    #         max_info = np.max(data[start_frame:end_frame, :], axis=0)
    #         sort_idx = np.argsort(max_info)

    #         # Highlight significant neurons if provided
    #         if significant_neurons is not None:
    #             significant_neuron_mask = np.isin(np.arange(data.shape[1]), significant_neurons[dataset])
    #             significant_idx = np.where(significant_neuron_mask)[0]
    #             sort_idx = np.concatenate([sort_idx, significant_idx])  # Put significant neurons at the end for clarity

    #         plt.subplot(len(results_dict), 1, list(results_dict.keys()).index(dataset) + 1)
    #         sns.heatmap(data[:, sort_idx].T, cmap='viridis', xticklabels=20, yticklabels=False)
    #         plt.title(f'{dataset} Neuron Performance')

    #     plt.tight_layout()
    #     plt.show()


    # def plot_significant_neurons_distribution(self, significant_neurons_data, save_path = None):
    #     """Plot distribution of significant neurons."""
    #     fig, axes = plt.subplots(1, 2, figsize=(6, 3))

    #     for celltype, color in self.celltypecolors.items():
    #         all_peaks = []
    #         all_peaks_locs = []

    #         for dataset in significant_neurons_data:
    #             peaks = significant_neurons_data[dataset][celltype]['peak_values']
    #             peaks_locs = significant_neurons_data[dataset][celltype]['peak_frames']

    #             all_peaks.extend(peaks)
    #             all_peaks_locs.extend(peaks_locs)

    #         axes[0].hist(all_peaks, alpha=1.0, color=color, label=celltype, histtype='step', linewidth=2)
    #         axes[1].hist(all_peaks_locs, alpha=1.0, color=color, label=celltype, histtype='step', linewidth=2)

    #     axes[0].set_xlabel('Information (bits)')
    #     axes[0].spines['top'].set_visible(False)
    #     axes[0].spines['right'].set_visible(False)

    #     axes[1].set_xlabel('Peak Frame')
    #     axes[1].spines['top'].set_visible(False)
    #     axes[1].spines['right'].set_visible(False)

    #     fig.suptitle('Significant Neurons Distribution by Cell Type')
    #     plt.tight_layout()
    #     plt.show()

    #     # Save plot if save_path is provided
    #     if save_path: 
    #         plt.savefig(save_path, bbox_inches='tight')

    def plot_time_course_by_cell_type(self, results_dict, decoder_type, start_frame=14, end_frame=None, 
                                 metric='sc_instantaneous_information_mean', significance_struc=None):
        """
        Plot average information time course by cell type.
        
        Parameters:
        -----------
        results_dict : dict
            Dictionary containing results data
        decoder_type : str
            Type of decoder used
        start_frame : int
            Starting frame for analysis
        end_frame : int, optional
            Ending frame for analysis
        metric : str
            Metric to plot
        significance_struc : dict, optional
            Dictionary containing significant neuron indices by dataset
        """
        plt.figure(figsize=(3, 3))
        for cel_index, (celltype, color) in enumerate(self.celltypecolors.items()):
            all_traces = []
            for dataset in results_dict:
                data = results_dict[dataset][decoder_type][metric]
                if end_frame is None:
                    end_frame = len(data)
                celltype_idx = results_dict[dataset]['celltype_array'] == cel_index

                # Filter for significant neurons if provided
                if significance_struc is not None and dataset in significance_struc:
                    if celltype in significance_struc[dataset]:
                        sig_neurons = significance_struc[dataset][celltype]['neuron_indices']
                        celltype_idx = np.logical_and(celltype_idx, 
                                                    np.isin(np.arange(len(celltype_idx)), 
                                                    sig_neurons))

                if np.any(celltype_idx):
                    mean_trace = np.mean(data[start_frame:end_frame, celltype_idx], axis=1)
                    all_traces.append(mean_trace)

            if all_traces:
                mean = np.mean(all_traces, axis=0)
                sem = np.std(all_traces, axis=0) / np.sqrt(len(all_traces))
                plt.plot(mean, color=color, label=celltype)
                plt.fill_between(range(len(mean)), mean-sem, mean+sem, alpha=0.2, color=color)

        ax = plt.gca()
        # ax.axvline(x=start_frame, color='k', linestyle=':', alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_box_aspect(1)

        plt.title('Average Information Time Course by Cell Type')
        # plt.xlabel('Time (frames)')
        plt.ylabel('Information (bits)')

        event_onset = self.get_event_frame_for_decoder(decoder_type)
        print(event_onset)
        xticks_in, xticks_lab = self.x_axis_sec_aligned(event_onset- start_frame, end_frame - start_frame, interval=10, frame_rate=30)
        ax.axvline(x=event_onset, color='k', linestyle=(0, (10.5,6.8)), alpha=1,lw=0.7)

        plt.xticks(ticks=xticks_in, labels=xticks_lab)
        plt.xlabel('Time (s)')

        plt.show()

    def plot_significant_neuron_percentages_by_celltype_total(self, significance_struc, neuron_groups, save_path=None, figsize = (3,3)):
        """
        Plot the percentage of significantly modulated neurons per dataset for each cell type and all neurons combined.
        This function calculates the percentage of significant neurons for each cell type across all neurons.
        Parameters:
        -----------
        significance_struc : dict
            Dictionary containing significant neuron data by dataset and cell type, including 'sig_neurons_all'
        neuron_groups : dict
            Dictionary containing all neuron indices for each dataset, organized by cell type
        save_path : str, optional
            Path to save the plot
        """
        plt.rcParams.update({'font.size': 7, 'font.family': 'arial'})
        mpl.rcParams['pdf.fonttype'] = 42   # TrueType fonts (editable)
        percentages_by_celltype = {ct: [] for ct in self.celltypecolors.keys()}
        percentages_by_celltype["All"] = []
        
        # Track totals across all datasets
        total_significant_across_datasets = 0
        total_neurons_across_datasets = 0

        for dataset in significance_struc:
            total_neurons_all = 0
            total_significant_all = len(significance_struc[dataset]['sig_neurons_all'])
            total_significant_across_datasets += total_significant_all

            # Get total neurons for this dataset
            total_neurons_all = sum(len(neurons.flatten()) 
                                for neurons in neuron_groups[dataset]['neuron_groups'].values())

            for celltype in self.celltypecolors:
                celltype_neurons = neuron_groups[dataset]['neuron_groups'].get(celltype, np.array([])).flatten()
                # total_neurons = sum(len(neurons.flatten()) 
                #                 for neurons in neuron_groups[dataset]['neuron_groups'].values())
                # total_neurons_all += total_neurons

                total_neurons_celltype = len(celltype_neurons)
                
                if total_neurons_celltype > 0:
                    significant_neurons = significance_struc[dataset][celltype]['neuron_indices']
                    total_significant = len(significant_neurons)
                    percentages_by_celltype[celltype].append((total_significant / total_neurons_all) * 100)
                else:
                    percentages_by_celltype[celltype].append(0)

            total_neurons_across_datasets += total_neurons_all
            if total_neurons_all > 0:
                percentages_by_celltype["All"].append((total_significant_all / total_neurons_all) * 100)

        print(f'Total significant neurons across all datasets: {total_significant_across_datasets}')
        print(f'Total neurons across all datasets: {total_neurons_across_datasets}')

        # Calculate mean and SEM for each cell type
        means = {ct: np.mean(percentages_by_celltype[ct]) for ct in percentages_by_celltype}
        sems = {ct: np.std(percentages_by_celltype[ct]) / np.sqrt(len(percentages_by_celltype[ct]))
                for ct in percentages_by_celltype}

        # Plot bar chart
        fig, ax = plt.subplots(figsize=figsize)
        x_positions = np.arange(len(self.celltypecolors) + 1)  # One bar per cell type + "All"
        colors = [self.celltypecolors[ct] for ct in self.celltypecolors.keys()] + ["black"] #[(0, 0, 0)]#

        # # ax.bar(x_positions, [means[ct] for ct in list(self.celltypecolors.keys()) + ["All"]], 
        # #     yerr=[sems[ct] for ct in list(self.celltypecolors.keys()) + ["All"]],
        # #     color=colors, edgecolor="black", capsize=5, alpha=1, width=0.6)

        # # Bar plot with error bars, colored edges, white interior, and error bar caps

        # Plot each bar individually to set the color, edgecolor, and error bar styling
        for i, (mean, sem, color) in enumerate(zip(
            [means[ct] for ct in list(self.celltypecolors.keys()) + ["All"]],
            [sems[ct] for ct in list(self.celltypecolors.keys()) + ["All"]],
            colors,
        )):
            # Create the bar
            bar = ax.bar(
                x_positions[i], 
                mean, 
                facecolor='white',    # Empty inside
                edgecolor=color,      # Edge color of the bar
                alpha=1, 
                width=0.6, 
                linewidth=2
            )
            # Add error bars
            ax.errorbar(
                x_positions[i], 
                mean, 
                yerr=sem, 
                fmt='none',  # Don't plot additional points
                ecolor=color,  # Color for error bars and caps
                elinewidth=2,  # Thickness of error lines
                capsize=5,     # Cap size
                capthick=2     # Cap thickness
            )

        # Aesthetics
        ax.set_xticks(x_positions)
        ax.set_xticklabels(list(self.cell_type_labels.values())+ ["All"], fontsize=12) #list(self.celltypecolors.keys()) + ["All"], fontsize=12)
        ax.set_ylabel("% Modulated Neurons", fontsize=7)
        #ax.set_title("Significantly Modulated Neurons Across Cell Types", fontsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Save or show plot
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        plt.tight_layout()
        plt.show()

        # Print summary
        print("Significantly Modulated Neurons (% ± SEM):")
        for celltype in list(self.celltypecolors.keys()) + ["All"]:
            print(f"{celltype}: {means[celltype]:.2f} ± {sems[celltype]:.2f}%")


    def plot_significant_neuron_percentages_by_celltype(self, significance_struc, neuron_groups, save_path=None, star_height_percentage=0.1):
        """
        Plot the percentage of significantly modulated neurons per dataset for each cell type (% within each cell type).
        
        Parameters:
        -----------
        significance_struc : dict
            Dictionary containing significant neuron data by dataset and cell type.
        neuron_groups : dict
            Dictionary containing all neuron indices for each dataset, organized by cell type.
        save_path : str, optional
            Path to save the plot
        """
        plt.rcParams.update({'font.size': 7, 'font.family': 'arial'})
        percentages_by_celltype = {ct: [] for ct in self.celltypecolors.keys()}
        
        # Track totals across all datasets
        total_significant_across_datasets = 0
        total_neurons_across_datasets = 0

        for dataset in significance_struc:
            for celltype in self.celltypecolors:
                celltype_neurons = neuron_groups[dataset]['neuron_groups'].get(celltype, np.array([])).flatten()
                total_neurons_celltype = len(celltype_neurons)
                total_neurons_across_datasets += total_neurons_celltype

                if total_neurons_celltype > 0:
                    significant_neurons = significance_struc[dataset][celltype]['neuron_indices']
                    total_significant = len(significant_neurons)
                    total_significant_across_datasets += total_significant
                    percentages_by_celltype[celltype].append((total_significant / total_neurons_celltype) * 100)
                else:
                    percentages_by_celltype[celltype].append(0)

        print(f'Total significant neurons across all datasets: {total_significant_across_datasets}')
        print(f'Total neurons across all datasets: {total_neurons_across_datasets}')

        # Calculate mean and SEM for each cell type
        means = {ct: np.mean(percentages_by_celltype[ct]) for ct in percentages_by_celltype}
        sems = {ct: np.std(percentages_by_celltype[ct]) / np.sqrt(len(percentages_by_celltype[ct])) 
                for ct in percentages_by_celltype}

        # Plot bar chart
        fig, ax = plt.subplots(figsize=(1, 1), dpi=300)
        x_positions = np.arange(len(self.celltypecolors))
        colors = [self.celltypecolors[ct] for ct in self.celltypecolors]

        for i, (ct, color) in enumerate(zip(self.celltypecolors, colors)):
            mean = means[ct]
            sem = sems[ct]
            # ax.bar(
            #     x_positions[i],
            #     mean,
            #     facecolor='white',
            #     edgecolor=color,
            #     alpha=1,
            #     width=0.6,
            #     linewidth=1
            # )
            # ax.errorbar(
            #     x_positions[i],
            #     mean,
            #     yerr=sem,
            #     fmt='none',
            #     ecolor=color,
            #     elinewidth=1,
            #     capsize=2,
            #     capthick=1
            # )
            ax.bar(
                x_positions[i],
                mean,
                facecolor=color,
                edgecolor='white',
                alpha=1,
                width=0.6,
                linewidth=0.1
            )
            ax.errorbar(
                x_positions[i],
                mean,
                yerr=sem,
                fmt='none',
                ecolor='black',
                elinewidth=.5,
                capsize=2,
                capthick=.5
            )

        # Aesthetics
        ax.set_xticks(x_positions)
        ax.set_xticklabels([self.cell_type_labels[ct] for ct in self.celltypecolors])
        plt.xticks(rotation=45)
        ax.set_ylabel("% Informative")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ylims =  plt.gca().get_ylim()

        #kruskal wallis test across all cell types
        group1 = np.array(percentages_by_celltype['pyr'])
        group2 = np.array(percentages_by_celltype['som'])
        group3 = np.array(percentages_by_celltype['pv'])

        # Optional: remove NaNs
        group1 = group1[~np.isnan(group1)]
        group2 = group2[~np.isnan(group2)]
        group3 = group3[~np.isnan(group3)]
        # Run Kruskal–Wallis test
        # call your method
        kw_table = self.stats.kruskal_wallis_to_pd('percentages_by_celltype', group1, group2, group3)
        kw_significant = (kw_table["p_value"] < 0.05).any()

        # perform permutation test for each cell type
        celltype_keys = list(self.celltypecolors.keys())
        all_p_values = []
        comparisons = []
        all_stats_dict = {}
        test_stats = []
        comparisons_names = []
        # Loop through each pair of cell types
        for i, celltype in enumerate(celltype_keys):
            for j in range(i + 1, len(celltype_keys)):
                other_celltype = celltype_keys[j]
                comparisons.append((i, j))
                # Get data for the two cell types
                data_i = np.array(percentages_by_celltype[celltype])
                data_j = np.array(percentages_by_celltype[other_celltype])
                if len(data_i) == 0 or len(data_j) == 0:
                    print(f"No significant peaks found for {celltype} or {other_celltype}. Skipping permutation test.")
                    continue
                
                # Perform permutation test
                p_value, stat = self.stats.perform_permutation_test(data_i, data_j, paired=True, n_permutations=10000)
                all_p_values.append(p_value)
                test_stats.append(stat)
                comparisons_names.append((f"{celltype}_%_sig_info", f"{other_celltype}_%_sig_info"))

                print(f"paired permutation test p-value for {celltype} vs {other_celltype}: {p_value:.4f}")   
                # Save stats for each group
                label1 = f"{celltype}_%_sig_info"
                label2 = f"{other_celltype}_%_sig_info"
                all_stats_dict[label1] = self.stats.get_basic_stats(data_i)
                all_stats_dict[label2] = self.stats.get_basic_stats(data_j) 


        _, significance_stars = self.stats.calculate_bonferroni_significance(all_p_values, alpha=0.05)

        # Add significance stars to the plot
        if kw_significant:
            count = 0
            for (i, j), star in zip(comparisons, significance_stars):
                if star != 'ns':  # Only add significance line if there is a star
                    self.add_significance_line(ax, x1=i, x2=j, y=ylims[1]+count,significance=star, color='black',star_height_percentage =star_height_percentage)
                    count += ylims[1]*.2

        # get actual save_path by getting the string in front of the last /
        if save_path and '/' in save_path:
            save_path_updated = save_path[:save_path.rfind('/')]

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            df_tests = self.stats.to_table(comparisons_names, test_stats, all_p_values, save_path=f'{save_path_updated}/stat_tests_info_sig_percent.csv',type='permutation paired')
            df_tests = pd.concat([df_tests, kw_table], ignore_index=True)
            df_tests.to_csv(f'{save_path_updated}/stat_tests_info_sig_percent.csv', index=False)
            df_stats = self.stats.basic_stats_to_table(all_stats_dict, save_path=f'{save_path_updated}/basic_stats_info_sig_percent.csv')

        plt.tight_layout()
        plt.show()

        print("Significantly Modulated Neurons (% ± SEM):")
        for ct in self.celltypecolors:
            print(f"{ct}: {means[ct]:.2f} ± {sems[ct]:.2f}%")

        return means, sems, percentages_by_celltype


    def scatter_plot_with_sem(self,labels, means, sems, colors=['blue', 'red'], title='Scatter Plot', ylabel='Value', save_dir=None):
        """
        Create a scatter plot with error bars (SEM).
        
        Parameters:
        - labels: list of str, labels for the points
        - means: list of float, mean values for each point
        - sems: list of float, standard error of the mean for each point
        - colors: list of str, colors for the points
        - title: str, title of the plot
        - ylabel: str, label for the y-axis
        - save_dir: str, directory to save the plot
        """
        x = np.arange(len(labels))  # the label locations

        fig, ax = plt.subplots(figsize=(3, 3))
        # Set global font size and family 
        plt.rcParams.update({'font.size': 7, 'font.family': 'arial'})
        
        # Plot scatter points with error bars
        for i in range(len(labels)):
            ax.errorbar(x[i], means[i], yerr=sems[i], fmt='o', color=colors[i], capsize=5, markersize=10, label=labels[i])

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        # Clean up the appearance
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        fig.tight_layout()

        # Get the y-axis limits (used for significance stars)
        y_limits = plt.gca().get_ylim()
        x_limits = plt.gca().get_xlim()
        ax.set(ylim=(y_limits[0]*.9,y_limits[1]*1.1))
        ax.set(xlim=(x_limits[0]-1,x_limits[1]+1))
        ax.set_xticklabels(labels,rotation=45)

        # Save the plot if save_dir is provided
        if save_dir:
            plt.savefig(f"{save_dir}_scatter_plot.png", bbox_inches='tight')

        #plt.show()
        return ax, x, y_limits
    def simple_bar_plot(self,labels, means, sems,colors = ['blue','red'], title='Bar Plot', ylabel='Value', save_dir=None, figsize = (3,3)):
        """
        Create a simple bar plot with error bars.
        
        Parameters:
        - labels: list of str, labels for the bars
        - means: list of float, mean values for each bar
        - sems: list of float, standard error of the mean for each bar
        - title: str, title of the plot
        - ylabel: str, label for the y-axis
        - save_dir: str, directory to save the plot
        """
        
        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots(figsize=figsize)
        # Set global font size and family 
        plt.rcParams.update({'font.size': 7, 'font.family': 'arial'})
        bars = ax.bar(x, means, width, yerr=sems, capsize=4, edgecolor= colors, facecolor='white', linewidth=2) #,ecolor=colors

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        # Clean up the appearance
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Get the y-axis limits (used for significance stars)
        y_limits = plt.gca().get_ylim()
        ax.set(ylim=(0,y_limits[1]*1.1))

        fig.tight_layout()

        # Save the plot if save_dir is provided
        if save_dir:
            plt.savefig(f"{save_dir}_bar_plot.png", bbox_inches='tight') #/{title.replace(' ', '_')}

        #plt.show()
        return ax,x, y_limits

    def simple_plot_wrapper(self,labels, data,sem, plot_type = 'bar',colors= ['blue','red'], frames = None, ylabel = 'Mean Value', save_dir = None):
        """ create wrapper for bar plot to calculate significance
        data: list of values for conditions (active, passive) 
        """
        
        # Calculate mean and SEM

        if frames is None:
            frames = np.arange(len(data))   
        means = list(np.mean(data[:,frames], axis = 1))
        sems = list(np.mean(sem[:,frames], axis = 1)) #, axis = 1

        # Create the bar plot   
        if 'bar' in plot_type:
            ax,x, ymax = self.simple_bar_plot(labels, means, sems,colors = colors, title= None, ylabel= ylabel, save_dir=save_dir)
        elif 'scatter' in plot_type:
            ax,x, ymax = self.scatter_plot_with_sem(labels, means, sems, colors=colors, title= None, ylabel=ylabel, save_dir=save_dir)

        # Perform pairwise comparisons
        all_p_values = []
        comparisons = []
        for i in range(data.shape[0]):
            for j in range(i + 1, data.shape[0]):
                print(f' {data[i, frames], data[j, frames]} ')
                # Fixed the argument order in permutation_test call
                p_value, _ = self.stats.perform_permutation_test(
                    data1=data[i, frames],
                    data2=data[j, frames],
                    paired=False,
                    n_permutations=10000
                )
                all_p_values.append(p_value)
                comparisons.append((i, j))

        corrected_p_values, significance_stars = self.stats.calculate_bonferroni_significance(all_p_values, alpha=0.05)

        # Add significance stars to the plot
        for (i, j), star in zip(comparisons, significance_stars):
            if star != 'ns':  # Only add significance line if there is a star
                self.add_significance_line(ax, x1=x[i], x2=x[j], y=ymax[1], significance=star, color='black')

        plt.show()  # Ensure the plot is updated
        return all_p_values,corrected_p_values

    def plot_selected_metric_with_sem(self,mean_results_list, decoder_types, metric, start_frame=0, xlim=None, ylim=None, title='Decoding Results', xlabel='Frames', ylabel=None, colors =['blue', 'red', 'green', 'purple', 'orange'],labels =['Active','Passive'], save_dir=None, text_loc = None):
        """
        Plot the selected metric from multiple mean_results on the same plot with SEM shading.
        
        Parameters:
        - mean_results_list: list of dicts, results for different conditions
        - decoder_types: list of str, the type of decoder used for each mean_results
        - metric: str, the metric to plot
        - start_frame: int, the starting frame for the plot
        - xlim: tuple, x-axis limits
        - ylim: tuple, y-axis limits
        - title: str, title of the plot
        - xlabel: str, label for the x-axis
        - ylabel: str, label for the y-axis
        - save_dir: str, directory to save the plot
        """
        #labels = [f'Condition {i+1}' for i in range(len(mean_results_list))]
        
        plt.figure(figsize=(4,3))
        
        all_means = []
        all_sems = []
        
        for idx, (mean_results, decoder_type) in enumerate(zip(mean_results_list, decoder_types)):
            # Get event frames from first dataset
            first_dataset = list(mean_results.keys())[0]
            event_frames = mean_results[first_dataset][decoder_type]['event_frame_mean']
            
            # Determine if the metric is 'sc' or 'pop'
            plot_type = 'sc' if 'sc' in metric else 'pop'
            
            # Collect data across datasets
            all_data = []
            for dataset in mean_results.keys():
                if decoder_type in mean_results[dataset]:
                    data = mean_results[dataset][decoder_type][metric]

                    # Average across neurons for sc data
                    if plot_type == 'sc' and len(data.shape) == 2:  # frames x neurons
                        data = np.mean(data, axis=1)  # average across neurons

                    all_data.append(data)
            
            # Convert lists to NumPy arrays
            all_data = np.array(all_data)
            
            # Use only the frames within xlim if provided
            if xlim:
                used_frames = slice(xlim[0], xlim[1])
            else:
                used_frames = slice(0, all_data.shape[1])
            
            all_data_final = all_data[:, used_frames]
            
            # Calculate mean and SEM
            mean_trace = np.mean(all_data_final, axis=0)
            sem_trace = np.std(all_data_final, axis=0) / np.sqrt(len(all_data_final))
            
            all_means.append(mean_trace)
            all_sems.append(sem_trace)

            x = np.arange(len(mean_trace)) / 30.0  # Convert frames to seconds
            
            # Plot the metric values
            plt.plot(x, mean_trace, '-', label=labels[idx], color=colors[idx % len(colors)])
            plt.fill_between(x, mean_trace - sem_trace, mean_trace + sem_trace, alpha=0.3, color=colors[idx % len(colors)])
            
            # Add event markers
            for frame in event_frames:
                if frame < len(mean_trace):
                    plt.axvline(x=(frame - start_frame) / 30.0, color='k', linestyle=(0, (10.5,6.8)), alpha=1,lw=0.7)
            
            # x = np.arange(len(mean_trace))
            
            # # Plot the metric values
            # plt.plot(mean_trace, '-', label=labels[idx], color=colors[idx % len(colors)])
            # plt.fill_between(x, mean_trace - sem_trace, mean_trace + sem_trace, alpha=0.3, color=colors[idx % len(colors)])
            
            # # Add event markers
            # for frame in event_frames:
            #     if frame < len(mean_trace):
            #         plt.axvline(x=frame - start_frame, color='k', linestyle=':', alpha=0.5)
        
        # Formatting
        plt.title(title)
        xticks_in, xticks_lab = self.x_axis_sec_aligned(event_frames[0], len(x), interval=1, frame_rate=30) 
        
        if ylabel:
            plt.ylabel(ylabel)
        else:
            plt.ylabel('Bits' if 'information' in metric else 'Fraction Correct')
        
        #plt.xlim(xlim if xlim else (0 + start_frame, len(mean_trace) - start_frame))
        if xlim:
            xlim = (xlim[0] / 30.0, xlim[1] / 30.0)
        plt.xlim(xlim if xlim else (0 + start_frame / 30.0, len(mean_trace) / 30.0 - start_frame / 30.0))
        xlims = plt.gca().get_xlim()
        if ylim:
            plt.ylim(ylim)
        else:
            ylim = plt.gca().get_ylim()
        
        # Add text annotations for the labels
        for i, label in enumerate(labels):
            if text_loc is None:
                plt.text(xlims[1]-xlims[1]*.1/ 30.0, ylim[1] - ylim[1]*.1*(i+1), label, color=colors[i], verticalalignment='center')
                #plt.text(xlims[1]-20, ylim[1] - ylim[1]*.1*(i+1), label, color=colors[i], verticalalignment='center')
            else:
                plt.text(text_loc[0], text_loc[1], label, color=colors[i], verticalalignment='center')   
        
        plt.grid(False)
        plt.tight_layout()

        # Clean up the appearance
        ax = plt.gca()  # get current axis  
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        #set up x axis
        plt.xlabel('Time (s)')
        ax.set_xticks(xticks_in/30)  # Set x-ticks to the integer values (divide by 30 bc x was in sec)
        ax.set_xticklabels(xticks_lab)  # Set x-tick labels to the integer values
        
        # Save the plot if save_dir is provided
        if save_dir:
            plt.savefig(f"{save_dir}/{metric}_comparison_traces.png", bbox_inches='tight')
        
        plt.show()
        
        # Concatenate all means and SEMs
        concatenated_means = np.vstack(all_means)
        concatenated_sems = np.vstack(all_sems)
        
        return concatenated_means, concatenated_sems


    def plot_dataset_metric_with_sem(self, mean_results_list, dataset_key, decoder_types, metric, 
                                start_frame=0, xlim=None, ylim=None, title=None, 
                                xlabel='Frames', ylabel=None, colors=['blue', 'red'], 
                                labels=['Active','Passive'], save_dir=None):
        """Plot metric for specific dataset across conditions."""
        plt.figure(figsize=(4,3))
        
        for idx, (mean_results, decoder_type) in enumerate(zip(mean_results_list, decoder_types)):
            # Get data for specific dataset
            if dataset_key not in mean_results:
                print(f"Dataset {dataset_key} not found in results")
                continue
                
            # Get event frames
            event_frames = mean_results[dataset_key][decoder_type]['event_frame_mean']
            
            # Determine metric type
            plot_type = 'sc' if 'sc' in metric else 'pop'
            
            # Get data for this dataset
            data = mean_results[dataset_key][decoder_type][metric]
            
            # Calculate mean and SEM
            if plot_type == 'sc':
                mean_trace = np.mean(data, axis=1)
                sem_trace = scipy.stats.sem(data, axis=1)
            else:
                mean_trace = data
            
            
            # Create x-axis
            x = np.arange(len(mean_trace))
            
            # Plot
            plt.plot(x[start_frame:], mean_trace[start_frame:], 
                    color=colors[idx], label=labels[idx])
            if plot_type == 'sc':
                plt.fill_between(x[start_frame:], 
                                mean_trace[start_frame:] - sem_trace[start_frame:],
                                mean_trace[start_frame:] + sem_trace[start_frame:],
                                alpha=0.2, color=colors[idx])
            
            # Add event frame line
            plt.axvline(x=event_frames[0], color='k', linestyle='--', alpha=0.5)
        
        # Customize plot
        if title is None:
            title = f"{dataset_key} - {metric}"
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel if ylabel else metric)
        plt.legend(frameon=False)

        xticks_in, xticks_lab = self.x_axis_sec_aligned(event_frames[0], len(mean_trace), interval=1, frame_rate=30)
        plt.xticks(ticks=xticks_in, labels=xticks_lab)
        plt.xlabel('Time (s)')
        
        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"{dataset_key}_{metric}.png"))
        
        plt.show()


    def get_event_frame_for_decoder(self, decoder_type):
        """
        Get the event frame for a given decoder type.
        
        Parameters:
        -----------
        decoder_type : str
            Type of decoder (e.g., 'outcome', 'choice', 'sound_category', 'photostim')
            Can include 'shuffled/' prefix
        
        Returns:
        --------
        float
            Event frame number (6 for sound/photostim, 131 for choice, 145 for outcome)
        """
        # Remove 'shuffled/' prefix if present
        decoder_base = decoder_type.replace('shuffled/', '')
        
        # Map decoder types to event frame indices
        decoder_to_event = {
            'outcome': self.event_frames[4],    # Frame 145
            'choice': self.event_frames[3],     # Frame 131
            'sound_category': self.event_frames[0],  # Frame 6
            'photostim': self.event_frames[0]   # Frame 6
        }
        
        return decoder_to_event.get(decoder_base, self.event_frames[0])  # Default to first frame if not found
    def plot_significant_neuron_session_means(
        self,
        significant_neurons_data,
        event_frames=None,
        save_path=None,
        figure_type='cdf',
        star_height_percentage=0.05,
        fig_size=(3, 1.6),
    ):
        """
        Plot distribution of session-mean peak values for significant informative neurons.
        Each data point is the mean for a session/cell type.
        """

        fig, axes = plt.subplots(1, 2, figsize=fig_size, dpi=300)
        plt.subplots_adjust(wspace=0.1, left=0.2, top=2)
        plt.rcParams.update({'font.size': 7, 'font.family': 'arial'})
        mpl.rcParams['pdf.fonttype'] = 42   # TrueType fonts (editable)
        celltypes = list(self.celltypecolors.keys())
        # celltypes = [celltype.upper() for celltype in self.celltypecolors.keys()]
        bin_edges_bits = np.arange(0.06, .2, .01)

        # Collect session means for each cell type
        session_means = {celltype: {'peak_values': [], 'peak_frames': []} for celltype in celltypes}

        for celltype, color in self.celltypecolors.items():
            means_peaks = []
            means_locs = []
            for dataset in significant_neurons_data:
                peaks = np.array(significant_neurons_data[dataset][celltype]['peak_values'])
                peaks_locs = np.array(significant_neurons_data[dataset][celltype]['peak_frames'])
                if len(peaks) > 0:
                    means_peaks.append(np.mean(peaks))
                if len(peaks_locs) > 0:
                    means_locs.append(np.mean(peaks_locs))
            session_means[celltype]['peak_values'] = means_peaks
            session_means[celltype]['peak_frames'] = means_locs

            # Plotting (same as original, but with session means)
            if figure_type == 'violin':
                sns.violinplot(
                    x=[celltype] * len(means_peaks),
                    y=means_peaks,
                    ax=axes[0],
                    color=color,
                    inner='box',
                    linewidth=1,
                    edgecolor=color,
                )
                axes[0].set_ylabel("Mean Peak Info. (bits)")
                axes[0].set_xticklabels([ct.upper() for ct in celltypes], rotation=45)
                axes[0].set_ylim(0, 0.2)
            if figure_type == 'cdf':
                x1 = np.linspace(0.05, .15, 100)
                n1, _ = np.histogram(means_peaks, bins=x1)
                p1 = n1 / np.sum(n1) if np.sum(n1) > 0 else n1
                cdf = np.cumsum(p1)
                axes[0].plot(x1[:-1], cdf, linewidth=1, color=color)
                axes[0].set_ylabel("Cumulative Fraction")
            elif figure_type == 'histogram':
                weights = np.ones_like(means_peaks) / len(means_peaks) if len(means_peaks) > 0 else None
                axes[0].hist(means_peaks, alpha=1.0, color=color, label=celltype,
                            histtype='step', linewidth=1, density=False, bins=bin_edges_bits, weights=weights)
                axes[0].set_ylabel("Fraction")

            # Plot for mean peak locations
            x1 = np.linspace(1, 169-14, 169-14)
            n1, _ = np.histogram(means_locs, bins=x1)
            p1 = n1 / np.sum(n1) if np.sum(n1) > 0 else n1
            cdf = np.cumsum(p1)
            axes[1].hist(means_locs, alpha=0.7, color=color, bins=np.arange(0, 169, 15), label=celltype,
                        density=False, weights=np.ones_like(means_locs) / len(means_locs) if len(means_locs) > 0 else None,
                        histtype='step', linewidth=1)

        axes[1].set_ylabel("Fraction")
        axes[1].set_xlabel('Peak Info. (bits)')
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)

        if event_frames is not None:
            xlim = axes[1].get_xlim()
            for frame in event_frames:
                if frame < xlim[1]:
                    axes[1].axvline(x=frame, color='k', linestyle=(0, (10.5,6.8)), alpha=1,lw=0.7)
            axes[1].set_xticks(event_frames)
            axes[1].set_xticklabels(self.event_labels)
            plt.xticks(rotation=45)
        plt.tight_layout()

        # Statistical tests (on session means)
        celltype_keys = list(self.celltypecolors.keys())
        all_p_values = []
        comparisons = []
        comparisons_names = []
        p_values = []
        test_stats = []
        all_stats_dict = {}

        for i, celltype in enumerate(celltype_keys):
            for j in range(i + 1, len(celltype_keys)):
                other_celltype = celltype_keys[j]
                comparisons.append((i, j))
                data_i = np.array(session_means[celltype]['peak_values'])
                data_j = np.array(session_means[other_celltype]['peak_values'])
                if len(data_i) == 0 or len(data_j) == 0:
                    print(f"No session means found for {celltype} or {other_celltype}. Skipping permutation test.")
                    continue
                p_value, stat = self.stats.perform_permutation_test(data_i, data_j, paired=False, n_permutations=10000)
                all_p_values.append(p_value)
                comparisons_names.append((f"{celltype}_peak_vals", f"{other_celltype}_peak_vals"))
                test_stats.append(stat)
                p_values.append(p_value)
                label1 = f"{celltype}_peakvals"
                label2 = f"{other_celltype}_peakvals"
                all_stats_dict[label1] = self.stats.get_basic_stats(data_i)
                all_stats_dict[label2] = self.stats.get_basic_stats(data_j)
                print(f"Permutation test p-value for {celltype} vs {other_celltype}: {p_value:.4f}")

        _, significance_stars = self.stats.calculate_bonferroni_significance(all_p_values, alpha=0.05)

        ylims = axes[0].get_ylim()
        count = 0
        for (i, j), star in zip(comparisons, significance_stars):
            if star != 'ns':
                star_y = ylims[1] - .03 + count
                self.add_significance_line(axes[0], x1=i, x2=j, y=star_y, significance=star, color='black', star_height_percentage=star_height_percentage, fontsize=7)
                count += .03

        for i, celltype in enumerate(celltype_keys):
            for j in range(i + 1, len(celltype_keys)):
                other_celltype = celltype_keys[j]
                comparisons.append((i, j))
                data_i = np.array(session_means[celltype]['peak_frames'])
                data_j = np.array(session_means[other_celltype]['peak_frames'])
                if len(data_i) == 0 or len(data_j) == 0:
                    print(f"No session means found for {celltype} or {other_celltype}. Skipping permutation test.")
                    continue
                p_value, stat = self.stats.perform_permutation_test(data_i, data_j, paired=False, n_permutations=10000)
                all_p_values.append(p_value)
                comparisons_names.append((f"{celltype}_peak_locs", f"{other_celltype}_peak_locs"))
                test_stats.append(stat)
                p_values.append(p_value)
                label1 = f"{celltype}_peaklocs"
                label2 = f"{other_celltype}_peaklocs"
                all_stats_dict[label1] = self.stats.get_basic_stats(data_i)
                all_stats_dict[label2] = self.stats.get_basic_stats(data_j)
                print(f"Permutation test locs p-value for {celltype} vs {other_celltype}: {p_value:.4f}")

        if save_path and '/' in save_path:
            save_path_updated = save_path[:save_path.rfind('/')]
        if save_path:
            plt.savefig(save_path, dpi=300)
            df_tests = self.stats.to_table(comparisons_names, test_stats, p_values, save_path=f'{save_path_updated}/stat_tests_info_neurons_peaks_sessionmeans.csv', type='permutation')
            df_stats = self.stats.basic_stats_to_table(all_stats_dict, save_path=f'{save_path_updated}/basic_stats_info_neurons_peaks_sessionmeans.csv')
        else:
            df_tests = self.stats.to_table(comparisons_names, test_stats, all_p_values, type='permutation')
        plt.show()
    def plot_significant_neurons_distribution(self,significant_neurons_data, event_frames=None, save_path=None, figure_type='cdf',star_height_percentage=0.05, figsize=(3, 1.6), bin_size=3, ylim_axis0 = None, plot_peak_locs = True): 
        """Plot distribution of significant informative neurons."""
        
        if plot_peak_locs == False:
            fig, axes = plt.subplots(1, 1, figsize=(figsize[0]/2, figsize[1]), dpi=300) #, constrained_layout=True
            axes = [axes]  # Make axes a list for consistent indexing
        else:
            fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=300) #, constrained_layout=True
        plt.subplots_adjust(wspace=0.1,left=0.2, top=2)    # Adjust for more space between plots
        plt.rcParams.update({'font.size': 7, 'font.family': 'arial'})  # Updated font size for clarity
        mpl.rcParams['pdf.fonttype'] = 42   # TrueType fonts (editable)

        celltypes = list(['PYR', 'SOM', 'PV'])  # Define cell types to plot
        # Adjusted bin edges and color palette
        bin_edges_bits = np.arange(0.06, .2, .01)

        #save peaks and peak locations for each cell type across all datasets concatenated
        collected_peaks = {celltype: {'peak_values': [], 'peak_frames': []} for celltype in self.celltypecolors.keys()}

        # Collect peaks and peak locations for each cell type across datasets   
        for celltype, color in self.celltypecolors.items():
            all_peaks = []
            all_peaks_locs = []
            for dataset in significant_neurons_data:
                peaks = significant_neurons_data[dataset][celltype]['peak_values']
                peaks_locs = significant_neurons_data[dataset][celltype]['peak_frames']
                all_peaks.extend(peaks)
                all_peaks_locs.extend(peaks_locs)

            # collect all peaks and peak locations for each cell type
            if not all_peaks:  # Check if there are no peaks for this cell type
                print(f"No significant peaks found for cell type {celltype} in dataset {dataset}. Skipping plot.")
            else:
                #save the peaks and peak locations for each cell type
                collected_peaks [celltype]['peak_values'] = all_peaks
                collected_peaks [celltype]['peak_frames'] = all_peaks_locs
                
            if figure_type == 'violin':
                # Plot violin plot with more customization
                sns.violinplot(
                    x=[celltype] * len(all_peaks),
                    y=all_peaks,
                    ax=axes[0],
                    color=color, #color
                    inner='box',
                    linewidth=.5,
                    edgecolor=color  # Add edge color for better visibility
                    # bw=0.3  # Adjust bandwidth for smoother distributions
                )
                axes[0].set_ylabel("Peak Info. (bits)")
                axes[0].set_xticklabels(celltypes, rotation=45)
                ylims1 =  plt.gca().get_ylim()
                if ylims1[1] < 0.25:  # Check if y-axis limit is greater than 0.25
                    axes[0].set_ylim(0, 0.25)  # Setting y-axis limits to match example
                else:
                    axes[0].set_ylim(0, ylims1[1]+.03)  # Adjust y-axis limit to be slightly above max value
            if figure_type == 'cdf':
                # Calculate CDF
                x1 = np.linspace(0.05, .15, 100)  # Define range of x values
                n1, _ = np.histogram(all_peaks, bins=x1)  # Histogram counts
                p1 = n1 / np.sum(n1)  # Probability
                cdf = np.cumsum(p1)  # Cumulative sum to get CDF
                
                axes[0].plot(x1[:-1], cdf, linewidth= 1, color=color)  # x1[:-1] because histogram bins include right edge
                axes[0].set_ylabel("Cumulative Fraction")
            elif figure_type == 'histogram':
                # Plot histogram
                ## Normalize to percent using density=True and scaling y by 100
                weights = np.ones_like(all_peaks) / len(all_peaks)  # Normalize weights to sum to 1
                axes[0].hist(all_peaks, alpha=1.0, color=color, label=celltype,
                            histtype='step', linewidth=1, density=False,bins = bin_edges_bits,weights=weights)
                axes[0].set_ylabel("Fraction")

            # get ylimits
            ylims =  plt.gca().get_ylim()
            # Histogram for peak locations with customized style

            # Calculate CDF
            x1 = np.linspace(1, 169-14, 169-14)  # Define range of x values
            n1, _ = np.histogram(all_peaks_locs, bins=x1)  # Histogram counts
            p1 = n1 / np.sum(n1)  # Probability
            cdf = np.cumsum(p1)  # Cumulative sum to get CDF
            
            # axes[1].plot(x1[:-1], cdf, linewidth= 1, color=color)  # x1[:-1] because histogram bins include right edge
            # axes[1].set_ylabel("cdf")
            if plot_peak_locs == True:
                axes[1].hist(all_peaks_locs, alpha=0.7, color=color, bins=np.arange(0, 169, bin_size), label=celltype, density=False, weights=np.ones_like(all_peaks_locs) / len(all_peaks_locs), histtype='step', linewidth=1)   
        if plot_peak_locs == True:
            axes[1].set_ylabel("Fraction")
            axes[1].set_xlabel('Peak Info. (bits)')
            axes[1].spines['top'].set_visible(False)
            axes[1].spines['right'].set_visible(False)
            # Adding labels for event frames if provided
            if event_frames is not None:
                
                xlim = axes[1].get_xlim()
                for frame in event_frames:
                    if frame < xlim[1]:
                        axes[1].axvline(x=frame, color='k', linestyle=(0, (10.5,6.8)), alpha=1,lw=0.7)
                axes[1].set_xticks(event_frames)
                axes[1].set_xticklabels(self.event_labels)
                plt.xticks(rotation=45)

        #clean up plots
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)
        #had to redo ylims for violin theya re changing based on the histogram...
        if figure_type == 'violin':
            if ylims1[1] < 0.25:  # Check if y-axis limit is greater than 0.25
                axes[0].set_ylim(0, 0.25)  # Setting y-axis limits to match example
            else:
                axes[0].set_ylim(0, ylims1[1]+.03)  # Adjust y-axis limit to be slightly above max value
        # plt.gcf().subplots_adjust(top=0.85)  # Increase space at top for stars

        
        plt.tight_layout()

        #kruskal wallis test across all cell types for values and locs
        kw_table_all = [] 
        # Loop over the two types of data in collected_peaks
        for key in ['peak_values', 'peak_frames']:
            if key == 'peak_frames':
                group1 = group1 / 30
                group2 = group2 / 30
                group3 = group3 / 30

            # Select data for each cell type
            group1 = np.array(collected_peaks['pyr'][key])
            group2 = np.array(collected_peaks['som'][key])
            group3 = np.array(collected_peaks['pv'][key])

            # Optional: remove NaNs
            group1 = group1[~np.isnan(group1)]
            group2 = group2[~np.isnan(group2)]
            group3 = group3[~np.isnan(group3)]

            # call your method
            kw_row = self.stats.kruskal_wallis_to_pd(key, group1, group2, group3)

            # append the returned DataFrame row to a list
            kw_table_all.append(kw_row)

        # concatenate all rows into a single DataFrame
        df_kw = pd.concat(kw_table_all, ignore_index=True)
        # Determine if either Kruskal–Wallis test is significant
        # kw_significant = (df_kw["p_value"] < 0.05).any()
        kw_significant = (
            df_kw.loc[df_kw["Group1"] == "peak_values", "p_value"].iloc[0] < 0.05
        )

        # perform permutation test for each cell type
        # Perform pairwise comparisons
        celltype_keys = list(self.celltypecolors.keys())
        all_p_values = []
        comparisons = []
        comparisons_names = []
        p_values = []
        test_stats = []
        all_stats_dict = {}

        for i, celltype in enumerate(celltype_keys):
            for j in range(i + 1, len(celltype_keys)):
                other_celltype = celltype_keys[j]
                comparisons.append((i, j))
                # Get data for the two cell types
                data_i = np.array(collected_peaks[celltype]['peak_values'])
                data_j = np.array(collected_peaks[other_celltype]['peak_values'])
                if len(data_i) == 0 or len(data_j) == 0:
                    print(f"No significant peaks found for {celltype} or {other_celltype}. Skipping permutation test.")
                    continue
                
                # Perform permutation test
                p_value, stat = self.stats.perform_permutation_test(data_i, data_j, paired=False, n_permutations=10000)
                all_p_values.append(p_value)

                comparisons_names.append((f"{celltype}_peak_vals", f"{other_celltype}_peak_vals"))
                test_stats.append(stat)
                p_values.append(p_value)

                # Save stats for each group
                label1 = f"{celltype}_peakvals"
                label2 = f"{other_celltype}_peakvals"
                all_stats_dict[label1] = self.stats.get_basic_stats(data_i)
                all_stats_dict[label2] = self.stats.get_basic_stats(data_j)

                print(f"Permutation test p-value for {celltype} vs {other_celltype}: {p_value:.4f}")    

        _, significance_stars = self.stats.calculate_bonferroni_significance(all_p_values, alpha=0.05)

        # Add significance stars to the plot
        ylims = axes[0].get_ylim()
        y_range = ylims[1] - ylims[0]
        base_height = ylims[1] - (y_range * 0.1)  # Start stars at 90% of max height
        step_height = y_range * star_height_percentage  # Use parameter to control spacing


        if kw_significant:
            # Add significance stars to the plot
            count = 0
            for (i, j), star in zip(comparisons, significance_stars):
                if star != 'ns':  # Only add significance line if there is a star
                    star_y = ylims[1]-.03+count# base_height + (step_height * count) #
                    self.add_significance_line(axes[0], x1=i, x2=j, y=star_y,significance=star, color='black',star_height_percentage = star_height_percentage, fontsize=7)
                    count += .03


        for i, celltype in enumerate(celltype_keys):
            for j in range(i + 1, len(celltype_keys)):
                other_celltype = celltype_keys[j]
                comparisons.append((i, j))
                # Get data for the two cell types
                data_i = np.array(collected_peaks[celltype]['peak_frames'])/ 30  # convert frames to seconds
                data_j = np.array(collected_peaks[other_celltype]['peak_frames'])/ 30  # convert frames to seconds
                if len(data_i) == 0 or len(data_j) == 0:
                    print(f"No significant peaks found for {celltype} or {other_celltype}. Skipping permutation test.")
                    continue
                
                # Perform permutation test
                p_value, stat = self.stats.perform_permutation_test(data_i, data_j, paired=False, n_permutations=10000)
                all_p_values.append(p_value)

                comparisons_names.append((f"{celltype}_peak_locs", f"{other_celltype}_peak_locs"))
                test_stats.append(stat)
                p_values.append(p_value)

                # Save stats for each group
                label1 = f"{celltype}_peaklocs"
                label2 = f"{other_celltype}_peaklocs"
                all_stats_dict[label1] = self.stats.get_basic_stats(data_i)
                all_stats_dict[label2] = self.stats.get_basic_stats(data_j)

                print(f"Permutation test locs p-value for {celltype} vs {other_celltype}: {p_value:.4f}")

        #redo y limits for axes[0]
        if ylim_axis0 is not None:
            axes[0].set_ylim(ylim_axis0)
        else:   
            max_y = max([max(collected_peaks[ct]['peak_values'] or [0]) for ct in celltype_keys])
            axes[0].set_ylim(0, max(0.25, max_y + 0.03))  # Adjust y-axis limit to be slightly above max value

        #include bin_size in the name of the saved file
        if bin_size is None:
            bin_size = 3
        # get actual save_path by getting the string in front of the last /
        if save_path and '/' in save_path:
            save_path_updated = save_path[:save_path.rfind('/')]
        if save_path:
            plt.savefig(save_path,  dpi=300) #bbox_inches='tight',
            df_tests = self.stats.to_table(comparisons_names, test_stats, p_values, save_path=f'{save_path_updated}/stat_tests_info_neurons_peaks_bin{bin_size}.csv',type='permutation')
            df_tests = pd.concat([df_tests, df_kw], ignore_index=True)
            df_tests.to_csv(f'{save_path_updated}/stat_tests_info_neurons_peaks_bin{bin_size}.csv', index=False)
            df_stats = self.stats.basic_stats_to_table(all_stats_dict, save_path=f'{save_path_updated}/basic_stats_info_neurons_peaks_bin{bin_size}.csv')
        else:
            df_tests = self.stats.to_table(comparisons_names, test_stats, all_p_values,type='permutation')
   
        plt.show()

    def plot_significant_neurons_dataset_means(
        self,
        significant_neurons_data,
        event_frames=None,
        save_path=None,
        figure_type='bar',
        star_height_percentage=0.05
    ):
        """
        Plot the mean (and SEM) of peak values per dataset, per cell type.
        Each point/bar represents the mean peak value for a cell type in a single dataset.
        """

        plt.rcParams.update({'font.size': 7, 'font.family': 'arial'})
        celltypes = list(self.celltypecolors.keys())
        dataset_names = list(significant_neurons_data.keys())

        fig, ax = plt.subplots(1, 2, figsize=(3, 1.6), dpi=300) #, constrained_layout=True
        plt.subplots_adjust(wspace=0.1,left=0.2, top=2)    # Adjust for more space between plots
        
        # Collect means and SEMs for each cell type across datasets
        means_by_celltype = {ct: [] for ct in celltypes}
        sems_by_celltype = {ct: [] for ct in celltypes}

        for ct in celltypes:
            for dataset in dataset_names:
                peaks = np.array(significant_neurons_data[dataset][ct]['peak_values'])
                if len(peaks) > 0:
                    means_by_celltype[ct].append(np.mean(peaks))
                    sems_by_celltype[ct].append(np.std(peaks) / np.sqrt(len(peaks)))
                else:
                    means_by_celltype[ct].append(np.nan)
                    sems_by_celltype[ct].append(np.nan)

        # Convert to arrays for easier handling
        means_arr = np.array([means_by_celltype[ct] for ct in celltypes])
        sems_arr = np.array([sems_by_celltype[ct] for ct in celltypes])

        fig, ax = plt.subplots(figsize=(3, 2), dpi=300)

        # Plotting
        x = np.arange(len(celltypes))
        means = np.nanmean(means_arr, axis=1)
        sems = np.nanstd(means_arr, axis=1) / np.sqrt(np.sum(~np.isnan(means_arr), axis=1))

        if figure_type == 'bar':
            bars = ax.bar(
                x,
                means,
                yerr=sems,
                capsize=5,
                color='white',
                edgecolor=[self.celltypecolors[ct] for ct in celltypes],
                linewidth=2,
            )
            # Overlay individual dataset means as scatter
            for i, ct in enumerate(celltypes):
                yvals = means_by_celltype[ct]
                ax.scatter(
                    np.full(len(yvals), x[i]),
                    yvals,
                    color=self.celltypecolors[ct],
                    alpha=0.7,
                    zorder=10,
                    s=20,
                )
        elif figure_type == 'violin':
            # Prepare data for violin plot
            plot_data = [means_by_celltype[ct] for ct in celltypes]
            sns.violinplot(
                data=plot_data,
                ax=ax,
                palette=[self.celltypecolors[ct] for ct in celltypes],
                inner='box',
                linewidth=1,
            )
            ax.set_xticks(x)
            ax.set_xticklabels([self.cell_type_labels[ct] for ct in celltypes], rotation=45)
            ax.set_ylabel("Dataset Mean Peak Info (bits)")
        elif figure_type == 'scatter':
            for i, ct in enumerate(celltypes):
                yvals = means_by_celltype[ct]
                ax.scatter(
                    np.full(len(yvals), x[i]),
                    yvals,
                    color=self.celltypecolors[ct],
                    alpha=0.7,
                    zorder=10,
                    s=30,
                    label=self.cell_type_labels[ct]
                )
            ax.set_xticks(x)
            ax.set_xticklabels([self.cell_type_labels[ct] for ct in celltypes], rotation=45)
            ax.set_ylabel("Dataset Mean Peak Info (bits)")

        ax.set_xticks(x)
        ax.set_xticklabels([self.cell_type_labels[ct] for ct in celltypes], rotation=45)
        ax.set_ylabel("Dataset Mean Peak Info (bits)")
        ax.set_title("Mean Peak Info per Dataset")

        # Statistical comparisons between cell types (using dataset means)
        all_p_values = []
        comparisons = []
        test_stats = []
        for i in range(len(celltypes)):
            for j in range(i + 1, len(celltypes)):
                data_i = np.array(means_by_celltype[celltypes[i]])
                data_j = np.array(means_by_celltype[celltypes[j]])
                # Remove NaNs
                mask = ~np.isnan(data_i) & ~np.isnan(data_j)
                if np.sum(mask) > 1:
                    p_value, stat = self.stats.perform_permutation_test(
                        data_i[mask], data_j[mask], paired=False, n_permutations=10000
                    )
                    all_p_values.append(p_value)
                    test_stats.append(stat)
                    comparisons.append((i, j))
                else:
                    all_p_values.append(np.nan)
                    test_stats.append(np.nan)
                    comparisons.append((i, j))

        # Bonferroni correction and significance stars
        valid_pvals = [p for p in all_p_values if not np.isnan(p)]
        _, significance_stars = self.stats.calculate_bonferroni_significance(valid_pvals, alpha=0.05)

        # Add significance stars
        count = 0
        ylims = ax.get_ylim()
        for (i, j), star in zip([c for c, p in zip(comparisons, all_p_values) if not np.isnan(p)], significance_stars):
            if star != 'ns':
                star_y = ylims[1] - 0.05 + count
                self.add_significance_line(ax, x1=i, x2=j, y=star_y, significance=star, color='black', star_height_percentage=star_height_percentage)
                count += 0.05

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()

    # def plot_scale_bar(scalebar_length, scalebar_start, ax, scalebar_color='black', scalebar_linewidth=1, frame_rate = 30):
#     """
#     Plots a scale bar on the given axis.

#     Parameters:
#     scalebar_length: length of the scale bar in frames
#     scalebar_start: starting x position of the scale bar
#     ax: matplotlib axis to plot on
#     scalebar_color: color of the scale bar
#     """
#     scalebar_end = scalebar_start + scalebar_length
    
#     ylims = ax.get_ylim()
#     y_range = ylims[1] - ylims[0]
    
#     # place scalebar just below traces (5% of axis range)
#     y_pos = ylims[0] - 0.05 * y_range  
    
#     ax.plot([scalebar_start, scalebar_end],
#             [y_pos, y_pos],
#             color=scalebar_color,
#             linewidth=scalebar_linewidth)

#     # optional label under scalebar
#     ax.text((scalebar_start + scalebar_end) / 2,
#             y_pos - 0.02 * y_range,
#             f"{scalebar_length/frame_rate:.0f} sec",
#             ha='center', va='top', fontsize=6)
    def plot_scale_bar(self,scalebar_length, scalebar_start, ax, 
                    scalebar_color='black', scalebar_linewidth=1, frame_rate=30, xticks=None):
        """
        Plots a scale bar on the given axis without changing y-limits.
        Draws in axis coordinates so it sits below the data.
        """
        # transform scalebar x from data, y from axes
        trans = ax.get_xaxis_transform()  # x in data coords, y in axes coords
        
        # x positions in data coords
        scalebar_end = scalebar_start + scalebar_length

        # y position in axes coords (e.g., -0.08 below the x-axis)
        if xticks is not None:
            y_pos = -0.1  
            ypos_mins = 0.03
        else:
            y_pos = +0.01  
            ypos_mins = 0.02

        # plot scalebar
        ax.plot([scalebar_start, scalebar_end],
                [y_pos, y_pos],
                transform=trans,
                color=scalebar_color,
                linewidth=scalebar_linewidth,
                clip_on=False)  # so it's visible outside axes

        # label
        ax.text((scalebar_start + scalebar_end) / 2,
                y_pos - ypos_mins,   # further down a bit
                f"{scalebar_length/frame_rate:.0f} sec",
                ha='center', va='top', fontsize=6,
                transform=trans,
                clip_on=False)

    def plot_glm_predictors_and_decoder(self,variables, sorted_idx, results_pre, do_highlight_trial=None, save_path=None, figsize = (6, 10 * 0.3), number_subplots = 4, norm_y_pred = False, subplot_4_type = 'traces'):
        mpl.rcParams['pdf.fonttype'] = 42   # TrueType fonts (editable)
        plt.rcParams.update({'font.size': 7, 'font.family': 'arial'})  # Updated font size for clarity
        frames = variables['frames']
        behav_matrix = variables['behav_matrix']
        behav_matrix_ids_raw = variables['behav_matrix_ids_raw']
        behav_big_matrix = variables['behav_big_matrix']
        behav_big_matrix_ids = variables['behav_big_matrix_ids']
        info_data = variables['info_data']
        decoder_singlecell = variables['decoder_singlecell']
        testing_trials_used = variables['testing_trials_used']
        condition_array_trials = variables['condition_array_trials']
        combined_frames_included = variables['combined_frames_included']
        example_dataset = variables['example_dataset']
        fold_number = variables['fold_number']
        model_output_behav = results_pre[example_dataset[0]]['model_output_behav']
        y_pred_model = model_output_behav[fold_number]['y_pred'][frames, :] #frames x Neurons

        # Clean and deduplicate variable names
        flat_ids = [str(var[0]) if isinstance(var, np.ndarray) else str(var) for var in behav_matrix_ids_raw]
        clean_ids = [v.replace('upcoming ', '') for v in flat_ids]
        unique_names = {}
        for idx, name in enumerate(clean_ids):
            if name not in unique_names:
                unique_names[name] = idx
        deduped_names = list(unique_names.keys())
        deduped_indices = list(unique_names.values())

        # Normalize original behavioral matrix
        normalized_matrix = np.zeros_like(behav_matrix, dtype=float)
        for i in range(behav_matrix.shape[0]):
            row = behav_matrix[i]
            min_val = np.min(row)
            max_val = np.max(row)
            normalized_matrix[i] = (row - min_val) / (max_val - min_val) if max_val - min_val != 0 else 0

        # Determine trials included from condition array and frames
        
        _, trials_included, relative_trial_starts = self.glm_data_utils.get_testing_trial_frames(combined_frames_included, frames, condition_array_trials)

        # Determine trials to highlight
        highlight_indices, _ = self.glm_data_utils.get_highlight_trial_indices(trials_included, testing_trials_used)

        # Plotting
        fig, axs = plt.subplots(1, number_subplots, figsize=figsize, sharex=False,gridspec_kw={'width_ratios': [1,1,1,1],'height_ratios': [1],'wspace': .6}) #(6, len(deduped_names) * 0.3)
        offset = 1.3

        # (1) Predictors
        for i, idx in enumerate(deduped_indices):
            axs[0].plot(np.array(frames), normalized_matrix[idx, frames] + i * offset, color='black', linewidth=0.5)
        axs[0].set_yticks(np.arange(len(deduped_names)) * offset)
        axs[0].set_yticklabels(deduped_names, fontsize=7)
        axs[0].set_title("(1) Predictors")
        axs[0].spines['top'].set_visible(False)
        axs[0].spines['right'].set_visible(False)
        axs[0].spines['bottom'].set_visible(False)
        axs[0].spines['left'].set_visible(False)
        axs[0].set_xticks([])
        axs[0].set_xticklabels([])

        # (2) Convolved Predictors
        for i, name in enumerate(deduped_names):
            matches = [j for j, bigname in enumerate(behav_big_matrix_ids) if name in str(bigname[0])]
            for k, idx in enumerate(matches[:]): #matches[:5] to plot 5 convolutions
                axs[1].plot(np.array(frames), behav_big_matrix[idx, frames] + i * offset, alpha=1, linewidth=0.3)
        axs[1].set_yticks(np.arange(len(deduped_names)) * offset)
        axs[1].set_yticklabels([])
        axs[1].set_title("(2) Convolved Predictors")
        axs[1].spines['top'].set_visible(False)
        axs[1].spines['right'].set_visible(False)
        axs[1].spines['bottom'].set_visible(False)
        axs[1].spines['left'].set_visible(False)
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        axs[1].set_xticklabels([])

        # (3) Example predicted weights for multiple neurons (dummy example)
        offset_trace = 0.25 # vertical spacing between neurons
        if norm_y_pred:
            offset_trace = 1.5 # vertical spacing between neurons
        n_neurons = 10
        for i in range(n_neurons):
            trace = y_pred_model[:, sorted_idx[i]]
            if norm_y_pred:
                trace = (trace - np.min(trace)) / (np.max(trace) - np.min(trace))  # Normalize to [0, 1]
            axs[2].plot(np.array(frames), trace + i * offset_trace, linewidth=0.5, color='black')

        axs[2].set_title("(3) Predicted Activity")
        axs[2].set_xticks([])
        axs[2].set_yticks(np.arange(n_neurons) * offset_trace)
        axs[2].set_yticklabels([str(i + 1) for i in range(n_neurons)], fontsize=6)
        axs[2].set_ylabel("Neuron ID", fontsize=7, rotation=90, labelpad=2)
        axs[2].set_xticklabels([])
        axs[2].spines['top'].set_visible(False)
        axs[2].spines['right'].set_visible(False)
        axs[2].spines['bottom'].set_visible(False)
        axs[2].spines['left'].set_visible(False)

        #add optional rectangle to highlight trial
        highlight_trial = highlight_indices # set to None if no highlight desired

        for ax in axs[:do_highlight_trial]:
            if highlight_trial.size > 0:
                # trial indices are 1-based in your description
                trial_idx = highlight_trial[0] - 1  
                x_start = relative_trial_starts[trial_idx] + frames[0]  # adjust for actual frame indices
                if trial_idx < len(relative_trial_starts) - 1:
                    x_end = relative_trial_starts[trial_idx + 1] - 1 + frames[0]  # adjust for actual frame indices
                else:
                    x_end = frames[-1]  # until end of data

                ylims = ax.get_ylim()#axs[2].get_ylim()
                # rectangle spans full y-limits
                rect = patches.Rectangle((x_start, ylims[0]),
                                            x_end - x_start,
                                            ylims[1] - ylims[0],
                                            linewidth=0,
                                            facecolor="gray",
                                            alpha=0.1,
                                            zorder=-1)  # send behind traces
                ax.add_patch(rect)

        ref_pos = axs[0].get_position()
        #add scale bars
        scalebar_length = 30 *5  # in frames(5 seconds at 30 fps)
        scalebar_linewidth = 1.5
        scalebar_start = frames[0] + 10  # small offset from left edge
        for ax in axs[:3]:  # add to first three subplots
            self.plot_scale_bar(scalebar_length, scalebar_start, ax, scalebar_color='black', scalebar_linewidth=scalebar_linewidth , frame_rate = 30)
        
        # (4) Decoder
        offset_decoder = 1.5
        trial_id = np.where(np.isin(testing_trials_used,trials_included))[0]  # Example, adjust as needed
        if trial_id.size > 0 and number_subplots == 4:
            if subplot_4_type == 'traces':
                for i in range(n_neurons):
                    axs[3].plot(np.arange(decoder_singlecell.shape[0]),decoder_singlecell[:, trial_id, sorted_idx[i]] + i * offset_decoder, linewidth=0.5, color='black')

                event_frames = [6, 38, 70, 131, 145]
                event_labels = ['S1', 'S2', 'S3', 'T', 'R']
                for frame, event_label in zip(event_frames, event_labels):
                    axs[3].axvline(x=frame, color='gray', linestyle=(0, (10.5,6.8)), alpha=1,lw=0.7)

                axs[3].set_title('(4) Bayesian Decoder\n(Example Trial)')
                neuron_start_id = 1  # since you’re plotting i+200
                axs[3].set_yticks(np.arange(n_neurons) * offset_decoder ) #*1.15
                axs[3].set_yticklabels([str(i + neuron_start_id) for i in range(n_neurons)], fontsize=6)
                axs[3].set_ylabel("Neuron ID", fontsize=7, rotation=90, labelpad=2)  # <-- adds the extra label
                axs[3].set_xticklabels([])
                axs[3].spines['top'].set_visible(False)
                axs[3].spines['right'].set_visible(False)
                axs[3].spines['bottom'].set_visible(False)
                axs[3].spines['left'].set_visible(False)
                axs[3].set_xticks(event_frames)
                axs[3].set_xticklabels(event_labels, fontsize=6)
                # # Get reference box from subplot 0 (or any other)
                axs[3].set_xlim([0,decoder_singlecell.shape[0]])
                
                pos3 = axs[3].get_position()
                # axs[3].set_position([pos3.x0, ref_pos.y0+0.05, ref_pos.width, ref_pos.height-.05])
                axs[3].set_position([pos3.x0, ref_pos.y0+0.05, ref_pos.width, ref_pos.height-0.05])
                # axs[3].set_position(axs[2].get_position())  # match position of subplot 2
                scalebar_length = 30  # in frames(5 seconds at 30 fps)
                scalebar_linewidth = 1.5
                scalebar_start = 1  # small offset from left edge
                self.plot_scale_bar(scalebar_length, scalebar_start, axs[3], scalebar_color='black', scalebar_linewidth=scalebar_linewidth , frame_rate = 30, xticks=event_frames)
        
            elif subplot_4_type == 'heatmap':
                # Plot heatmap of decoder activity for all trials
                decoded_per_trial = []
                neuron_start_id = 1
                for i in range(n_neurons):
                    trial_data = decoder_singlecell[:, trial_id, sorted_idx[i]]  # shape: (frames, trials, neurons)
                    trial_value = (trial_data.mean(axis=0) > 0.5).astype(int)
                    decoded_per_trial.append(trial_value)
                im = axs[3].imshow(decoded_per_trial, aspect='auto', cmap='binary', interpolation='nearest')
                axs[3].set_title('(4) Bayesian Decoder\n(Example Trial)')
                axs[3].set_yticks(np.arange(n_neurons)) 
                axs[3].set_yticklabels([str(i + neuron_start_id) for i in range(n_neurons)], fontsize=6)
                axs[3].set_xticklabels([])
                axs[3].set_xticks([])
                axs[3].set_ylabel("Neuron ID", fontsize=7, rotation=90, labelpad=2)  # <-- adds the extra label

        # Save or show
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        # plt.show()

    def plot_scatter_all_datasets_by_celltype(self,
            peak_info_struc,
            celltypes=("Pyr","SOM","PV"),
            stim_feature="sound_category",
            choice_feature="choice",
            threshold=0.06,
            figsize=(9,3),
            colors=None,
            lims=(0, 0.3),
            subplots = False, save_path = None, uniformative_colors = None):

        """
        Plot scatter plot of peak information for all datasets, by cell type.
        """
        mpl.rcParams['pdf.fonttype'] = 42   # TrueType fonts (editable)
        plt.rcParams.update({'font.size': 7, 'font.family': 'arial'})
        datasets = list(peak_info_struc.keys())

        # dataset colors (if none passed)
        if colors is None:
            
            cmap = cm.get_cmap('tab10', len(datasets))
            colors = {ds: cmap(i) for i, ds in enumerate(datasets)}

        # =====================================================================
        # CASE 1: multiple cell types → separate subplots
        # =====================================================================
        if subplots:
            fig, axes = plt.subplots(1, len(celltypes), figsize=figsize)

            for i, ct in enumerate(celltypes):
                ax = axes[i]
                for ds in datasets:
                    if ct not in peak_info_struc[ds]:
                        continue

                    stim_vals   = peak_info_struc[ds][ct][stim_feature]["peak_values"]
                    choice_vals = peak_info_struc[ds][ct][choice_feature]["peak_values"]

                    mask = (stim_vals < threshold) & (choice_vals < threshold)
                    mask_informative = (stim_vals >= threshold) | (choice_vals >= threshold)


                    # uniformative neurons
                    if uniformative_colors is not None:
                        ax.scatter(
                            stim_vals[mask], choice_vals[mask],
                            alpha=0.6, s=2, #s=5
                            edgecolors=uniformative_colors,
                            facecolors='none',
                            linewidths=0.5
                        )

                        ax.scatter(
                            stim_vals[mask_informative], choice_vals[mask_informative],
                            alpha=0.6, s=2, #s=5
                            edgecolors=self.default_colors[ct.lower()],
                            facecolors='none',
                            linewidths=0.5,
                            label=ct if ds == datasets[0] else None
                        )

                    else:
                        ax.scatter(
                        stim_vals, choice_vals,
                        alpha=0.7, s=2, 
                        edgecolors=self.default_colors[ct.lower()],
                        facecolors='none',
                        linewidths=0.5,
                        label=ct if ds == datasets[0] else None
                    )

                ax.axvline(threshold, ls="--", color="black", lw=0.75)
                ax.axhline(threshold, ls="--", color="black", lw=0.75)

                ax.set_title(ct)
                ax.set_xlim(lims)
                ax.set_ylim(lims)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

                if i == 0:
                    ax.set_ylabel(f"Peak {choice_feature} info (bits)")

            fig.tight_layout()
            fig.supxlabel(f"Peak {stim_feature} info (bits)", y=0.0)
            if save_path:
                plt.savefig(save_path,  dpi=300) #bbox_inches='tight',
            plt.show()

        # =====================================================================
        # CASE 2: single cell type input → plot ALL cell types on one axis
        # =====================================================================

        else:
            # Determine all available cell types
            # all_celltypes = sorted({
            #     ct for ds in datasets for ct in peak_info_struc[ds].keys()
            #     if isinstance(peak_info_struc[ds][ct], dict)
            # })
            

            fig, ax = plt.subplots(figsize=figsize)

            for ct in celltypes:
                for ds in datasets:
                    if ct not in peak_info_struc[ds]:
                        continue

                    stim_vals   = peak_info_struc[ds][ct][stim_feature]["peak_values"]
                    choice_vals = peak_info_struc[ds][ct][choice_feature]["peak_values"]

                    mask = (stim_vals < threshold) & (choice_vals < threshold)
                    mask_informative = (stim_vals >= threshold) | (choice_vals >= threshold)


                    # uniformative neurons
                    if uniformative_colors is not None:
                        ax.scatter(
                            stim_vals[mask], choice_vals[mask],
                            alpha=0.6, s=2, #s=5
                            edgecolors=uniformative_colors,
                            facecolors='none',
                            linewidths=0.5
                        )

                        ax.scatter(
                            stim_vals[mask_informative], choice_vals[mask_informative],
                            alpha=0.6, s=2, #s=5
                            edgecolors=self.default_colors[ct.lower()],
                            facecolors='none',
                            linewidths=0.5,
                            label=ct if ds == datasets[0] else None
                        )

                    else:
                        ax.scatter(
                        stim_vals, choice_vals,
                        alpha=0.7, s=2, 
                        edgecolors=self.default_colors[ct.lower()],
                        facecolors='none',
                        linewidths=0.5,
                        label=ct if ds == datasets[0] else None
                    )

            ax.axvline(threshold, ls="--", color="black", lw=0.75)
            ax.axhline(threshold, ls="--", color="black", lw=0.75)

            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_xlabel(f"Peak {stim_feature.replace('_', ' ')}\ninfo (bits)") #set_labels = (f1.replace("_", " ").title(), f2.replace("_", " ").title()), 
            ax.set_ylabel(f"Peak {choice_feature.replace('_', ' ')}\ninfo (bits)")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # legend sorted by input ordering
            handles, labels = ax.get_legend_handles_labels()
            ordered = [ct for ct in celltypes if ct in labels]

            ax.legend(
                [handles[labels.index(ct)] for ct in ordered],
                ordered,
                frameon=False,
                fontsize=6,
                handletextpad=0.2,
                loc="upper right"
            )

            plt.tight_layout()
            if save_path:
                plt.savefig(save_path,  dpi=300) #bbox_inches='tight',

            plt.show()

    def plot_informative_pie(self,info_fracs, f1,f2,figsize = (1.5,1.5), colors = ["black", "lightgray"],save_path = None):
        '        Plot a pie chart showing the fraction of informative neurons per feature.'
        
        mpl.rcParams['pdf.fonttype'] = 42   # TrueType fonts (editable)
        mean_val = np.mean(info_fracs)
        sd_val   = np.std(info_fracs)

        f1_label = f1.replace("_", " ").title()
        f2_label = f2.replace("_", " ").title()

        plt.figure(figsize=figsize)
        plt.pie([mean_val, 100-mean_val],
                colors=colors,
                startangle=90,
                
                wedgeprops={'edgecolor':'none'})
        plt.title(f"Informative Neurons\n{mean_val:.0f}% ± {sd_val:.0f}%", fontsize = 7) #{f1_label} ∩ {f2_label}\n

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path,  dpi=300) #bbox_inches='tight',
        plt.show()

    def plot_overlap_pie(self,
        venn_f1,
        venn_f2,
        venn_both,
        f1,
        f2,
        figsize=(1.6, 1.6),
        colors=None,
        save_path=None,
        radius = 1.15
    ):
        """
        Pie chart showing partition of informative neurons.

        Categories:
        - f1 only
        - f2 only
        - both
        """
        mpl.rcParams['pdf.fonttype'] = 42   # TrueType fonts (editable)
        # Mean ± SD
        m1, s1 = np.mean(venn_f1), np.std(venn_f1)
        m2, s2 = np.mean(venn_f2), np.std(venn_f2)
        mb, sb = np.mean(venn_both), np.std(venn_both)

        # Default muted colors (can be overridden)
        if colors is None:
            c1 = "#4B3F92"   # muted purple
            c2 = "#F2B134"   # muted yellow
            cb = "#6EC5FF"   # soft blue
        else:
            c1, c2, cb = colors

        fig, ax = plt.subplots(figsize=figsize)

        vals = [m1, mb, m2]

        wedges, _ = ax.pie(
            vals,
            colors=[c1, cb, c2],
            startangle=90,
            counterclock=False,
            wedgeprops=dict(edgecolor='white', linewidth=0.6)
        )

        # Percentage labels ONLY (outside)
        pct_labels = [
            f"{m1:.0f} ± {s1:.0f}%",
            f"{mb:.0f} ± {sb:.0f}%",
            f"{m2:.0f} ± {s2:.0f}%"
        ]

        for w, label in zip(wedges, pct_labels):
            ang = (w.theta2 + w.theta1) / 2
            x = radius * np.cos(np.deg2rad(ang))
            y = radius * np.sin(np.deg2rad(ang))
            ax.text(x, y, label, ha='center', va='center', fontsize=6)

        # Legend (clean, journal style)
        legend_labels = [
            f"{f1.replace('_', ' ').title()} only",
            "Both",
            f"{f2.replace('_', ' ').title()} only"
        ]

        ax.legend(
            wedges,
            legend_labels,
            frameon=False,
            fontsize=6,
            loc="lower center",
            bbox_to_anchor=(0.5, -.1),
            bbox_transform=fig.transFigure,
            ncol=1
        )

        ax.set_title(
            "Proportion of\ninformative neurons",
            fontsize=7,
            pad=6
        )

        ax.axis("equal")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, format="pdf")

        plt.show()

    def plot_overlap_stacked(self,
        venn_f1,
        venn_f2,
        venn_both,
        f1,
        f2,
        figsize=(1.2, 2.0),
        colors=None,
        show_error=True,
        save_path=None
    ):
        """
        Stacked bar plot for overlap between two informative features.

        venn_f1, venn_f2, venn_both : arrays (percentages per dataset)
        f1, f2 : feature names (strings)
        """
        mpl.rcParams['pdf.fonttype'] = 42   # TrueType fonts (editable)
        # Mean ± SD
        m1, s1 = np.mean(venn_f1), np.std(venn_f1)
        m2, s2 = np.mean(venn_f2), np.std(venn_f2)
        mb, sb = np.mean(venn_both), np.std(venn_both)

        # Colors
        if colors is None:
            c1 = "#87A96B"   # soft green (f1)
            cb = "#9E9E9E"   # gray (both)
            c2 = "#7FA6D6"   # soft blue (f2)
        else:
            c1, c2 = colors
            cb = (0.5, 0.5, 0.5)

        fig, ax = plt.subplots(figsize=figsize)

        # Bottoms for stacking
        bottom_both = m1
        bottom_f2 = m1 + mb

        # Bars
        ax.bar(0, m1, color=c1, width=0.6, label=f"{f1} only")
        ax.bar(0, mb, bottom=bottom_both, color=cb, width=0.6, label="Both")
        ax.bar(0, m2, bottom=bottom_f2, color=c2, width=0.6, label=f"{f2} only")

        # Optional error bars (shown at segment centers)
        if show_error:
            ax.errorbar(
                [0], [m1 / 2],
                yerr=[s1],
                fmt='none', ecolor='k', capsize=2, lw=0.8
            )
            ax.errorbar(
                [0], [bottom_both + mb / 2],
                yerr=[sb],
                fmt='none', ecolor='k', capsize=2, lw=0.8
            )
            ax.errorbar(
                [0], [bottom_f2 + m2 / 2],
                yerr=[s2],
                fmt='none', ecolor='k', capsize=2, lw=0.8
            )

        # Text labels inside bars
        ax.text(0, m1 / 2, f"{m1:.0f}%\n±{s1:.0f}", ha='center', va='center', fontsize=7)
        ax.text(0, bottom_both + mb / 2, f"{mb:.0f}%\n±{sb:.0f}", ha='center', va='center', fontsize=7)
        ax.text(0, bottom_f2 + m2 / 2, f"{m2:.0f}%\n±{s2:.0f}", ha='center', va='center', fontsize=7)

        # Formatting
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(0, 100)
        ax.set_xticks([])
        ax.set_ylabel("Fraction of informative\nneurons (%)", fontsize=7)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Legend (small, clean)
        ax.legend(
            frameon=False,
            fontsize=6,
            loc="upper right",
            bbox_to_anchor=(0.5, 1.15),
            ncol=1
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)

        plt.show()

    def plot_overlap(self, venn_f1, venn_f2, venn_both, f1, f2, mode="venn", figsize = (1.5,1.5),colors = None, save_path = None):
        """
        venn_f1, venn_f2, venn_both : arrays (per dataset)
        f1, f2 : feature names
        mode : "pie" or "venn"
        """
        mpl.rcParams['pdf.fonttype'] = 42   # TrueType fonts (editable)
        # Mean ± SD
        m1, s1 = np.mean(venn_f1), np.std(venn_f1)
        m2, s2 = np.mean(venn_f2), np.std(venn_f2)
        mb, sb = np.mean(venn_both), np.std(venn_both)

        # Colors
        if colors is None:
            c1 = "#87A96B"   # soft green
            c2 = "#7FA6D6"   # soft blue
            cb = "#C58882"   # muted red
        else:
            c1 = colors[0]
            c2 = colors[1]
            cb = (0.5,0.5,0.5)

        if mode == "pie":
            # --- PIE VERSION (fast + simple) ---
            plt.figure(figsize=figsize)
            vals = [m1, m2, mb]
            labels = [
                f"{f1} only\n{m1:.0f}% ± {s1:.0f}",
                f"{f2} only\n{m2:.0f}% ± {s2:.0f}",
                f"Both\n{mb:.0f}% ± {sb:.0f}"
            ]
            plt.pie(vals, labels=labels, colors=[c1,c2,cb],
                    startangle=90, wedgeprops={'edgecolor':'k'})
            # plt.title(f"{f1} ∩ {f2}")
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=300) #bbox_inches='tight',
            plt.show()
        else:
        # --- VENN VERSION (overlapping circles) ---
            plt.figure(figsize=figsize)
            ax = plt.gca()

            v = venn2(subsets = (m1, m2, mb),
                set_labels = (f1.replace("_", " ").title(), f2.replace("_", " ").title()), 
                set_colors = colors,
                alpha=0.6)

            # Override labels with formatted mean ± SD
            if v.get_label_by_id('10'):   # feature 1 only
                v.get_label_by_id('10').set_text(f"{m1:.0f}%\n± {s1:.0f}")

            if v.get_label_by_id('01'):   # feature 2 only
                v.get_label_by_id('01').set_text(f"{m2:.0f}%\n± {s2:.0f}")

            if v.get_label_by_id('11'):   # overlap
                v.get_label_by_id('11').set_text(f"{mb:.0f}%\n± {sb:.0f}")

            for text in v.set_labels:
                text.set_fontsize(7)

            plt.tight_layout()
            if save_path:
                plt.savefig(save_path,  dpi=300) #bbox_inches='tight',
            plt.show()

    def plot_synergy_vs_peak_pooled(self,pooled_scatter,celltypes=("Pyr","SOM","PV"), figsize=(6,5), threshold = None, xlims=None, feature_names = None, colors = None, save_path = None):
    
        mpl.rcParams['pdf.fonttype'] = 42   # TrueType fonts (editable)
        fig, axes = plt.subplots(1, len(celltypes), figsize=figsize) #, sharex=True, sharey=True
        plt.rcParams.update({'font.size': 7, 'font.family': 'arial'})

        # default: use the keys from pooled_scatter for the first cell type
        if feature_names is None:
            # assume all feature keys except "synergy"
            example_cell = celltypes[0]
            feature_names = [k for k in pooled_scatter[example_cell].keys()
                            if k != "synergy"]
            
        # set colors for features
        color_cycle = ('red','blue') #plt.cm.tab10(np.linspace(0, 1, len(feature_names)))
        if colors is not None:
            color_cycle = colors
            
        # ensure axes is always iterable
        if len(celltypes) == 1:
            axes = np.array([axes])

        for i, celltype in enumerate(celltypes):
            ax = axes[i]

            for f, color in zip(feature_names, color_cycle):
                ax.scatter(
                    pooled_scatter[celltype][f],
                    pooled_scatter[celltype]["synergy"],
                    alpha=0.6, s=5,
                    label= f.replace("_", " ").title(),#f"{f}",
                    facecolors='none',
                    edgecolor=color,
                )

            if threshold is not None:
                ax.axvline(threshold, linestyle="--", color="black", linewidth=0.75)
            if xlims is not None:
                ax.set_xlim(xlims)
            # ax.set_xlabel("Peak information (bits)")
            # ax.set_ylabel("Synergy index")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            # ax.set_title(f"{celltype}")
            ax.set_title(celltype.upper())
            if feature_names is not None and i == len(celltypes) - 1:
                ax.legend( fontsize=6, frameon=False, loc='upper right', bbox_to_anchor=(1.2, 1),handletextpad=0.2)

            if i == 0:
                ax.set_ylabel("Synergy index")
            # ax.set_xlabel(f"Peak {stim_feature} info (bits)")
            ax.set_yticks([0, 0.5, 1])

        fig.tight_layout()
        fig.supxlabel(f"Peak information (bits)", y=0.0, fontsize=7)
        if save_path:
            plt.savefig(save_path,  dpi=300) #bbox_inches='tight',
        fig.show()

    def plot_synergy_violin(self,synergy_all,
                        celltypes=["Pyr","SOM","PV"],
                        figsize=(6,5),
                        colors=None, save_path = None):
        """
        Violin plots of synergy index for each cell type
        """
        mpl.rcParams['pdf.fonttype'] = 42   # TrueType fonts (editable)
        fig ,axes = plt.subplots(figsize=figsize)
        plt.rcParams.update({'font.size': 7, 'font.family': 'arial'})

        data = []
        labels = []

        for ct in celltypes:
            data.append(synergy_all[ct])
            labels.append(ct)

        if colors is None:
            colors = sns.color_palette("Set2", len(celltypes))
        sns.violinplot(data=data, inner="box", cut=0, palette=colors,linewidth=1) #inner='box',
                        
        plt.xticks(range(len(labels)), labels)
        plt.ylabel("Synergy Index")
        # plt.title("Synergy between Sound and Choice Information", fontsize=7)
        # plt.axhline(0, linestyle="--", color="gray")
        
        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path,  dpi=300) #bbox_inches='tight',
        plt.show()

    def plot_scatter_plot_weights_overlay_noerrorn(self,synergy_all,
                        celltypes=["Pyr","SOM","PV"],
                        figsize=(6,5),
                        colors=None, save_path = None, ylabel = "Synergy Index"):
        """
        Violin plots of index for each cell type
        """
        mpl.rcParams['pdf.fonttype'] = 42   # TrueType fonts (editable)
        fig ,axes = plt.subplots(figsize=figsize)
        plt.rcParams.update({'font.size': 7, 'font.family': 'arial'})

        data = []
        labels = []

        for ct in celltypes:
            data.append(synergy_all[ct])
            labels.append(ct)

        if colors is None:
            colors = sns.color_palette("Set2", len(celltypes))
        sns.violinplot(data=data, inner="box", cut=0, palette=colors,linewidth=1) #inner='box',
                        
        plt.xticks(range(len(labels)), labels)
        plt.ylabel(ylabel)
        # plt.title("Synergy between Sound and Choice Information", fontsize=7)
        # plt.axhline(0, linestyle="--", color="gray")
        
        axes.spines['top'].set_visible(False)
        axes.spines['right'].set_visible(False)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path,  dpi=300) #bbox_inches='tight',
        plt.show()

    def plot_avg_predictors_by_condition(self,
                                        avg_results: dict,
                                        dataset_key: str,
                                        title_prefix: str = '',
                                        colors: list = None,
                                        save_path: str = None,
                                        ylims: tuple = None,
                                        celltype=None,
                                        figsize: tuple = (6,6)):
        """
        Plot average predictors per condition (mean ± SEM) for a given dataset.

        celltype:
            None                → plot all celltypes
            'pyr'               → single celltype
            ['pyr', 'som']      → multiple celltypes
            {'pyr', 'pv'}       → multiple celltypes
        """
        mpl.rcParams['pdf.fonttype'] = 42   # TrueType fonts (editable)
        data = avg_results[dataset_key]
        labels = data['labels']
        mean_list = data['mean']
        sem_list = data['sem']

        # Mapping from celltype → factor indices
        celltype_to_idx = {
            'pyr': slice(0, 3),
            'som': slice(3, 6),
            'pv': slice(6, 9)
        }

        # ---------- Normalize celltype input ----------
        if celltype is None:
            celltypes = list(celltype_to_idx.keys())
        elif isinstance(celltype, str):
            celltypes = [celltype.lower()]
        elif isinstance(celltype, (list, tuple, set)):
            celltypes = [ct.lower() for ct in celltype]
        else:
            raise TypeError(
                "celltype must be None, a string, or a list/tuple/set of strings"
            )

        invalid = [ct for ct in celltypes if ct not in celltype_to_idx]
        if invalid:
            raise ValueError(f"Invalid celltype(s): {invalid}")

        # ---------- Build factor indices & labels ----------
        factor_indices = []
        factor_labels = []
        for ct in celltypes:
            sl = celltype_to_idx[ct]
            idxs = list(range(sl.start, sl.stop))
            factor_indices.extend(idxs)
            factor_labels.extend([ct.upper()] * len(idxs))

        time_axis = np.arange(mean_list[0].shape[1])
        n_conditions = len(labels)

        # Default colors
        if colors is None:
            colors = self.celltypecolors

        plt.figure(figsize=(figsize[0] * n_conditions, figsize[1]))
        plt.rcParams.update({'font.size': 7, 'font.family': 'arial'})

        # ---------- Plot ----------
        for i, (label, mean_vals, sem_vals) in enumerate(zip(labels, mean_list, sem_list)):
            ax = plt.subplot(1, n_conditions, i + 1)

            mean_vals = mean_vals[factor_indices, :]
            sem_vals = sem_vals[factor_indices, :]

            for f_idx in range(mean_vals.shape[0]):
                ct_label = factor_labels[f_idx]
                color = colors[f_idx % len(colors)]

                plt.plot(
                    time_axis,
                    mean_vals[f_idx, :],
                    label=f'{ct_label} {f_idx + 1}',
                    color=color,
                    linewidth=1
                )

                plt.fill_between(
                    time_axis,
                    mean_vals[f_idx, :] - sem_vals[f_idx, :],
                    mean_vals[f_idx, :] + sem_vals[f_idx, :],
                    alpha=0.2,
                    color=color
                )

            # plt.xlabel('Frames Relative to Alignment')
            if i == 0:
                plt.ylabel('Avg. Coupling Predictor')
            plt.title(f"{title_prefix}{label}")
            # plt.legend(frameon=False)

            # Global legend (only once, for all lines) to the rightmost subplot

            handles, labels_legend = ax.get_legend_handles_labels()
            if i == n_conditions - 1:
                fig = plt.gcf()
                fig.legend(
                    handles,
                    labels_legend,
                    loc='center left',
                    bbox_to_anchor=(1.01, 0.5),  # position to the right of all subplots
                    frameon=False
                    )
            


            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_box_aspect(1)

            if ylims is not None:
                ax.set_ylim(ylims)

            #hide choice concatenation
            choice_concat_frame, reward_concat_frame = 101,144
            ax.axvline(x=choice_concat_frame-1, color='w', linestyle='-', alpha=1, linewidth = 3)
            # ax.axvline(x=choice_concat_frame-1, color='w', linestyle='-', alpha=1, linewidth = 2)
            ax.axvline(x=choice_concat_frame, color='w', linestyle='-', alpha=1, linewidth = 2)
            ax.axvline(x=reward_concat_frame, color='w', linestyle='-', alpha=1, linewidth = 2)
            ax.axvline(x=reward_concat_frame-1, color='w', linestyle='-', alpha=1, linewidth = 3)
            #had to add some more bc they were not thick enough...

            for frame, event_label in zip(self.event_frames, self.event_labels):
                ax.axvline(x=frame, color='k', linestyle='--', alpha=0.5, ymin=0.05, ymax=0.95)#linestyle='--', alpha=0.5)

            ax.set_xticks(self.event_frames)
            ax.set_xticklabels(self.event_labels)

        # plt.figure(figsize=(figsize[0] * n_conditions + 1.5, figsize[1]))
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path,  dpi=300)
        plt.show()

    def bar_plot_avg_predictor_intervals(self,
                                     avg_results: dict,
                                     dataset_key: str,
                                     factors,
                                     factor_labels=None,
                                     colors=None,
                                     bar_width=0.25,
                                     ylims=None,
                                     save_path=None,
                                     figsize=(2, 2)):
        """
        Bar plot of average predictors per event (all events shown).

        Parameters
        ----------
        avg_results : dict
            Output of average_folds_by_condition
        dataset_key : str
            Dataset key inside avg_results
        factors : list of int
            Factor indices to plot (e.g. [0,1,2] or [0,9])
        factor_labels : list of str, optional
            Labels for each factor (same length as factors)
        colors : list, optional
            Colors for factors
        bar_width : float
            Width of individual bars
        ylims : tuple, optional
            Y-axis limits
        save_path : str, optional
            Save location
        figsize : tuple
            Figure size
        """

        mpl.rcParams['pdf.fonttype'] = 42
        plt.rcParams.update({'font.size': 7, 'font.family': 'arial'})

        data = avg_results[dataset_key]

        # Expect shape: (n_vars, n_events)
        mean_vals = data['mean'][0]
        sem_vals  = data['sem'][0]


        mean_vals = mean_vals[factors, :]
        sem_vals = sem_vals[factors, :]

        n_factors, n_events = mean_vals.shape

        # Default factor labels
        if factor_labels is None:
            factor_labels = [f'Factor {f}' for f in factors]

        # Default colors
        if colors is None:
            colors = self.celltypecolors
        colors = colors[:n_factors]

        fig, ax = plt.subplots(figsize=figsize)

        x = np.arange(n_events)

        # Center bars within each event
        offsets = (np.arange(n_factors) - (n_factors - 1) / 2) * bar_width

        for f_idx in range(n_factors):
            ax.bar(
                x + offsets[f_idx],
                mean_vals[f_idx, :],
                yerr=sem_vals[f_idx, :],
                width=bar_width,
                color=colors[f_idx],#'white',
                edgecolor='white',#colors[f_idx],
                linewidth=1.0,
                capsize=1,
                label=factor_labels[f_idx],
                error_kw={'ecolor': 'black','capthick': 1, 'elinewidth': 1} #colors[f_idx]
            )

        ax.set_xticks(x)
        ax.set_xticklabels(self.event_labels)
        ax.set_ylabel('Avg. Coupling Predictor')

        if ylims is not None:
            ax.set_ylim(ylims)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # ax.legend(frameon=False)

        # Global legend (only once, for all lines)
        handles, labels_legend = ax.get_legend_handles_labels()
        fig = plt.gcf()
        fig.legend(
            handles,
            labels_legend,
            loc='center left',
            bbox_to_anchor=(1.01, 0.5),  # position to the right of all subplots
            frameon=False
            )
        # plt.figure(figsize=(figsize[0] * 1 + 1.5, figsize[1]))

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path,  dpi=300)
        plt.show()

    def bar_box_plot_avg_predictor_intervals(self,
                                         avg_results: dict,
                                         factors,
                                         factor_labels=None,
                                         colors=None,
                                         bar_width=0.25,
                                         ylims=None,
                                         save_path=None,
                                         figsize=(2, 2),
                                         plot_type='bar'):
        """
        Plot average predictors per event, either as bar plot (mean ± SEM) or boxplot (per-dataset means).

        Parameters
        ----------
        avg_results : dict
            Output of match_and_aggregate_factors
        factors : list of int
            Indices of factors to plot
        factor_labels : list of str, optional
            Labels for each factor (must match length of `factors`)
        colors : list, optional
            Colors for each factor
        bar_width : float
            Width of individual bars
        ylims : tuple, optional
            Y-axis limits
        save_path : str, optional
            Path to save figure
        figsize : tuple
            Figure size
        plot_type : str
            'bar' or 'box'
        """

        mpl.rcParams['pdf.fonttype'] = 42
        plt.rcParams.update({'font.size': 7, 'font.family': 'arial'})

        # Default labels/colors
        if factor_labels is None:
            factor_labels = [f'Factor {f}' for f in factors]

        if colors is None:
            colors = self.celltypecolors
        colors = colors[:len(factors)]

        fig, ax = plt.subplots(figsize=figsize)

        # Prepare bar data
        if plot_type == 'bar':
            data = avg_results['all_datasets']
            mean_vals = data['interval_mean'][0][factors, :]  # shape: (n_factors, n_events)
            sem_vals  = data['interval_sem'][0][factors, :]

            n_events = mean_vals.shape[1]
            x = np.arange(n_events)
            offsets = (np.arange(len(factors)) - (len(factors) - 1) / 2) * bar_width

            for f_idx in range(len(factors)):
                ax.bar(
                    x + offsets[f_idx],
                    mean_vals[f_idx, :],
                    yerr=sem_vals[f_idx, :],
                    width=bar_width,
                    color=colors[f_idx],
                    edgecolor='white',
                    linewidth=1.0,
                    capsize=1,
                    label=factor_labels[f_idx],
                    error_kw={'ecolor': 'black', 'capthick': 1, 'elinewidth': 1}
                )

        # Prepare boxplot data
        elif plot_type == 'box':
            dataset_keys = [k for k in avg_results if k != 'all_datasets']
            n_events = avg_results[dataset_keys[0]]['interval_mean'][0].shape[1]
            x = np.arange(n_events)
            width = bar_width#0.8 / len(factors)
            offsets = (np.arange(len(factors)) - (len(factors) - 1) / 2) * width

            for f_idx, factor in enumerate(factors):
                for ev_idx in range(n_events):
                    data_points = []
                    for dataset_key in dataset_keys:
                        interval_mean = avg_results[dataset_key]['interval_mean'][0]  # (n_factors, n_events)
                        data_points.append(interval_mean[factor, ev_idx])
                    # Plot box for this factor × event
                    ax.boxplot(
                        data_points,
                        positions=[x[ev_idx] + offsets[f_idx]],
                        widths=width,
                        patch_artist=True,
                        boxprops=dict(facecolor=colors[f_idx], linewidth=1),
                        medianprops=dict(color='black'),
                        whiskerprops=dict(linewidth=1),
                        capprops=dict(linewidth=1),
                        flierprops=dict(marker='o', markersize=2, alpha=0.3)
                    )

        else:
            raise ValueError("plot_type must be 'bar' or 'box'")

        # Final touches
        ax.set_xticks(x)
        ax.set_xticklabels(self.event_labels)
        ax.set_ylabel('Avg. Coupling Predictor')

        if ylims is not None:
            ax.set_ylim(ylims)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if plot_type == 'bar':
            handles, labels_legend = ax.get_legend_handles_labels()
            fig.legend(
                handles,
                labels_legend,
                loc='center left',
                bbox_to_anchor=(1.01, 0.5),
                frameon=False
            )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_within_between_scatter(self,all_df, group_colors=None, title='Within vs. Between Coupling', figsize=(4, 4), save_path=None, mode = 'mean',x_ylim=None):

        plt.figure(figsize=figsize)
        mpl.rcParams['pdf.fonttype'] = 42
        plt.rcParams.update({'font.size': 7, 'font.family': 'arial'})

        for group in all_df['group'].unique():
            subset = all_df[all_df['group'] == group]
            plt.scatter(
                subset['coupling_within'],
                subset['coupling_between'],
                edgecolor=group_colors.get(group, 'gray'),
                facecolors='none',
                label=group,
                alpha=1,
                s=10,
                linewidths=1
            )

        min_val = min(all_df['coupling_within'].min(), all_df['coupling_between'].min())
        max_val = max(all_df['coupling_within'].max(), all_df['coupling_between'].max())
        if x_ylim is not None:
            plt.ylim(x_ylim)
            plt.xlim(x_ylim)
            min_val, max_val = x_ylim

        plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1)

        plt.xlabel('Coupling to Same Group')
        plt.ylabel('Coupling to Other Group(s)')
        if 'abs' in mode:
            plt.xlabel('|Coupling to Same Group|')
            plt.ylabel('|Coupling to Other Group(s)|')
            
        plt.grid(True, alpha=0.3)
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.title(title, fontsize=7)
        plt.legend(frameon=False)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()

    def plot_quadrant_heatmap_across_datasets(self,
        combined_df,
        groups=('sound', 'opto'),
        dataset_col='dataset',
        group_col='group',
        save_dir=None,
        figsize=(3, 3),
        decimal_places = 0,
        vmax = 100,
        colormap='coolwarm',
        string = None
    ):
        mpl.rcParams['pdf.fonttype'] = 42
        plt.rcParams.update({'font.size': 7, 'font.family': 'arial'})

        # quadrant_labels = np.array([["+/+", "+/–"], ["–/+", "–/–"]])
        # quadrant_names = ['+/+', '+/–', '–/+', '–/–']
        quadrant_labels = np.array([["+/–", "+/+"],
                            ["–/–", "–/+"]])

        quadrant_names = ['+/–', '+/+', '–/–', '–/+']

        
        #total groups
        ntotal = len(groups)
        # Store per-group stats
        all_group_stats = {}

        for group in groups:
            fractions = []
            raw_counts = []

            for dataset in combined_df[dataset_col].unique():
                subset = combined_df[
                    (combined_df[group_col] == group) & (combined_df[dataset_col] == dataset)
                ]
                sign_within = np.sign(subset['coupling_within'].values)
                sign_between = np.sign(subset['coupling_between'].values)

                quadrant_counts = np.zeros((2, 2), dtype=int)
                # for s_w, s_b in zip(sign_within, sign_between):
                #     row = 0 if s_w > 0 else 1
                #     col = 0 if s_b > 0 else 1
                #     quadrant_counts[row, col] += 1
                for s_w, s_b in zip(sign_within, sign_between):
                    # row = BETWEEN sign (y-axis)
                    row = 0 if s_b > 0 else 1
                    # col = WITHIN sign (x-axis)
                    col = 0 if s_w < 0 else 1
                    quadrant_counts[row, col] += 1


                # Normalize to get fractions
                total = quadrant_counts.sum()
                if total == 0:
                    continue
                fractions.append(quadrant_counts.flatten() / total)
                raw_counts.append(quadrant_counts.flatten())

            fractions = np.array(fractions) *100  # Convert to percentages
            mean_frac = np.nanmean(fractions, axis=0)
            std_frac = np.nanstd(fractions, axis=0)

            all_group_stats[group] = {
                'mean': mean_frac,
                'std': std_frac,
                'raw_counts': np.array(raw_counts)
            }

            # Plot heatmap of mean fractions
            mean_frac_matrix = mean_frac.reshape(2, 2)
            label_matrix = np.array([[f"{mean_frac_matrix[i,j]:.{decimal_places}f} ± {std_frac.reshape(2,2)[i,j]:.{decimal_places}f}\n{quadrant_labels[i,j]}"
                                    for j in range(2)] for i in range(2)])

            plt.figure(figsize=figsize)
            # sns.heatmap(mean_frac_matrix, annot=label_matrix, fmt='', cmap=colormap, cbar=False,
            #             xticklabels=["Between +", "Between –"], yticklabels=["Within +", "Within –"],vmin=-10, vmax=vmax)
            sns.heatmap(mean_frac_matrix, annot=label_matrix, fmt='', cmap=colormap, cbar=False,
                        xticklabels=["Within –", "Within +"], yticklabels=["Between +", "Between –"],vmin=-10, vmax=vmax)
            plt.title(f'{group.capitalize()} Coupling Quadrants', fontsize=7) #\n(Mean ± SD)
            plt.tight_layout()

            if save_dir:
                if string:
                    plt.savefig(f"{save_dir}/{group}_quadrant_heatmap_{str(ntotal)}_{string}.pdf", dpi=300)
                else:
                    plt.savefig(f"{save_dir}/{group}_quadrant_heatmap_{str(ntotal)}.pdf", dpi=300)
                

            plt.show()

        # Perform chi-square test between groups (on total counts)
        if len(groups) >= 2:
            group1, group2 = groups[:2]
            total_counts_1 = np.sum(all_group_stats[group1]['raw_counts'], axis=0)
            total_counts_2 = np.sum(all_group_stats[group2]['raw_counts'], axis=0)

            contingency_table = np.stack([total_counts_1, total_counts_2], axis=0)
            chi2, p_val, dof, expected = chi2_contingency(contingency_table)

            print(f"Chi-square test between {group1} and {group2}:")
            print(f"  χ² = {chi2:.2f}, p = {p_val:.4g}, dof = {dof}")
            print("  Contingency Table:")
            print(pd.DataFrame(contingency_table, index=[group1, group2], columns=quadrant_names))

            # Optionally save
            if save_dir:
                pd.DataFrame(contingency_table, index=[group1, group2], columns=quadrant_names).to_csv(
                    f"{save_dir}/quadrant_chi2_contingency.csv"
                )
                with open(f"{save_dir}/quadrant_chi2_result.txt", "w") as f:
                    f.write(f"Chi-square test between {group1} and {group2}:\n")
                    f.write(f"χ² = {chi2:.2f}, p = {p_val:.4g}, dof = {dof}\n")
                    f.write("Contingency Table:\n")
                    f.write(pd.DataFrame(contingency_table, index=[group1, group2], columns=quadrant_names).to_string())

        return all_group_stats
    
    def plot_group_coupling_differences(self,
        df, 
        mode='mean_abs', 
        save_path=None, 
        paired=True, 
        figure_size=(3, 3), 
        ylim=None, 
        group_colors=None,
        group_order=None,
        plot_type='bar',  # 'bar' or 'box'
        width = 0.6,
        showfliers=True
    ):


        mpl.rcParams['pdf.fonttype'] = 42
        plt.rcParams.update({'font.size': 7, 'font.family': 'arial'})

        df['diff'] = df['coupling_within'] - df['coupling_between']

        # Determine group order
        if group_order is None:
            group_order = sorted(df['group'].unique())
        else:
            group_names =  group_order
            group_order = [s.lower() for s in group_order] # ensure list for indexing
            

        # Grouped summary
        summary = df.groupby('group')['diff'].agg(['mean', 'sem']).reindex(group_order)

        # Set up group colors
        if group_colors is None:
            palette = sns.color_palette('muted', n_colors=len(group_order))
            group_colors = dict(zip(group_order, palette))
        
        fig, ax = plt.subplots(figsize=figure_size)
        ax.axhline(0, linestyle='--', color='gray')
        if plot_type == 'bar':
            for i, group in enumerate(group_order):
                mean_val = summary.loc[group, 'mean']
                sem_val = summary.loc[group, 'sem']
                color = group_colors.get(group, 'gray')
                ax.bar(i, mean_val, color=color, width=width)
                ax.errorbar(i, mean_val, yerr=sem_val, fmt='none', ecolor='black', capsize=2)
            ax.set_xticks(range(len(group_order)))
            ax.set_xticklabels(group_names)

        elif plot_type == 'box':
            # Draw manually for color control
            for i, group in enumerate(group_order):
                data = df[df['group'] == group]['diff'].dropna().values
                bp = ax.boxplot(data, positions=[i], widths=width, patch_artist=True,showfliers=showfliers,flierprops=dict(marker='o', markersize=2, markerfacecolor='white', markeredgecolor='gray'),
                                boxprops=dict(facecolor=group_colors.get(group, 'gray')))
                for patch in bp['boxes']:
                    patch.set_facecolor(group_colors.get(group, 'gray'))
                for element in ['medians']: #medians
                    for line in bp[element]:
                        line.set_color('white')
                        line.set_linewidth(1.2)
                for element in ['whiskers', 'caps']:
                    for line in bp[element]:
                        line.set_color('black')
                        line.set_linewidth(1)
                
            ax.set_xticks(range(len(group_order)))
            ax.set_xticklabels(group_names)
        if mode == 'mean_abs':
            ax.set_ylabel(f'Mean Coupling |Within| - |Between|')
        else:
            ax.set_ylabel(f'{mode.capitalize()} Coupling (Within - Between)')
        ax.set_title(f'Coupling Selectivity')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if ylim:
            ax.set_ylim(ylim)
        plt.tight_layout()

        all_p_values = []
        all_stats_dict = {}
        test_stats = []
        comparisons_names = []
        # Paired permutation test
        for g in group_order:
            vals = df[df['group'] == g]['diff'].dropna().values
            p, stat = self.stats.perform_permutation_test(vals, np.zeros_like(vals), paired=paired)
            all_p_values.append(p)
            test_stats.append(stat)
            print(f"{g}: p={p:.4f}, stat={stat:.3f}")

            label1 = f"{g}_coupling_diff_stats"
            all_stats_dict[label1] = self.stats.get_basic_stats(vals)
            comparisons_names.append((label1,'zero'))

        # Calculate Bonferroni significance
        corrected_p_values, significance_stars = self.stats.calculate_bonferroni_significance(all_p_values)

        # Draw significance lines and stars
        for idx, (p, star) in enumerate(zip(corrected_p_values, significance_stars)):
            if star != 'ns':
                bottom, top = ax.get_ylim()
                y = top - (top * 0.1)  # Adjust y-coordinate
                # x1 = bar_positions[idx] - bar_width / 2
                # x2 = bar_positions[idx] + bar_width / 2
                x1 = idx
                self.add_significance_line(ax, x1, y=y, significance=star)

        if save_path:
            plt.savefig(save_path, dpi=300)
        if save_path and '/' in save_path:
            save_path_updated = save_path[:save_path.rfind('/')]
            print(f"Saving stats to {save_path_updated}")
            name_without_ext = save_path.split('/')[-1].split('.')[0]

            df_tests = self.stats.to_table(comparisons_names, test_stats, all_p_values, save_path=f'{save_path_updated}/stat_tests_{name_without_ext}.csv',type='permutation paired')
            df_stats = self.stats.basic_stats_to_table(all_stats_dict, save_path=f'{save_path_updated}/basic_stats_{name_without_ext}.csv')
        return summary
    
    def plot_active_passive_quadrant_difference(self,
            quad_stats_active,
            quad_stats_passive,
            save_dir=None,
            figsize=(3, 3),
            decimal_places=0,
            vmax=50,
            colormap='coolwarm',
            string = None
        ):

        mpl.rcParams['pdf.fonttype'] = 42
        plt.rcParams.update({'font.size': 7, 'font.family': 'arial'})

        # quadrant_labels = np.array([["+/+", "+/–"], ["–/+", "–/–"]])
        quadrant_labels = np.array([["+/–", "+/+"],
                            ["–/–", "–/+"]])

        groups = quad_stats_active.keys()

        results = {}

        for group in groups:

            active_counts = quad_stats_active[group]['raw_counts']   # datasets x 4
            passive_counts = quad_stats_passive[group]['raw_counts'] # datasets x 4

            # Convert to fractions per dataset
            active_frac = active_counts / active_counts.sum(axis=1, keepdims=True) * 100
            passive_frac = passive_counts / passive_counts.sum(axis=1, keepdims=True) * 100

            # Difference per dataset
            diff = active_frac - passive_frac

            mean_diff = np.nanmean(diff, axis=0)
            std_diff = np.nanstd(diff, axis=0)

            results[group] = {
                'mean_diff': mean_diff,
                'std_diff': std_diff,
                'all_dataset_diff': diff
            }

            # Reshape to 2x2 for plotting
            mean_matrix = mean_diff.reshape(2, 2)
            std_matrix = std_diff.reshape(2, 2)

            label_matrix = np.array([
                [
                    f"{mean_matrix[i,j]:.{decimal_places}f} ± {std_matrix[i,j]:.{decimal_places}f}\n{quadrant_labels[i,j]}"
                    for j in range(2)
                ] for i in range(2)
            ])

            plt.figure(figsize=figsize)
            sns.heatmap(
                mean_matrix,
                annot=label_matrix,
                fmt='',
                cmap=colormap,
                center=0,
                cbar=False,
                vmin=-vmax,
                vmax=vmax,
                xticklabels=["Within –", "Within +"],
                yticklabels=["Between +", "Between –"]
            )

            plt.title(f"{group.capitalize()}:\nActive – Passive (%)", fontsize=7)
            plt.tight_layout()

            if save_dir:
                if string:
                    plt.savefig(f"{save_dir}/{group}_active_minus_passive_heatmap_{string}.pdf", dpi=300)
                else:
                    plt.savefig(f"{save_dir}/{group}_active_minus_passive_heatmap.pdf", dpi=300)

            plt.show()

        return results
    
    def plot_quadrant_means_across_datasets(self,
            combined_df,
            groups=('sound', 'opto'),
            dataset_col='dataset',
            group_col='group',
            save_dir=None,
            figsize=(3, 3),
            decimal_places=2,
            vmax=None,
            colormap='coolwarm',
            string=None,
            pool_across_datasets=False,
            metric='difference',   # 'within', 'between', or 'difference',
            scale_factor = 100
        ):

        mpl.rcParams['pdf.fonttype'] = 42
        plt.rcParams.update({'font.size': 7, 'font.family': 'arial'})

        quadrant_labels = np.array([["+/–", "+/+"],
                                    ["–/–", "–/+"]])

        all_group_stats = {}

        for group in groups:

            subset_group = combined_df[combined_df[group_col] == group]

            # function to compute metric of interest
            def get_metric(df):
                if metric == 'within':
                    return df['coupling_within'].values
                elif metric == 'between':
                    return df['coupling_between'].values
                elif metric == 'difference':
                    return df['coupling_within'].values - df['coupling_between'].values 
                elif metric == 'distance':
                    w = df['coupling_within'].values
                    b = df['coupling_between'].values
                    return np.sqrt(w**2 + b**2) * scale_factor
                else:
                    raise ValueError("metric must be 'within', 'between','distance', or 'difference'")

            # helper to compute quadrant means for a dataframe
            def compute_quadrant_means(df):

                sign_within = np.sign(df['coupling_within'].values)
                sign_between = np.sign(df['coupling_between'].values)

                values = get_metric(df)

                quadrant_vals = { (0,0): [], (0,1): [], (1,0): [], (1,1): [] }

                for s_w, s_b, val in zip(sign_within, sign_between, values):

                    row = 0 if s_b > 0 else 1
                    col = 0 if s_w < 0 else 1

                    quadrant_vals[(row, col)].append(val)

                means = np.zeros((2,2))
                stds = np.zeros((2,2))

                for (r,c), arr in quadrant_vals.items():
                    if len(arr) > 0:
                        means[r,c] = np.nanmean(arr)
                        stds[r,c] = np.nanstd(arr)
                    else:
                        means[r,c] = np.nan
                        stds[r,c] = np.nan

                return means, stds

            # ---- OPTION 1: POOL ALL NEURONS ----
            if pool_across_datasets:

                mean_matrix, std_matrix = compute_quadrant_means(subset_group)

                all_group_stats[group] = {
                    'mean': mean_matrix,
                    'std': std_matrix
                }

            # ---- OPTION 2: AVERAGE PER DATASET FIRST ----
            else:

                dataset_means = []
                for dataset in subset_group[dataset_col].unique():
                    subset = subset_group[subset_group[dataset_col] == dataset]
                    means, _ = compute_quadrant_means(subset)
                    dataset_means.append(means)

                dataset_means = np.array(dataset_means)

                mean_matrix = np.nanmean(dataset_means, axis=0)
                std_matrix = np.nanstd(dataset_means, axis=0)

                all_group_stats[group] = {
                    'mean': mean_matrix,
                    'std': std_matrix,
                    'all_dataset_means': dataset_means
                }

            # ----- PLOTTING -----

            label_matrix = np.array([
                [
                    f"{mean_matrix[i,j]:.{decimal_places}f} ± {std_matrix[i,j]:.{decimal_places}f}\n{quadrant_labels[i,j]}"
                    for j in range(2)
                ] for i in range(2)
            ])

            plt.figure(figsize=figsize)
            if metric == 'distance':
                center = None
                vmin=0
            else:
                center = 0
                vmin=-vmax if vmax else None

            sns.heatmap(
                mean_matrix,
                annot=label_matrix,
                fmt='',
                cmap=colormap,
                center=center,
                cbar=True,
                vmin=vmin,
                vmax=vmax,
                xticklabels=["Within –", "Within +"],
                yticklabels=["Between +", "Between –"]
            )

            mode = "pooled" if pool_across_datasets else "per dataset"

            plt.title(f'{group.capitalize()} Quadrant Means', fontsize=7) #\nMetric: {metric} ({mode})
            plt.tight_layout()
            ax = plt.gca()
            ax.set_box_aspect(1)

            if save_dir:
                tag = "_pooled" if pool_across_datasets else "_perDataset"
                if string:
                    fname = f"{save_dir}/{group}_quadrant_means_{metric}_{string}{tag}.pdf"
                else:
                    fname = f"{save_dir}/{group}_quadrant_means_{metric}{tag}.pdf"

                plt.savefig(fname, dpi=300)

            plt.show()

        return all_group_stats
    
    def plot_coupling_index_vs_decoding_info(self,
        coupling_index_by_celltype,
        peak_info_struc,
        decoded_feature='sound_category',
        state='Active',
        threshold=None,
        celltype_colors={'pyr': (0.37, 0.75, 0.49), 'som': (0.17, 0.35, 0.8), 'pv': (0.82, 0.04, 0.04)},
        figsize=(3, 3),
        marker_size=20,
        peak_to_get = 'peak_values',
        alpha = 1,
        save_path = None,
        ylim=(-.01, .5),
        xlim=(-0.01, 1.01)
    ):
        """
        Plots coupling index vs peak decoding info for each neuron (all datasets pooled).
        Optionally threshold coupling index.
        """
        mpl.rcParams['pdf.fonttype'] = 42   # TrueType fonts (editable)
        all_rows = []

        for dataset in peak_info_struc:
            for cell_type in peak_info_struc[dataset]:

                # Get decoding info
                info = peak_info_struc[dataset][cell_type].get(decoded_feature, None)
                if info is None:
                    print(f"Missing decoding info for {dataset}, {cell_type}, {decoded_feature}")
                    continue

                peak_vals = info[peak_to_get]
                if peak_to_get == 'peak_frames':
                    peak_vals = peak_vals/169 #normalize by number of frames
                neuron_indices = info['neuron_indices'].flatten()

                # Get coupling index
                try:
                    coupling_vals = coupling_index_by_celltype[('No Coupling', 'All')][cell_type.lower()][state]
                except KeyError:
                    print(f"Missing coupling index for {dataset}, {cell_type}, {state}")
                    continue

                for i, neuron_idx in enumerate(neuron_indices):
                    if neuron_idx >= len(coupling_vals):
                        continue
                    if coupling_vals[neuron_idx] < 0 or  np.isnan(coupling_vals[neuron_idx]):
                        continue

                    row = {
                        'dataset': dataset,
                        'cell_type': cell_type.lower(),
                        'neuron_idx': neuron_idx,
                        'coupling_index': coupling_vals[neuron_idx],
                        'peak_info': peak_vals[i],
                    }

                    if threshold is not None or row['coupling_index'] > threshold:
                        all_rows.append(row)

        df = pd.DataFrame(all_rows)

        # Plot
        plt.figure(figsize=figsize)
        for cell_type, color in celltype_colors.items():
            sub = df[df['cell_type'] == cell_type]
            plt.scatter(
                sub['coupling_index'],
                sub['peak_info'],
                label=cell_type.upper(),
                facecolors='none',
                edgecolors=color,
                s=marker_size,
                linewidth=.5,
                alpha=alpha
            )

        # Correlation
        r, p = pearsonr(df['coupling_index'], df['peak_info'])
        plt.title(f'{decoded_feature.capitalize()}', fontsize=7) 
        if decoded_feature == 'sound_category':
            plt.title('Sound category', fontsize=7)
        # plt.title(f'{decoded_feature}\nr = {r:.2f}, p = {p:.3f}', fontsize=7)

        plt.xlabel('Coupling Index', fontsize=7)
        plt.ylabel('Peak Info (bits)', fontsize=7)
        ax = plt.gca()
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
        ax.set_yticklabels(ax.get_yticks(), fontsize=7)
        ax.set_xticklabels(ax.get_xticks(), fontsize=7)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
        plt.legend(frameon=False, fontsize=6)
        sns.despine(right=True, top=True)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()

        return df

    def plot_decoding_by_coupling_bin2(
            coupling_index_by_celltype,
            peak_info_struc,
            decoded_feature='sound_category',
            state='Active',
            celltype_colors={'pyr': (0.37, 0.75, 0.49), 'som': (0.17, 0.35, 0.8), 'pv': (0.82, 0.04, 0.04)},
            figsize=(3, 3),
            marker_size=20,
            peak_to_get='peak_values',
            alpha=1,
            bins=[0, 0.25, 0.5, 0.75, 1.01],  # right-inclusive last bin
            threshold=None,
            top_neurons=False,
            save_path = None,
            use_quantile_bins=False,          # ✅ NEW
            n_quantiles=4,
            group_celltypes_within_bin=False
        ):
            """
            Bins coupling index and plots mean ± SEM of peak decoding info per bin, per cell type.
            """
            all_rows = []
            mpl.rcParams['pdf.fonttype'] = 42   # TrueType fonts (editable)

            top_pyr_color = (0.10, 0.36, 0.16) 

            if top_neurons is True:
                celltype_colors = {
                    'pyr': (0.37, 0.75, 0.49),
                    'som': (0.17, 0.35, 0.8),
                    'pv': (0.82, 0.04, 0.04),
                    'top_pyr': top_pyr_color
                }

            for dataset in peak_info_struc:
                for cell_type in peak_info_struc[dataset]:

                    info = peak_info_struc[dataset][cell_type].get(decoded_feature, None)
                    if info is None:
                        continue

                    peak_vals = info[peak_to_get]
                    if peak_to_get == 'peak_frames':
                        peak_vals = peak_vals / 169  # normalize by frames
                    neuron_indices = info['neuron_indices'].flatten()

                    # First, filter to top 10 PYR neurons if requested
                    if top_neurons is True and cell_type.lower() == 'pyr':
                        # Sort indices of top neurons by decoding strength
                        sorted_indices = np.argsort(peak_vals)[::-1]
                        top_indices = sorted_indices[:10]  # Top 10 neurons

                    try:
                        coupling_vals = coupling_index_by_celltype[('No Coupling', 'All')][cell_type.lower()][state]
                    except KeyError:
                        print(f"Missing coupling index for {dataset}, {cell_type}, {state}")
                        continue

                    for i, neuron_idx in enumerate(neuron_indices):
                        if neuron_idx >= len(coupling_vals):
                            continue
                        if coupling_vals[neuron_idx] < 0 or  np.isnan(coupling_vals[neuron_idx]):
                            continue
                        row = {
                            'dataset': dataset,
                            'cell_type': cell_type.lower(),
                            'neuron_idx': neuron_idx,
                            'coupling_index': coupling_vals[neuron_idx],
                            'peak_info': peak_vals[i],
                        }
                        if threshold is not None:
                            # Use threshold filtering
                            if row['coupling_index'] > threshold:
                                all_rows.append(row)
                        elif top_neurons is True and cell_type.lower() == 'pyr':
                            # print(f"Top 10 PYR neurons for {dataset}: {neuron_indices[top_indices]}")
                            if i in top_indices:
                                row = {
                                    'dataset': dataset,
                                    'cell_type': 'top_pyr',
                                    'neuron_idx': neuron_idx,
                                    'coupling_index': coupling_vals[neuron_idx],
                                    'peak_info': peak_vals[i],
                                }
                            all_rows.append(row)
                        else:
                            all_rows.append(row)

            df = pd.DataFrame(all_rows)
            # df['coupling_bin'] = pd.cut(df['coupling_index'], bins=bins, include_lowest=True, right = False)
            if use_quantile_bins:
                df['coupling_bin'] = None  # initialize

                for cell_type in df['cell_type'].unique():
                    mask = df['cell_type'] == cell_type
                    try:
                        df.loc[mask, 'coupling_bin'] = pd.qcut(
                            df.loc[mask, 'coupling_index'],
                            q=n_quantiles,
                            duplicates='drop'
                        )
                    except ValueError:
                        # fallback if not enough unique values
                        df.loc[mask, 'coupling_bin'] = pd.cut(
                            df.loc[mask, 'coupling_index'],
                            bins=n_quantiles
                        )
            else:
                df['coupling_bin'] = pd.cut(
                    df['coupling_index'],
                    bins=bins,
                    include_lowest=True,
                    right=False
                )

            plt.figure(figsize=figsize)

            # bin_centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins) - 1)]
            
            
            for cell_type, color in celltype_colors.items():
                sub = df[df['cell_type'] == cell_type]
                means = []
                errors = []
                bin_data_all = {}

                categories = sorted(sub['coupling_bin'].dropna().unique()) # categories = sub['coupling_bin'].dropna().cat.categories
                bin_centers = [(b.left + b.right) / 2 for b in categories]

                if group_celltypes_within_bin:
                    bin_centers = np.arange(len(categories))

                for b in categories: #for b in df['coupling_bin'].cat.categories:
                    bin_data = sub[sub['coupling_bin'] == b]['peak_info']
                    means.append(bin_data.mean())
                    errors.append(sem(bin_data) if len(bin_data) > 1 else 0)
                    bin_data_all[b] = bin_data

                #perform stat test
                # ---- Permutation test ----
                bins = list(bin_data_all.keys())

                if len(bins) == 2:
                    data1 = bin_data_all[bins[0]]
                    data2 = bin_data_all[bins[1]]

                    # only run if both bins have data
                    if len(data1) > 0 and len(data2) > 0:
                        all_p_values = []
                        all_stats_dict = {}
                        test_stats = []
                        comparisons_names = []
                        pval, stat = self.stats.perform_permutation_test(
                            data1, data2,
                            paired=False,
                            n_permutations=10000
                        )
                        print(f'permutation test|| pval: {pval}, celltype {cell_type}')
                        all_p_values.append(pval)
                        test_stats.append(stat)
                        label1 = f"low_coupling"
                        label2 = f"hi_coupling"
                        all_stats_dict[label1] = self.stats.get_basic_stats(data1)
                        all_stats_dict[label2] = self.stats.get_basic_stats(data2)
                        comparisons_names.append((label1,label2))
                    else:
                        pval, stat = np.nan, np.nan
                else:
                    pval, stat = np.nan, np.nan

                plt.errorbar(
                    bin_centers, means, yerr=errors,
                    label=cell_type.upper(),
                    color=color,
                    marker='o',
                    markersize=marker_size / 5,
                    capsize=2,
                    lw=1
                )
                
                # select the two columns for the cell type
                x = df[df['cell_type'] == cell_type]['coupling_index']
                y = df[df['cell_type'] == cell_type]['peak_info']

                # remove NaNs
                mask = ~np.isnan(x) & ~np.isnan(y)
                r, p = pearsonr(x[mask], y[mask])
                print(f'{cell_type.upper()}: r = {r:.2f}, p = {p:.3f}')

            plt.xlabel('Coupling Bin', fontsize=7)
            plt.ylabel('Peak Info (bits)', fontsize=7)
            plt.title(decoded_feature.capitalize(), fontsize=7)
            if decoded_feature == 'sound_category':
                plt.title('Sound Category', fontsize=7)
            ax = plt.gca()
            ax.set_xticks(bin_centers)
            if use_quantile_bins:
                ax.set_xticklabels([f'Q{i+1}' for i in range(len(categories))], rotation=0)
            else:
                ax.set_xticklabels(
                    [f'{b.left:.2f}-{b.right:.2f}' for b in df['coupling_bin'].cat.categories],
                    fontsize=7,
                    rotation=45
                )
            # ax.set_xticklabels([f'{b.left:.2f}-{b.right:.2f}' for b in df['coupling_bin'].cat.categories], fontsize=7, rotation=45)
            ax.set_yticklabels(ax.get_yticks(), fontsize=7)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
            if len(bins) != 3:
                ax.set_box_aspect(1)
            else:
                ax.set_xlim(-.1,1.1)

            plt.legend(
                loc='center left',           # Anchor point of legend
                bbox_to_anchor=(1.05, 0.5),  # Position relative to axes
                borderaxespad=0.0,           # Padding between axes and legend
                frameon=False,                 # Draw a frame around legend
                fontsize=6
            )
            sns.despine()
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, bbox_inches='tight')
                save_path_updated = save_path[:save_path.rfind('/')]
                print(f"Saving stats to {save_path_updated}")
                name_without_ext = save_path.split('/')[-1].split('.')[0]
                df_tests = self.stats.to_table(comparisons_names, test_stats, all_p_values, save_path=f'{save_path_updated}/stat_tests_{name_without_ext}.csv',type='permutation unpaired')
                df_stats = self.stats.basic_stats_to_table(all_stats_dict, save_path=f'{save_path_updated}/basic_stats_{name_without_ext}.csv')
            plt.show()

            return df
    
    def make_plot_frames(self, frame_start, frame_end, max_frame):
            """
            Create valid plotting frame indices.
            """
            frames = np.arange(frame_start, min(frame_end, max_frame))

            if len(frames) == 0:
                raise ValueError(
                    "No frames to plot. frame_start={frame_start}, "
                    "frame_end={frame_end}, max_frame={max_frame}"
                )

            return frames


    def plot_running_glm_neuron_group(
        self,
        neuron_ids,
        frame_start,
        frame_end,
        response_matrix,
        y_pred,
        frac_dev_expl,
        running_trace_mode,
        supp_figure_dir,
        animalID,
        date,
        model_type,
        fold_number,
        label,
        plot_real_activity = True,
        running_group_summaries=None,
        velocity_display_traces=None,
        plot_smoothed_rate=True,
        imaging_rate_hz=30.4791,
        smooth_sigma_sec=0.5,
        prediction_color="deepskyblue",
        figsize_inputt= (8, 1.35*4),
        save_string = 'active'
    ):

        response_matrix = np.asarray(response_matrix)
        y_pred = np.asarray(y_pred)
        frac_dev_expl = np.asarray(frac_dev_expl).squeeze()

        max_frame = min(response_matrix.shape[1], y_pred.shape[0])

        if running_trace_mode == "grouped_predictors":
            if running_group_summaries is None:
                raise ValueError("running_group_summaries is required.")

            for _, trace in running_group_summaries.items():
                max_frame = min(max_frame, len(trace))

        elif running_trace_mode in ["raw_velocity", "raw_velocity_split"]:
            if velocity_display_traces is None:
                raise ValueError("velocity_display_traces is required.")

            for _, trace in velocity_display_traces.items():
                max_frame = min(max_frame, len(trace))

        else:
            raise ValueError(
                "running_trace_mode must be grouped_predictors, raw_velocity, or raw_velocity_split"
            )

        frames = self.make_plot_frames(frame_start, frame_end, max_frame)

        smooth_sigma_frames = smooth_sigma_sec * imaging_rate_hz

        n_rows = len(neuron_ids) + 1

        fig, axs = plt.subplots(n_rows, 1, figsize=(figsize_inputt), sharex=True)

        if n_rows == 1:
            axs = [axs]

        for i, neuron_idx in enumerate(neuron_ids):
            ax = axs[i]

            true_y = response_matrix[neuron_idx, frames]
            pred_y = y_pred[frames, neuron_idx]

            if plot_real_activity:
                ax.plot(frames, true_y, color=[0.75, 0.75, 0.75], linewidth=0.6)

            if plot_smoothed_rate:
                true_rate = gaussian_filter1d(
                    true_y.astype(float),
                    sigma=smooth_sigma_frames,
                    mode="nearest"
                )
            
                ax.plot(frames, true_rate, color=[0.25, 0.25, 0.25], linewidth=1.0,label=f'Smoothed rate, σ={smooth_sigma_sec:.1f}s')

            ax.plot(frames, pred_y, color=prediction_color, linewidth=1.0,label=f'Running-only GLM, FDE={frac_dev_expl[neuron_idx]:.2f}')

            ax.set_ylabel("N{}".format(neuron_idx), fontsize=7)
            ax.legend(frameon=False, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=6)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        ax = axs[-1]

        if running_trace_mode == "grouped_predictors":
            for trace_name, trace in running_group_summaries.items():
                ax.plot(frames, trace[frames], linewidth=0.9, label=trace_name)

            ax.set_ylabel("Convolved pred.", fontsize=7)

        elif running_trace_mode in ["raw_velocity", "raw_velocity_split"]:
            for trace_name, trace in velocity_display_traces.items():
                ax.plot(frames, trace[frames], linewidth=0.9, label=trace_name)

            ax.set_ylabel("Vel.", fontsize=7)
            if running_trace_mode == "raw_velocity_split":
                ax.set_ylabel("|Vel.|", fontsize=7)

        ax.set_xlabel("Frame")

        ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=7)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.suptitle(
            "{} {}, fold {}, {}".format(
                animalID, date, fold_number + 1, label
            ),
            fontsize=8
        )

        plt.tight_layout()

        os.makedirs(supp_figure_dir, exist_ok=True)

        save_path = os.path.join(
            supp_figure_dir,
            "{}_{}_{}_fold{}_{}to{}_{}.pdf".format(
                label,
                running_trace_mode,
                animalID,
                fold_number + 1,
                frames[0],
                frames[-1],
                save_string
            )
        )

        plt.savefig(save_path, bbox_inches="tight")
        plt.show()

        print("Saved:", save_path)

        return fig, axs, save_path


    def plot_single_neuron_predictor_heatmap(
        self,
        neuron_idx,
        frame_start,
        frame_end,
        response_matrix,
        y_pred,
        frac_dev_expl,
        running_predictors,
        supp_figure_dir,
        animalID,
        date,
        model_type,
        fold_number,
        label,
        plot_real_activity=True,
        plot_smoothed_rate=True,
        imaging_rate_hz=30.4791,
        smooth_sigma_sec=0.5,
        prediction_color="deepskyblue",
        heatmap_clip=3,
        figsize_inputt=(8, 4.5)
    ):

        response_matrix = np.asarray(response_matrix)
        y_pred = np.asarray(y_pred)
        frac_dev_expl = np.asarray(frac_dev_expl).squeeze()
        running_predictors = np.asarray(running_predictors)

        max_frame = min(
            response_matrix.shape[1],
            y_pred.shape[0],
            running_predictors.shape[1]
        )

        frames = self.make_plot_frames(frame_start, frame_end, max_frame)

        smooth_sigma_frames = smooth_sigma_sec * imaging_rate_hz

        pred_display = self.zscore_for_display(
            running_predictors[:, frames], axis=1
        )

        pred_display = np.clip(pred_display, -heatmap_clip, heatmap_clip)

        fig, axs = plt.subplots(
            3, 1, figsize=figsize_inputt, sharex=True,
            gridspec_kw={"height_ratios": [1.2, 1.2, 2.2]}
        )

        true_y = response_matrix[neuron_idx, frames]
        pred_y = y_pred[frames, neuron_idx]

        if plot_real_activity:
            axs[0].plot(frames, true_y, color=[0.75, 0.75, 0.75], linewidth=0.6)

        if plot_smoothed_rate:
            true_rate = gaussian_filter1d(
                true_y.astype(float),
                sigma=smooth_sigma_frames,
                mode="nearest"
            )

            axs[0].plot(frames, true_rate, color=[0.25, 0.25, 0.25], linewidth=1.0)

        axs[1].plot(frames, pred_y, color=prediction_color, linewidth=1.0)

        im = axs[2].imshow(
            pred_display,
            aspect="auto",
            interpolation="nearest",
            extent=[
                frames[0],
                frames[-1],
                running_predictors.shape[0],
                0
            ]
        )

        axs[2].set_ylabel("Running predictor", fontsize=7)
        axs[2].set_xlabel("Frame")

        plt.colorbar(im, ax=axs[2], fraction=0.025, pad=0.02)

        for ax in axs:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        fig.suptitle(
            "{} {}, fold {}, neuron {}".format(
                animalID, date, fold_number + 1, neuron_idx
            ),
            fontsize=10
        )

        plt.tight_layout()

        os.makedirs(supp_figure_dir, exist_ok=True)

        save_path = os.path.join(
            supp_figure_dir,
            "{}_heatmap_N{}_{}_{}_fold{}_{}to{}.pdf".format(
                label,
                neuron_idx,
                animalID,
                date,
                fold_number + 1,
                frames[0],
                frames[-1]
            )
        )

        plt.savefig(save_path, bbox_inches="tight")
        plt.show()

        print("Saved:", save_path)

        return fig, axs, save_path


    def zscore_for_display(self, x, axis=1, eps=1e-12):
        x = np.asarray(x).astype(float).copy()
        mu = np.nanmean(x, axis=axis, keepdims=True)
        sigma = np.nanstd(x, axis=axis, keepdims=True)
        sigma[sigma < eps] = 1.0
        return (x - mu) / sigma


    def average_aligned_across_folds(
        self,
        aligned_true,
        aligned_pred,
        aligned_velocity,
        key,
        folds=range(10),
        trial_average=True,
        velocity_axis_order=("y", "x"),
    ):
        """
        Average aligned true neural, predicted neural, and velocity across folds.

        Inputs:
            aligned_true[key][fold]: trials x neurons x frames
            aligned_pred[key][fold]: trials x neurons x frames
            aligned_velocity[key][fold]: trials x 2 x frames

        Returns:
            avg_true: neurons x frames
            avg_pred: neurons x frames
            avg_velocity: 2 x frames
            fold_counts: dict with number of trials per fold
        """

        true_fold_means = []
        pred_fold_means = []
        velocity_fold_means = []
        fold_counts = {}

        for fold in folds:
            if fold not in aligned_true[key]:
                continue
            if fold not in aligned_pred[key]:
                continue
            if fold not in aligned_velocity[key]:
                continue

            true_data = np.asarray(aligned_true[key][fold])
            pred_data = np.asarray(aligned_pred[key][fold])
            velocity_data = np.asarray(aligned_velocity[key][fold])

            if true_data.ndim != 3:
                raise ValueError(f"true data fold {fold} should be trials x neurons x frames, got {true_data.shape}")
            if pred_data.ndim != 3:
                raise ValueError(f"pred data fold {fold} should be trials x neurons x frames, got {pred_data.shape}")
            if velocity_data.ndim != 3:
                raise ValueError(f"velocity data fold {fold} should be trials x 2 x frames, got {velocity_data.shape}")

            if true_data.shape != pred_data.shape:
                raise ValueError(
                    f"True/pred shape mismatch for {key}, fold {fold}: "
                    f"true {true_data.shape}, pred {pred_data.shape}"
                )

            if velocity_data.shape[0] != true_data.shape[0]:
                raise ValueError(
                    f"Trial mismatch for {key}, fold {fold}: "
                    f"neural has {true_data.shape[0]} trials, velocity has {velocity_data.shape[0]}"
                )

            if velocity_data.shape[2] != true_data.shape[2]:
                raise ValueError(
                    f"Frame mismatch for {key}, fold {fold}: "
                    f"neural has {true_data.shape[2]} frames, velocity has {velocity_data.shape[2]}"
                )

            if velocity_data.shape[1] != 2:
                raise ValueError(
                    f"Expected velocity to be trials x 2 x frames, got {velocity_data.shape}"
                )

            fold_counts[fold] = true_data.shape[0]

            if trial_average:
                true_fold_means.append(np.nanmean(true_data, axis=0))          # neurons x frames
                pred_fold_means.append(np.nanmean(pred_data, axis=0))          # neurons x frames
                velocity_fold_means.append(np.nanmean(velocity_data, axis=0))  # 2 x frames
            else:
                true_fold_means.append(true_data)
                pred_fold_means.append(pred_data)
                velocity_fold_means.append(velocity_data)

        if len(true_fold_means) == 0:
            raise ValueError(f"No valid folds found for key: {key}")

        avg_true = np.nanmean(np.stack(true_fold_means, axis=0), axis=0)
        avg_pred = np.nanmean(np.stack(pred_fold_means, axis=0), axis=0)
        avg_velocity = np.nanmean(np.stack(velocity_fold_means, axis=0), axis=0)

        return avg_true, avg_pred, avg_velocity, fold_counts


    def smooth_neural_trace_for_display(
        self,
        neural_data,
        imaging_rate_hz=30.4791,
        smooth_sigma_sec=0.5,
        axis=-1,
    ):
        """
        Smooth neural data along the frame axis for display as estimated rate.

        Input can be:
            neurons x frames
            trials x neurons x frames
            any array where frames are axis=-1
        """
        sigma_frames = smooth_sigma_sec * imaging_rate_hz

        return gaussian_filter1d(
            np.asarray(neural_data).astype(float),
            sigma=sigma_frames,
            axis=axis,
            mode="nearest"
        )


    def minmax_for_display(self, x, eps=1e-12):
        """
        Display-only min-max scaling to 0-1.
        """
        x = np.asarray(x).astype(float).copy()
        x = x - np.nanmin(x)

        denom = np.nanmax(x)
        if denom < eps:
            denom = 1.0

        return x / denom


    def plot_aligned_fold_average_neurons(
        self,
        key,
        aligned_true,
        aligned_pred,
        aligned_velocity,
        neuron_ids,
        frame_axis=None,
        folds=range(10),
        response_mode="deconvolved",
        imaging_rate_hz=30.4791,
        smooth_sigma_sec=0.5,
        velocity_axis_order=("y", "x"),
        scale_velocity=True,
        title=None,
        save_dir=None,
        save_name=None,
        prediction_color="deepskyblue",
        figsize=(8, 1.35 * 4),
        event_frames = []
    ):
        """
        Plot average aligned true vs predicted neural activity across folds for one dataset.

        Inputs:
            aligned_true[key][fold]: trials x neurons x frames
            aligned_pred[key][fold]: trials x neurons x frames
            aligned_velocity[key][fold]: trials x 2 x frames

        response_mode:
            "deconvolved" = plot mean deconvolved activity
            "rate" = plot gaussian-smoothed mean deconvolved activity

        velocity_axis_order:
            ("y", "x") if velocity[:, 0, :] is y and velocity[:, 1, :] is x
            ("x", "y") if velocity[:, 0, :] is x and velocity[:, 1, :] is y
        """

        avg_true, avg_pred, avg_velocity, fold_counts = self.average_aligned_across_folds(
            aligned_true=aligned_true,
            aligned_pred=aligned_pred,
            aligned_velocity=aligned_velocity,
            key=key,
            folds=folds,
            velocity_axis_order=velocity_axis_order,
        )

        if response_mode == "deconvolved":
            true_to_plot = avg_true
            pred_to_plot = avg_pred
            response_label = "Mean deconvolved"

        elif response_mode == "rate":
            true_to_plot = self.smooth_neural_trace_for_display(
                avg_true,
                imaging_rate_hz=imaging_rate_hz,
                smooth_sigma_sec=smooth_sigma_sec,
                axis=-1,
            )

            pred_to_plot = self.smooth_neural_trace_for_display(
                avg_pred,
                imaging_rate_hz=imaging_rate_hz,
                smooth_sigma_sec=smooth_sigma_sec,
                axis=-1,
            )

            response_label = f"Estimated rate, σ={smooth_sigma_sec:.1f}s"

        else:
            raise ValueError("response_mode must be 'deconvolved' or 'rate'")

        n_frames = true_to_plot.shape[1]

        if frame_axis is None:
            frame_axis = np.arange(n_frames)
        else:
            frame_axis = np.asarray(frame_axis)
            if len(frame_axis) != n_frames:
                raise ValueError(
                    f"frame_axis length {len(frame_axis)} does not match n_frames {n_frames}"
                )

        n_rows = len(neuron_ids) + 1

        fig, axs = plt.subplots(
            n_rows,
            1,
            figsize=figsize,
            sharex=True
        )

        if n_rows == 1:
            axs = [axs]

        for i, neuron_idx in enumerate(neuron_ids):
            ax = axs[i]

            ax.plot(
                frame_axis,
                true_to_plot[neuron_idx, :],
                color=[0.25, 0.25, 0.25],
                linewidth=1.0,
                label=response_label,
            )

            ax.plot(
                frame_axis,
                pred_to_plot[neuron_idx, :],
                color=prediction_color,
                linewidth=1.0,
                label="Predicted",
            )

            ax.set_ylabel(f"N{neuron_idx}", fontsize=8)
            ax.legend(
                frameon=False,
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                fontsize=7,
            )

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # Velocity bottom panel
        ax = axs[-1]

        vel0_label, vel1_label = velocity_axis_order

        vel0 = avg_velocity[0, :]
        vel1 = avg_velocity[1, :]

        if scale_velocity:
            vel0 = self.minmax_for_display(np.abs(vel0))
            vel1 = self.minmax_for_display(np.abs(vel1))
            velocity_ylabel = "Velocity\nscaled"
            vel0_label = f"abs {vel0_label} velocity"
            vel1_label = f"abs {vel1_label} velocity"
        else:
            velocity_ylabel = "Velocity"

        ax.plot(
            frame_axis,
            vel0,
            linewidth=1.0,
            label=vel0_label,
        )

        ax.plot(
            frame_axis,
            vel1,
            linewidth=1.0,
            label=vel1_label,
        )

        for frame in event_frames:
            ax.axvline(x=frame, color='k', linestyle=':', alpha=0.5)

        ax.set_ylabel(velocity_ylabel, fontsize=7)
        ax.set_xlabel("Aligned frame")
        ax.legend(
            frameon=False,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            fontsize=7,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if title is None:
            total_trials = sum(fold_counts.values())
            title = (
                f"{key}, fold-averaged aligned GLM prediction, "
                f"{response_mode}, n_trials={total_trials}"
            )

        fig.suptitle(title, fontsize=10)

        plt.tight_layout()

        saved_path = None

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

            if save_name is None:
                save_name = (
                    f"{key}_fold_avg_aligned_true_vs_pred_"
                    f"{response_mode}.pdf"
                )

            saved_path = os.path.join(save_dir, save_name)
            plt.savefig(saved_path, bbox_inches="tight")
            print("Saved:", saved_path)

        plt.show()

        return fig, axs, saved_path, {
            "avg_true": avg_true,
            "avg_pred": avg_pred,
            "avg_velocity": avg_velocity,
            "fold_counts": fold_counts,
        }
    
    def scatter_dev_active_passive_sound_sig(self,
        dev_active_by_key,
        dev_passive_by_key,
        datasets,
        sound,
        sig_type = 'sig_cells',
        measure_string='Active',
        measure_string2='Passive',
        plot_lims=None,
        plot_lims_neg=None,
        save_path=None,
        sig_index_base=0
    ):
        """
        dev_active_by_key[key] = array, shape (n_neurons,)
        dev_passive_by_key[key] = array, shape (n_neurons,)

        sound[sig_type][0, dataset_idx] = indices of sound-responsive neurons
        """

        plt.rcParams.update({'font.size': 7, 'font.family': 'arial'})
        mpl.rcParams['pdf.fonttype'] = 42

        all_active = []
        all_passive = []
        all_is_sig = []

        for dataset_idx, (animalID, date, server) in enumerate(datasets):
            key = f'{animalID}_{date}'

            active_vals = np.asarray(dev_active_by_key[key]).squeeze()
            passive_vals = np.asarray(dev_passive_by_key[key]).squeeze()

            assert active_vals.shape == passive_vals.shape

            n_neurons = active_vals.shape[0]

            sig_cells = np.asarray(sound[sig_type][0, dataset_idx]).squeeze()

            if sig_cells.size == 0:
                sig_cells = np.array([], dtype=int)
            else:
                sig_cells = sig_cells.astype(int)

            # convert MATLAB 1-based indices to Python 0-based if needed
            if sig_index_base == 1:
                sig_cells = sig_cells - 1

            is_sig = np.zeros(n_neurons, dtype=bool)
            sig_cells = sig_cells[(sig_cells >= 0) & (sig_cells < n_neurons)]
            is_sig[sig_cells] = True

            all_active.append(active_vals)
            all_passive.append(passive_vals)
            all_is_sig.append(is_sig)

        active_plot = np.concatenate(all_active)
        passive_plot = np.concatenate(all_passive)
        is_sig_plot = np.concatenate(all_is_sig)

        good = (
            ~np.isnan(active_plot) &
            ~np.isnan(passive_plot)
        )

        active_plot = active_plot[good]
        passive_plot = passive_plot[good]
        is_sig_plot = is_sig_plot[good]

        fig, ax = plt.subplots(1, 1, figsize=(3, 3))

        if plot_lims is None:
            max_lim = np.nanmax([active_plot.max(), passive_plot.max(), 0.5])
        else:
            max_lim = plot_lims

        min_lim = 0 if plot_lims_neg is None else plot_lims_neg

        ax.plot(
            [min_lim, max_lim],
            [min_lim, max_lim],
            '--',
            color='black',
            zorder=1,
            label='Unity'
        )

        # non-sound-responsive cells
        ax.scatter(
            active_plot[~is_sig_plot],
            passive_plot[~is_sig_plot],
            c='lightgray',
            alpha=0.6,
            s=18,
            zorder=2,
            label='Not sound responsive'
        )

        # sound-responsive cells
        ax.scatter(
            active_plot[is_sig_plot],
            passive_plot[is_sig_plot],
            c='#4D3399', #purple
            alpha=0.8,
            s=22,
            zorder=3,
            label='Sound responsive'
        )

        ax.set_xlabel(measure_string, fontsize=7)
        ax.set_ylabel(measure_string2, fontsize=7)
        ax.set_title(f'{measure_string} vs {measure_string2}', fontsize=7)

        ax.set_xlim(min_lim, max_lim)
        ax.set_ylim(min_lim, max_lim)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_box_aspect(1)

        ax.legend(frameon=False, fontsize=6)

        new_rc_params = {
            'text.usetex': False,
            'svg.fonttype': 'none'
        }
        plt.rcParams.update(new_rc_params)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', format='pdf')

        plt.show()

        return active_plot, passive_plot, is_sig_plot
    

    def plot_within_between_cdf_two_contexts(self,all_df1, all_df2, group_colors=None,
                                            title='Coupling CDF',
                                            figsize=(4,4), save_path=None,
                                            mode='mean', x_ylim=None):



        mpl.rcParams['pdf.fonttype'] = 42
        plt.rcParams.update({'font.size': 7, 'font.family': 'arial'})

        def lighten_color(color, amount=0.6):
            c = np.array(mc.to_rgb(color))
            return tuple(c + (1 - c) * amount)

        def ecdf(data):
            data = np.sort(data)
            data = data[~np.isnan(data)]  # remove NaNs
            if len(data) == 0:
                return np.array([]), np.array([])
            y = np.arange(1, len(data)+1) / len(data)
            return data, y

        groups = sorted(list(set(all_df1['group']).union(set(all_df2['group']))))
        n_groups = len(groups)

        fig, axes = plt.subplots(1, n_groups,
                                figsize=(figsize[0]*n_groups, figsize[1]),
                                sharey=True)

        if n_groups == 1:
            axes = [axes]

        for i, group in enumerate(groups):

            ax = axes[i]

            subset1 = all_df1[all_df1['group'] == group]
            subset2 = all_df2[all_df2['group'] == group]

            base_color = group_colors.get(group, 'gray') if group_colors else 'gray'
            lighter_color = lighten_color(base_color)

            # Context 1
            x, y = ecdf(subset1['coupling_within'])
            ax.plot(x, y, color=base_color, linestyle='-', linewidth=1, label='within C1')

            x, y = ecdf(subset1['coupling_between'])
            ax.plot(x, y, color=base_color, linestyle='--', linewidth=1, label='between C1')

            # Context 2
            x, y = ecdf(subset2['coupling_within'])
            ax.plot(x, y, color=lighter_color, linestyle='-', linewidth=1, label='within C2')

            x, y = ecdf(subset2['coupling_between'])
            ax.plot(x, y, color=lighter_color, linestyle='--', linewidth=1, label='between C2')

            ax.set_title(group, fontsize=7)

            if x_ylim is not None:
                ax.set_xlim(x_ylim)

            ax.grid(True, alpha=0.3)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            ax.set_ylim([0,1])

        axes[0].set_ylabel('Fraction of Neurons')

        for ax in axes:
            ax.set_xlabel('Coupling Strength')
            if 'abs' in mode:
                ax.set_xlabel('|Coupling Strength|')

        axes[-1].legend(frameon=False, fontsize=6)

        plt.suptitle(title, fontsize=7)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)

        plt.show()

    def plot_aligned_population_average(self,
        aligned_true,
        aligned_pred,
        aligned_velocity,
        datasets,
        neuron_ids=None,
        folds=range(10),
        average_mode="neurons",
        response_mode="deconvolved",
        imaging_rate_hz=30.4791,
        smooth_sigma_sec=0.5,
        velocity_axis_order=("y", "x"),
        frame_axis=None,
        title=None,
        save_dir=None,
        save_name=None,
        prediction_color="deepskyblue",
        event_frames=None,
        event_labels=None,
        velocity_plot_mode="both" ):
        """
        Plot aligned true vs predicted traces averaged either across neurons or datasets.

        average_mode:
            "neurons"  = one dataset, average selected neurons, SEM across neurons
            "datasets" = multiple datasets, average each dataset first, SEM across datasets

        response_mode:
            "deconvolved" = plot mean deconvolved activity
            "rate" = smooth true activity for estimated event-rate display
        velocity_plot_mode: 
            "both"      # plot y and x
            "y"         # plot only y
            "x"         # plot only x
            "magnitude" # plot sqrt(x^2 + y^2)
        """


        if average_mode not in ["neurons", "datasets"]:
            raise ValueError("average_mode must be 'neurons' or 'datasets'")

        true_units = []
        pred_units = []
        velocity_units = []

        for dataset_idx, (animalID, date, server) in enumerate(datasets):
            key = f'{animalID}_{date}'
            avg_true, avg_pred, avg_velocity, fold_counts = self.average_aligned_across_folds(
                aligned_true=aligned_true,
                aligned_pred=aligned_pred,
                aligned_velocity=aligned_velocity,
                key=key,
                folds=folds,
                velocity_axis_order=velocity_axis_order,
            )

            # avg_true/avg_pred are neurons x frames
            # avg_velocity is 2 x frames

            if response_mode == "rate":
                avg_true = self.smooth_neural_trace_for_display(
                    avg_true,
                    imaging_rate_hz=imaging_rate_hz,
                    smooth_sigma_sec=smooth_sigma_sec,
                    axis=-1,
                )

                # I would leave prediction unsmoothed because GLM prediction is already rate-like.
                avg_pred = avg_pred

            elif response_mode != "deconvolved":
                raise ValueError("response_mode must be 'deconvolved' or 'rate'")

            if neuron_ids is None:
                use_neurons = np.arange(avg_true.shape[0])
            else:
                use_neurons =  np.asarray(neuron_ids[0, dataset_idx]).squeeze()  #np.asarray(neuron_ids)

            if average_mode == "neurons":
                if len(datasets) != 1:
                    raise ValueError("average_mode='neurons' expects one key/dataset at a time.")

                true_units = avg_true[use_neurons, :]       # neurons x frames
                pred_units = avg_pred[use_neurons, :]       # neurons x frames
                velocity_units = avg_velocity[None, :, :]   # 1 x 2 x frames

            elif average_mode == "datasets":
                true_units.append(np.nanmean(avg_true[use_neurons, :], axis=0))  # frames
                pred_units.append(np.nanmean(avg_pred[use_neurons, :], axis=0))  # frames
                velocity_units.append(avg_velocity)                              # 2 x frames

        if average_mode == "datasets":
            true_units = np.stack(true_units, axis=0)          # datasets x frames
            pred_units = np.stack(pred_units, axis=0)          # datasets x frames
            velocity_units = np.stack(velocity_units, axis=0)  # datasets x 2 x frames

        true_mean = np.nanmean(true_units, axis=0)
        pred_mean = np.nanmean(pred_units, axis=0)

        true_sem = np.nanstd(true_units, axis=0) / np.sqrt(true_units.shape[0])
        pred_sem = np.nanstd(pred_units, axis=0) / np.sqrt(pred_units.shape[0])

        n_frames = true_mean.shape[0]

        if frame_axis is None:
            frame_axis = np.arange(n_frames)
        else:
            frame_axis = np.asarray(frame_axis)
            if len(frame_axis) != n_frames:
                raise ValueError(
                    f"frame_axis length {len(frame_axis)} does not match n_frames {n_frames}"
                )

        fig, axs = plt.subplots(
            2,
            1,
            figsize=(8, 4.2),
            sharex=True,
            gridspec_kw={"height_ratios": [2.2, 1.0]},
        )

        ax = axs[0]

        ax.plot(
            frame_axis,
            true_mean,
            color=[0.25, 0.25, 0.25],
            linewidth=1.4,
            label="True",
        )

        ax.fill_between(
            frame_axis,
            true_mean - true_sem,
            true_mean + true_sem,
            color=[0.25, 0.25, 0.25],
            alpha=0.2,
            linewidth=0,
        )

        ax.plot(
            frame_axis,
            pred_mean,
            color=prediction_color,
            linewidth=1.4,
            label="Predicted",
        )

        ax.fill_between(
            frame_axis,
            pred_mean - pred_sem,
            pred_mean + pred_sem,
            color=prediction_color,
            alpha=0.2,
            linewidth=0,
        )

        if response_mode == "rate":
            ax.set_ylabel("Estimated\nrate", fontsize=9)
        else:
            ax.set_ylabel("Deconvolved\nactivity", fontsize=9)

        ax.legend(frameon=False, fontsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if event_frames is not None:         
            xlim = axs[0].get_xlim()
            for frame in event_frames:
                if frame < xlim[1]:
                    axs[0].axvline(x=frame, color='k', linestyle=(0, (10.5,6.8)), alpha=1,lw=0.7)
            axs[0].set_xticks(event_frames)
            axs[0].set_xticklabels(event_labels, rotation=45, fontsize=7)

        ax = axs[1]

        vel0_label, vel1_label = velocity_axis_order

        # -----------------------------
        # Velocity mean and SEM, unscaled
        # -----------------------------
        # velocity_units shape:
        #   average_mode="datasets": datasets x 2 x frames
        #   average_mode="neurons":  1 x 2 x frames
        #
        # velocity_axis_order tells us what row 0 and row 1 mean.
        # Example:
        #   velocity_axis_order=("y", "x")
        #   velocity_units[:, 0, :] = y velocity
        #   velocity_units[:, 1, :] = x velocity

        if velocity_units.ndim != 3 or velocity_units.shape[1] != 2:
            raise ValueError(
                f"Expected velocity_units to be units x 2 x frames, got {velocity_units.shape}"
            )

        velocity_traces_to_plot = {}

        if velocity_plot_mode == "both":
            velocity_units_abs = np.abs(velocity_units)

            velocity_traces_to_plot[f"abs {vel0_label} velocity"] = velocity_units_abs[:, 0, :]
            velocity_traces_to_plot[f"abs {vel1_label} velocity"] = velocity_units_abs[:, 1, :]

        elif velocity_plot_mode == "y":
            if vel0_label == "y":
                y_idx = 0
            elif vel1_label == "y":
                y_idx = 1
            else:
                raise ValueError("velocity_axis_order must contain 'y' for velocity_plot_mode='y'")

            velocity_traces_to_plot["abs y velocity"] = np.abs(velocity_units[:, y_idx, :])

        elif velocity_plot_mode == "x":
            if vel0_label == "x":
                x_idx = 0
            elif vel1_label == "x":
                x_idx = 1
            else:
                raise ValueError("velocity_axis_order must contain 'x' for velocity_plot_mode='x'")

            velocity_traces_to_plot["abs x velocity"] = np.abs(velocity_units[:, x_idx, :])

        elif velocity_plot_mode == "magnitude":
            if vel0_label == "y":
                y_idx = 0
            elif vel1_label == "y":
                y_idx = 1
            else:
                raise ValueError("velocity_axis_order must contain 'y' for velocity_plot_mode='magnitude'")

            if vel0_label == "x":
                x_idx = 0
            elif vel1_label == "x":
                x_idx = 1
            else:
                raise ValueError("velocity_axis_order must contain 'x' for velocity_plot_mode='magnitude'")

            y_vel = velocity_units[:, y_idx, :]
            x_vel = velocity_units[:, x_idx, :]

            velocity_traces_to_plot["velocity magnitude"] = np.sqrt((x_vel ** 2) + (y_vel ** 2))

        velocity_mean = {}
        velocity_sem = {}

        for trace_label, trace_units in velocity_traces_to_plot.items():
            velocity_mean[trace_label] = np.nanmean(trace_units, axis=0)

            if trace_units.shape[0] > 1:
                velocity_sem[trace_label] = np.nanstd(trace_units, axis=0) / np.sqrt(trace_units.shape[0])
            else:
                velocity_sem[trace_label] = np.zeros_like(velocity_mean[trace_label])

        # Plot selected velocity traces
        for trace_label in velocity_traces_to_plot.keys():
            trace_mean = velocity_mean[trace_label]
            trace_sem = velocity_sem[trace_label]

            ax.plot(
                frame_axis,
                trace_mean,
                linewidth=1.0,
                label=trace_label
            )

            ax.fill_between(
                frame_axis,
                trace_mean - trace_sem,
                trace_mean + trace_sem,
                alpha=0.2,
                linewidth=0,
            )

        if event_frames is not None:         
            xlim = axs[1].get_xlim()
            for frame in event_frames:
                if frame < xlim[1]:
                    axs[1].axvline(x=frame, color='k', linestyle=(0, (10.5,6.8)), alpha=1,lw=0.7)

            axs[1].set_xticks(event_frames)
            axs[1].set_xticklabels(event_labels, rotation=45, fontsize=7)

        ax.set_ylabel("Velocity", fontsize=7)
        # ax.set_xlabel("Aligned frame")
        ax.legend(frameon=False, fontsize=7)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if title is None:
            if average_mode == "neurons":
                title = f"{datasets[0]}, mean across neurons, SEM across neurons"
            else:
                title = f"Mean across datasets, SEM across datasets, n={len(datasets)}"

        fig.suptitle(title, fontsize=10)
        plt.tight_layout()

        saved_path = None

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

            if save_name is None:
                if average_mode == "neurons":
                    save_name = f"{datasets[0]}_aligned_mean_across_neurons_{response_mode}.pdf"
                else:
                    save_name = f"aligned_mean_across_datasets_{response_mode}.pdf"

            saved_path = os.path.join(save_dir, save_name)
            plt.savefig(saved_path, bbox_inches="tight")
            print("Saved:", saved_path)

        plt.show()

        return fig, axs, saved_path, {
            "true_units": true_units,
            "pred_units": pred_units,
            "velocity_units": velocity_units,
            "true_mean": true_mean,
            "pred_mean": pred_mean,
            "true_sem": true_sem,
            "pred_sem": pred_sem,
            "velocity_mean": velocity_mean,
            "velocity_sem": velocity_sem,
            "velocity_plot_mode": velocity_plot_mode,
                }

    @staticmethod
    def _dataset_key(animalID, date):
        return f"{animalID}_{date}"

    @staticmethod
    def _squeeze_1d(arr):
        arr = np.asarray(arr)
        if arr.dtype == object and arr.size == 1:
            arr = np.asarray(arr.item())
        arr = np.squeeze(arr)
        if arr.ndim == 0:
            arr = np.reshape(arr, (1,))
        if arr.size == 0:
            return np.array([], dtype=float)
        if arr.ndim != 1:
            raise ValueError(f"mask must be 1D after squeezing, got shape {arr.shape}")
        return arr

    def _extract_mask_entry(self, mask_array, dataset_idx):
        arr = np.asarray(mask_array, dtype=object)
        if arr.ndim == 2:
            if arr.shape[0] == 1:
                entry = arr[0, dataset_idx]
            elif arr.shape[1] == 1:
                entry = arr[dataset_idx, 0]
            elif dataset_idx < arr.shape[1]:
                entry = arr[0, dataset_idx]
            else:
                entry = arr[dataset_idx, 0]
        else:
            entry = arr[dataset_idx]
        return self._squeeze_1d(entry)

    def _boolean_mask_from_entry(self, mask_entry, n_neurons, key):
        """Convert a 1D mask entry to a boolean array of length n_neurons.

        If length matches n_neurons, treat as boolean/0-1.
        Otherwise treat as neuron indices.
        """
        mask_entry = self._squeeze_1d(mask_entry)
        if mask_entry.size == 0:
            return np.zeros(n_neurons, dtype=bool)

        if mask_entry.dtype == bool and mask_entry.size != n_neurons:
            raise ValueError(
                f"Mask length does not match MI length for {key!r}: "
                f"mask shape {mask_entry.shape}, MI shape ({n_neurons},)"
            )

        if mask_entry.size == n_neurons:
            return np.asarray(mask_entry).astype(bool)

        try:
            idx = np.asarray(mask_entry, dtype=int)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Mask for {key!r} has shape {mask_entry.shape} but MI has "
                f"length {n_neurons}. Expected a boolean mask of that length "
                f"or integer neuron indices."
            ) from exc

        if np.any(idx < 0) or np.any(idx >= n_neurons):
            raise ValueError(
                f"Mask indices for {key!r} are out of range for MI length "
                f"{n_neurons} (mask shape {mask_entry.shape})."
            )
        out = np.zeros(n_neurons, dtype=bool)
        out[idx] = True
        return out

    def make_sound_mask_dict(
        self,
        sound_source,
        datasets,
        mask_name="sig_cells",
        n_neurons_by_key=None,
        index_base=0,
    ):
        """Convert MATLAB-loaded sound masks into a per-dataset dictionary.

        Parameters
        ----------
        sound_source : dict-like
            `sound` for mask_name='sig_cells', or `sound_separated` for
            'sig_cells_pos' / 'sig_cells_neg'.
        datasets : list of (animalID, date, server)
        mask_name : {'sig_cells', 'sig_cells_pos', 'sig_cells_neg'}
        n_neurons_by_key : dict, optional
            If provided, values are converted to boolean arrays of that length.
        index_base : int
            0 if indices are already Python-style; 1 to convert from MATLAB.

        Returns
        -------
        mask_dict : dict
            mask_dict[key] is a 1D array. Boolean of shape (n_neurons,) if
            n_neurons_by_key is given, otherwise squeezed 1D indices/0-1.
        """
        allowed = {"sig_cells", "sig_cells_pos", "sig_cells_neg"}
        if mask_name not in allowed:
            raise ValueError(f"mask_name must be one of {sorted(allowed)}")

        if isinstance(sound_source, dict):
            if mask_name not in sound_source:
                raise KeyError(
                    f"{mask_name!r} not found in sound_source. "
                    "Pass `sound` for 'sig_cells' and `sound_separated` for "
                    "'sig_cells_pos' / 'sig_cells_neg'."
                )
            mask_array = sound_source[mask_name]
        else:
            mask_array = sound_source[mask_name]

        mask_dict = {}
        for dataset_idx, (animalID, date, *rest) in enumerate(datasets):
            key = self._dataset_key(animalID, date)
            entry = self._extract_mask_entry(mask_array, dataset_idx)
            if entry.size > 0 and index_base:
                entry = np.asarray(entry, dtype=int) - int(index_base)
            if n_neurons_by_key is not None:
                if key not in n_neurons_by_key:
                    raise KeyError(
                        f"n_neurons_by_key is missing dataset key {key!r}"
                    )
                n_neurons = int(n_neurons_by_key[key])
                mask_dict[key] = self._boolean_mask_from_entry(entry, n_neurons, key)
            else:
                mask_dict[key] = entry
        return mask_dict

    def _ordered_dataset_keys(self, mi_context_dicts, datasets=None):
        first = mi_context_dicts[0]
        if datasets is None:
            keys = [k for k in first.keys()]
        else:
            keys = []
            for animalID, date, *rest in datasets:
                key = self._dataset_key(animalID, date)
                if key in first:
                    keys.append(key)
        for d in mi_context_dicts[1:]:
            keys = [k for k in keys if k in d]
        return keys

    def _selected_mi_for_dataset(self, mi_arr, neuron_mask_dict, key):
        mi_arr = np.asarray(mi_arr, dtype=float).squeeze()
        if mi_arr.ndim != 1:
            raise ValueError(
                f"MI for {key!r} must be 1D, got shape {mi_arr.shape}"
            )
        n_neurons = mi_arr.shape[0]
        if neuron_mask_dict is None:
            return mi_arr
        if key not in neuron_mask_dict:
            raise KeyError(f"neuron_mask_dict is missing dataset key {key!r}")
        mask = self._boolean_mask_from_entry(neuron_mask_dict[key], n_neurons, key)
        if mask.shape[0] != n_neurons:
            raise ValueError(
                f"Mask length does not match MI length for {key!r}: "
                f"mask shape {mask.shape}, MI shape {mi_arr.shape}"
            )
        return mi_arr[mask]

    def plot_mi_heatmap_by_context(
        self,
        mi_context_dicts,
        context_labels,
        neuron_mask_dict=None,
        sort_context_index=0,
        vmin=-0.4,
        vmax=0.4,
        title=None,
        save_path=None,
        datasets=None,
        cmap="RdBu_r",
    ):
        """Heatmap of neuron MI across contexts (rows=neurons, columns=contexts).

        Neurons are concatenated across datasets and sorted by MI in
        ``sort_context_index`` (default: first context, e.g. Active).
        """
        if len(mi_context_dicts) != len(context_labels):
            raise ValueError("mi_context_dicts and context_labels must have the same length")

        plt.rcParams.update({"font.size": 7, "font.family": "arial"})
        mpl.rcParams["pdf.fonttype"] = 42

        keys = self._ordered_dataset_keys(mi_context_dicts, datasets=datasets)
        per_context = [[] for _ in mi_context_dicts]
        for key in keys:
            selected = [
                self._selected_mi_for_dataset(d[key], neuron_mask_dict, key)
                for d in mi_context_dicts
            ]
            n = selected[0].shape[0]
            for arr in selected[1:]:
                if arr.shape[0] != n:
                    raise ValueError(
                        f"Selected neuron counts differ across contexts for {key!r}"
                    )
            for i, arr in enumerate(selected):
                per_context[i].append(arr)

        if not per_context[0]:
            raise ValueError("No neurons available to plot")

        mi_matrix = np.column_stack(
            [np.concatenate(vals) for vals in per_context]
        )
        sort_vals = mi_matrix[:, sort_context_index]
        order = np.argsort(np.nan_to_num(sort_vals, nan=-np.inf))[::-1]
        mi_sorted = mi_matrix[order, :]

        fig, ax = plt.subplots(figsize=(1.4 + 0.45 * mi_sorted.shape[1], 4.2))
        colors = plt.get_cmap(cmap)(np.linspace(0, 1, 256))
        cmap_obj = mpl.colors.ListedColormap(colors)
        cmap_obj.set_bad(color=(0.7, 0.7, 0.7))
        masked = np.ma.masked_invalid(mi_sorted)
        im = ax.imshow(
            masked,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap_obj,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xticks(np.arange(len(context_labels)))
        ax.set_xticklabels(context_labels, rotation=45, ha="right")
        ax.set_ylabel("Neurons")
        ax.set_yticks([])
        if title:
            ax.set_title(title, fontsize=7)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Modulation Index")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        plt.show()
        return fig, ax, mi_sorted

    def summarize_abs_mi_by_dataset(
        self,
        mi_context_dicts,
        context_labels,
        neuron_mask_dict=None,
        datasets=None,
    ):
        """Mean |MI| per dataset and context over selected neurons."""
        if len(mi_context_dicts) != len(context_labels):
            raise ValueError("mi_context_dicts and context_labels must have the same length")

        keys = self._ordered_dataset_keys(mi_context_dicts, datasets=datasets)
        rows = []
        for key in keys:
            for context_label, mi_dict in zip(context_labels, mi_context_dicts):
                selected = self._selected_mi_for_dataset(
                    mi_dict[key], neuron_mask_dict, key
                )
                rows.append(
                    {
                        "dataset": key,
                        "context": context_label,
                        "mean_abs_mi": np.nanmean(np.abs(selected)) if selected.size else np.nan,
                        "n_neurons": int(np.sum(np.isfinite(selected))),
                    }
                )
        return pd.DataFrame(rows)

    def plot_abs_mi_paired_summary(
        self,
        summary_df,
        context_order=None,
        title=None,
        ylabel="|Modulation Index|",
        save_path=None,
        line_color=(0.55, 0.55, 0.55),
        mean_color="k",
    ):
        """Paired dataset lines across contexts, with mean ± SEM overlay."""
        plt.rcParams.update({"font.size": 7, "font.family": "arial"})
        mpl.rcParams["pdf.fonttype"] = 42

        df = summary_df.copy()
        if context_order is None:
            context_order = list(dict.fromkeys(df["context"].tolist()))
        x = np.arange(len(context_order))

        fig, ax = plt.subplots(figsize=(2.4, 2.6))
        for dataset, sub in df.groupby("dataset", sort=False):
            ys = []
            for ctx in context_order:
                match = sub.loc[sub["context"] == ctx, "mean_abs_mi"]
                ys.append(float(match.iloc[0]) if len(match) else np.nan)
            ax.plot(
                x,
                ys,
                "-",
                color=line_color,
                alpha=0.45,
                lw=0.8,
                marker="o",
                ms=2.5,
                markerfacecolor=line_color,
                markeredgecolor="none",
            )

        means = []
        sems = []
        for ctx in context_order:
            vals = df.loc[df["context"] == ctx, "mean_abs_mi"].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            means.append(np.mean(vals) if vals.size else np.nan)
            sems.append(
                np.std(vals, ddof=1) / np.sqrt(vals.size) if vals.size > 1 else np.nan
            )
        ax.errorbar(
            x,
            means,
            yerr=sems,
            fmt="o",
            color=mean_color,
            ms=3.5,
            lw=1.1,
            capsize=2,
            zorder=3,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(context_order, rotation=45, ha="right")
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title, fontsize=7)
        ax.set_xlim(-0.4, len(context_order) - 0.6)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(0, max(ymax, 0.05))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
        plt.show()
        return fig, ax

    def plot_true_vs_pred_abs_mi_summary(
        self,
        mi_true_dict,
        mi_pred_dict,
        neuron_mask_dict=None,
        title=None,
        save_path=None,
        datasets=None,
        true_label="True",
        pred_label="Predicted",
    ):
        """Paired mean |MI| summary of true vs predicted activity per dataset."""
        summary_df = self.summarize_abs_mi_by_dataset(
            [mi_true_dict, mi_pred_dict],
            [true_label, pred_label],
            neuron_mask_dict=neuron_mask_dict,
            datasets=datasets,
        )
        fig, ax = self.plot_abs_mi_paired_summary(
            summary_df,
            context_order=[true_label, pred_label],
            title=title,
            ylabel="|Modulation Index|",
            save_path=save_path,
        )
        return fig, ax, summary_df


