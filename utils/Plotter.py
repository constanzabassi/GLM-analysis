import numpy as np
import os
import pickle
import scipy
import h5py
import random

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
import itertools

#IMPORT PLOTTING FUNCTIONS!
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import wilcoxon
from analysis.AnalysisManagerEncoding import AnalysisManagerEncoding as analysisenc #using bonferroni correction

class Plotter:
    def __init__(self, data, celltypecolors=None, save_results=None, color_map_dict = None, event_frames=None):
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

        # Default cell type colors
        self.default_colors = {
            'pyr': (0.37, 0.75, 0.49),
            'som': (0.17, 0.35, 0.8),
            'pv': (0.82, 0.04, 0.04)
        }

        # Default variable colors (pairs for regular and shuffled) for each decoded variable
        self.default_variable_colors = {
            'sound_category': ['steelblue', 'lightskyblue'],
            'choice': ['saddlebrown', 'darkorange'],
            'photostim': ['darkslateblue', 'mediumslateblue'],
            'outcome': ['mediumvioletred', 'hotpink']
        }
        
        # Use custom colors if provided, otherwise use defaults
        self.celltypecolors = celltypecolors if celltypecolors is not None else self.default_colors

        self.default_cell_type_labels = {
            'pyr': 'Pyr',
            'som': 'SOM',
            'pv': 'PV'
        }
        # Use custom colors if provided, otherwise use defaults
        self.cell_type_labels = celltypecolors if celltypecolors is not None else self.default_cell_type_labels

    def add_significance_line(self,ax, x1, x2=None, y=None, significance='', color='black'):
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
            y = ylims[1] * 0.95

        if x2 is not None:  # Draw line if both x1 and x2 are provided
            # Calculate line height as small percentage of y-axis range
            line_height = (ylims[1] - ylims[0]) * 0.02
            line_y = y 
            text_y = y + line_height

            # Draw the line
            ax.plot([x1, x1, x2, x2], [y, line_y, line_y, y], 
                    lw=1.5, color=color)
            
            # Add text
            ax.text((x1 + x2) * 0.5, text_y, significance, 
                    ha='center', va='bottom', color=color, fontsize=14)
        else:  # Only x1 is provided, so draw only the significance star
            if y is not None:
                ax.text(x1, y, significance, 
                    ha='center', va='bottom', color=color, fontsize=14)
                    
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
    def plot_feature_weights(self,server,animalID, date, model_type, model_chosen, pyr_count=3, som_count=3, pv_count=3, no_abs=1):
        # Load actual response data
        behav_big_matrix_ids_mat = scipy.io.loadmat(
            os.path.join(f'{server}/Connie/ProcessedData/{animalID}/{date}/{model_type}/prepost trial cv 73 #1', 
                        'behav_big_matrix_ids.mat')
        )
        behav_big_matrix_ids = behav_big_matrix_ids_mat['behav_big_matrix_ids']
        feature_names = [name[0] for name in behav_big_matrix_ids[0]]  # Flatten the structure

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



    def plot_weights_heatmap(self,server,animalID, date, model_type, model_chosen, coupling_indices, cmap='coolwarm', no_abs=1, minmax=(-.2,.2)):
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
        behav_big_matrix_ids_mat = scipy.io.loadmat(
                os.path.join(f'{server}/Connie/ProcessedData/{animalID}/{date}/{model_type}/prepost trial cv 73 #1', 
                            'behav_big_matrix_ids.mat')
            )
        behav_big_matrix_ids = behav_big_matrix_ids_mat['behav_big_matrix_ids']
        feature_names = [name[0] for name in behav_big_matrix_ids[0]]
        unique_feature_names = list(set(feature_names))

        # Initialize an array to store the mean weights
        num_neurons = model_chosen[0]['B_weights'].shape[1]
        mean_weights = np.zeros((len(feature_names)+len(coupling_indices), num_neurons))
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
        other_weights = mean_weights[:183, :]  # Assuming non-coupling predictors are the first 183 rows
        coupling_weights = mean_weights[coupling_indices, :]


        # Combine other weights and coupling weights for heatmap
        combined_weights = np.vstack([other_weights, coupling_weights])


        
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

        return mean_neuron_feature_unique,updated_feature_names,mean_weights,feature_names,total_beta_feature_unique,max_beta_feature_unique

    # # Example usage
    # mean_neuron_feature_unique,unique_feature_names,mean_weights = plot_weights_heatmap(animalID= animalID, date= date, model_type= model_type,
    #                      model_output=model_output_all, coupling_indices = coupling_predictors_indices, save_results=save_results, no_abs=0)

    def unique_features_heatmap_celltypes(self,mean_neuron_feature_unique,behav_features_unique,neuron_groups, minmax=(-.2,.2),model_type = None,no_abs=1):
        """
        Create a heatmap to show average across unique features for all neurons
        """
        fig, ax = plt.subplots(1,3, figsize = (20,8))

        #get color palette
        palette = sns.color_palette("vlag", as_cmap=True)#sns.cubehelix_palette(start=.9, rot=-.95, as_cmap=True)#'viridis'#sns.color_palette("Blues", as_cmap=True)#sns.cubehelix_palette(start=.5, rot=-.75, as_cmap=True)

        for i, (group, cel_indices) in enumerate(neuron_groups.items()):
            
            sns.heatmap(np.squeeze(mean_neuron_feature_unique[:,cel_indices]),vmin= minmax[0], vmax = minmax[1], cmap = palette , ax=ax[i], cbar=False) #model_output_all[0]['B_weights']
            ax[i].set_xlabel(group, fontsize=14)
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
        plt.savefig(f'heatmap_avg{no_abs}_uniquebeta_celltypes_{model_type}.png')  
        plt.show()

    # unique_features_heatmap_celltypes(mean_neuron_feature_unique,unique_feature_names,neuron_groups,save_results)



    def scatter_plot_weights_overlay(self,neuron_groups, mean_neuron_feature_unique, updated_feature_names, model_type,animalID = None, date = None,no_abs=1,minmax=(-.1,.8)):
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
        plt.rcParams.update({'font.size': 14, 'font.family': 'arial'})

        # Initialize the plot
        plt.figure(figsize=(3,3))

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
                        fmt='o', color='white', ecolor=self.celltypecolors[group], capsize=5, 
                        label=group, alpha=1, markersize=10, markeredgewidth=2, markeredgecolor=self.celltypecolors[group])
            
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
        # ax.set_aspect('equal', adjustable='box')
        # plt.axis('scaled')
        
        # Save the figure
        if animalID is not None:
            plt.savefig(f'{self.save_results}/scatter_overlay_weights_avg{no_abs}_{animalID}_{date}_{model_type}.pdf')
        else:
            plt.savefig(f'{self.save_results}/scatter_overlay_weights_avg{no_abs}_{animalID}_{date}_{model_type}.pdf')
        plt.show()



    def scatter_plot_weights_overlay_noerror(self,neuron_groups, mean_neuron_feature_unique, updated_feature_names, model_type,animalID = None, date = None,no_abs=1,minmax=(-.1,.8),save_string = None):
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
        plt.rcParams.update({'font.size': 14, 'font.family': 'arial'})

        # Initialize the plot
        plt.figure(figsize=(3,3))

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
                        label=group, alpha=1, s=70, linewidths= 2)
            
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
                plt.savefig(f'{self.save_results}/scatter_overlay_updatedweights_avg{no_abs}_{animalID}_{date}_{model_type}_{save_string}.svg')
            else:
                plt.savefig(f'{self.save_results}/scatter_overlay_updatedweights_avg{no_abs}_{animalID}_{date}_{model_type}.svg')
        else:
            if save_string is not None:
                plt.ylabel(fr'{save_string} |$\beta$ Weights|')
                plt.savefig(f'{self.save_results}/scatter_overlay_updatedweights_avg{no_abs}_{model_type}_{save_string}.svg')
            else:
                plt.savefig(f'{self.save_results}/scatter_overlay_updatedweights_avg{no_abs}_{model_type}.svg')
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
        colors = sns.color_palette("vlag", as_cmap=True)
        sns.heatmap(np.squeeze(mean_neuron_feature_unique[specified_features,:]),vmin= minmax[0], vmax= minmax[1], cmap = colors) #model_output_all[0]['B_weights']

        # Graph title
        ax.set_title('Average Weights Across Features', fontsize=14)
            
        # Label x and y-axis
        ax.set_ylabel('Behavioral Features', fontsize=14)
        ax.set_xlabel('Neurons', fontsize=14)

        #set y labels
        # unique_feature_indices = {str(unique_f): idx for idx, unique_f in enumerate(behav_features_unique)}
        # Convert each element of the NumPy array to a regular Python string
        behav_features_unique_str = [str(label) for label in behav_features_ids]
        # Remove square brackets from the labels
        behav_features_unique_str = [label[1:-1] if label.startswith('[') and label.endswith(']') else label for label in behav_features_unique_str]
        ax.set_yticklabels( behav_features_unique_str)
        ax.tick_params(axis ='y', labelrotation =0)

        # Label x-axis ticks
        # ax.set_xticklabels(neuron_groups.keys(), fontsize=14)

        # Hide x-axis major ticks
        ax.tick_params(axis='x', which='major', length=0)

        
        # # Clean up the appearance
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_box_aspect(1)


    def scatter_model_dev_comparison(self,mean_deviance1, mean_deviance2, cell_ids, measure_string, measure_string2, colors=None, plot_lims= None, save_path = None, plot_lims_neg = None):
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
        plt.rcParams.update({'font.size': 14, 'font.family': 'arial'})
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        
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
        plt.plot([0, max_dev], [0, max_dev], '--', color='black', zorder = 1, label='Unity Line')


        # Create scatter plot
        # plt.scatter(mean_deviance2, mean_deviance1,linewidths=2, edgecolors=colors,facecolors = 'None', alpha=0.7, zorder = 2)# c = colors
        plt.scatter(mean_deviance2, mean_deviance1,c = colors, alpha=0.7, zorder = 2)# c = colors


        
        # Set labels and title
        plt.xlabel(f'{measure_string}', fontsize=14)
        plt.ylabel(f'{measure_string2}', fontsize=14)
        plt.title(f'{measure_string} vs {measure_string2}', fontsize=14)

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
            plt.savefig(save_path, bbox_inches='tight', format = 'svg')
        
        # Show plot
        plt.show()  


    def box_plot(self, data, neuron_groups, colors, measure_string, save_path = None): #plotting function for box plots to compare fraction deviance across celltypes
        """
        Create a box-and-whisker plot with significance bars.
        """
        # Set global font size and family 
        plt.rcParams.update({'font.size': 14, 'font.family': 'arial'})

        fig, ax = plt.subplots(1,1, figsize = (3,3))
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
        ax.set_title(f'{measure_string} Across Cell types', fontsize=14)
            
        # Label x and y-axis
        ax.set_ylabel(f'{measure_string}', fontsize=14)
        ax.set_xlabel('Cell type', fontsize=14)

        # Label x-axis ticks
        ax.set_xticks(positions)
        ax.set_xticklabels(self.cell_type_labels.values(), fontsize=14) #neuron_groups.keys()

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
        plt.rcParams.update({'font.size': 14, 'font.family': 'arial'})
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
        plt.rcParams.update({'font.size': 14, 'font.family': 'arial'})

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
        ax.set_title(f'{measure_string} Across Cell types', fontsize=14)
        ax.set_ylabel(f'{measure_string}', fontsize=14)
        ax.set_xlabel('Cell type', fontsize=14)
        ax.set_xticks(positions)
        ax.set_xticklabels(self.cell_type_labels.values(), fontsize=14) #neuron_groups.keys()
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
        plt.rcParams.update({'font.size': 14, 'font.family': 'arial'})

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
            ax.set_xlabel('Difference in Deviance Explained', fontsize=14)
            ax.set_ylabel('Frequency', fontsize=14)
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
        plt.rcParams.update({'font.size': 14, 'font.family': 'arial'})

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
            ax.set_title(f'{cell_type}', fontsize=14)
            ax.set_xticks(positions)
            ax.set_xticklabels(data.keys(), rotation=45, ha='right', fontsize=10)
            ax.tick_params(axis='x', which='major', length=0)

            # Clean up the appearance
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if i == 0:
                ax.set_ylabel(f'{measure_string}', fontsize=14)

            ax.set_ylim(minmax[0],minmax[1])
        
        # Add a global title for the figure
        fig.suptitle(f'{measure_string} Across Cell Types and Models', fontsize=16)
        
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
        plt.rcParams.update({'font.size': 14, 'font.family': 'arial'})

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
        plt.rcParams.update({'font.size': 14, 'font.family': 'arial'})
        
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
        xticks_in, xticks_lab = self.x_axis_sec_aligned(event_frames[0], len(x), interval=1, frame_rate=30) 
        # Set x-ticks to be in seconds
        for ax in plt.gcf().axes:
            # Ensure we only set x-ticks for the last plot
            if ax == plt.gcf().axes[-1]:
                plt.xticks(ticks=xticks_in, labels=xticks_lab)
                plt.xlabel('Time (s)')
            else:
                ax.set_xticks([])
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


    def plot_significant_neurons_distribution(self, significant_neurons_data, save_path = None):
        """Plot distribution of significant neurons."""
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))

        for celltype, color in self.celltypecolors.items():
            all_peaks = []
            all_peaks_locs = []

            for dataset in significant_neurons_data:
                peaks = significant_neurons_data[dataset][celltype]['peak_values']
                peaks_locs = significant_neurons_data[dataset][celltype]['peak_frames']

                all_peaks.extend(peaks)
                all_peaks_locs.extend(peaks_locs)

            axes[0].hist(all_peaks, alpha=1.0, color=color, label=celltype, histtype='step', linewidth=2)
            axes[1].hist(all_peaks_locs, alpha=1.0, color=color, label=celltype, histtype='step', linewidth=2)

        axes[0].set_xlabel('Information (bits)')
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)

        axes[1].set_xlabel('Peak Frame')
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)

        fig.suptitle('Significant Neurons Distribution by Cell Type')
        plt.tight_layout()
        plt.show()

        # Save plot if save_path is provided
        if save_path: 
            plt.savefig(save_path, bbox_inches='tight')

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
        ax.axvline(x=event_onset, color='k', linestyle=(0, (5, 5)), alpha=0.5)

        plt.xticks(ticks=xticks_in, labels=xticks_lab)
        plt.xlabel('Time (s)')

        plt.show()

    def plot_significant_neuron_percentages_by_celltype(self, significance_struc, neuron_groups, save_path=None):
        """
        Plot the percentage of significantly modulated neurons per dataset for each cell type and all neurons combined.
        
        Parameters:
        -----------
        significance_struc : dict
            Dictionary containing significant neuron data by dataset and cell type, including 'sig_neurons_all'
        neuron_groups : dict
            Dictionary containing all neuron indices for each dataset, organized by cell type
        save_path : str, optional
            Path to save the plot
        """
        plt.rcParams.update({'font.size': 14, 'font.family': 'arial'})
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
        fig, ax = plt.subplots(figsize=(3,3))
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
        ax.set_ylabel("% Modulated Neurons", fontsize=14)
        #ax.set_title("Significantly Modulated Neurons Across Cell Types", fontsize=14)
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
        plt.rcParams.update({'font.size': 14, 'font.family': 'arial'})
        
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
    def simple_bar_plot(self,labels, means, sems,colors = ['blue','red'], title='Bar Plot', ylabel='Value', save_dir=None):
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

        fig, ax = plt.subplots(figsize=(3,3))
        # Set global font size and family 
        plt.rcParams.update({'font.size': 14, 'font.family': 'arial'})
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
                stat, p_value = wilcoxon(data[i, frames], data[j, frames])
                all_p_values.append(p_value)
                comparisons.append((i, j))

        corrected_p_values, significance_stars = analysisenc.calculate_bonferroni_significance(self,all_p_values, alpha=0.05)

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
                    plt.axvline(x=(frame - start_frame) / 30.0, color='k', linestyle=(0, (5, 5)), alpha=0.5)
            
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



    # def plot_selected_metric_with_sem(self, mean_results_all, mean_results_all_passive, decoder_type, metric, start_frame = None,end_frame = None, xlim=None, ylim=None, title=None, xlabel='Frames', ylabel=None, colors = ('blue','red'), save_dir=None):
    #     """
    #     Plot the selected metric from mean_results_all and mean_results_all_passive on the same plot with SEM shading.
        
    #     Parameters:
    #     - mean_results_all: dict, results for the active condition
    #     - mean_results_all_passive: dict, results for the passive condition
    #     - decoder_type: str, the type of decoder used
    #     - metric: str, the metric to plot
    #     - plot_type: str, type of plot ('sc' for single cell, 'pop' for population)
    #     - xlim: tuple, x-axis limits
    #     - ylim: tuple, y-axis limits
    #     - title: str, title of the plot
    #     - xlabel: str, label for the x-axis
    #     - ylabel: str, label for the y-axis
    #     - save_dir: str, directory to save the plot
    #     """
    #     # Get event frames from first dataset
    #     first_dataset = list(mean_results_all.keys())[0]
    #     event_frames = mean_results_all[first_dataset][decoder_type]['event_frame_mean']
        
    #     # Collect data across datasets for active condition
    #     all_data_active = []
    #     for dataset in mean_results_all.keys():
    #         if decoder_type in mean_results_all[dataset]:
    #             data = mean_results_all[dataset][decoder_type][metric]

    #             # Average across neurons for sc data
    #             if 'sc' in metric and len(data.shape) == 2:  # frames x neurons
    #                 data = np.mean(data, axis=1)  # average across neurons

    #             all_data_active.append(data)
        
    #     # Collect data across datasets for passive condition
    #     all_data_passive = []
    #     for dataset in mean_results_all_passive.keys():
    #         if decoder_type in mean_results_all_passive[dataset]:
    #             data = mean_results_all_passive[dataset][decoder_type][metric]

    #             # Average across neurons for sc data
    #             if 'sc' in metric and len(data.shape) == 2:  # frames x neurons
    #                 data = np.mean(data, axis=1)  # average across neurons

    #             all_data_passive.append(data)

        
    #     if end_frame is None:
    #         end_frame = data.shape[0]
    #     if start_frame is None:
    #         start_frame = data.shape[0]

    #     used_frames = np.arange(start_frame, end_frame) # frames to use for plotting    
    #     # Convert lists to NumPy arrays
    #     all_data_active = np.array(all_data_active)
    #     all_data_passive = np.array(all_data_passive)
        
    #     # Calculate mean and SEM for active condition
    #     all_data_active_final = all_data_active[:,used_frames]
    #     mean_trace_active = np.mean(all_data_active_final, axis=0)
    #     sem_trace_active = np.std(all_data_active_final, axis=0) / np.sqrt(len(all_data_active_final))

    #     # Calculate mean and SEM for passive condition
    #     all_data_passive_final = all_data_passive[:,used_frames]  
    #     mean_trace_passive = np.mean(all_data_passive_final, axis=0)
    #     sem_trace_passive = np.std(all_data_passive_final, axis=0) / np.sqrt(len(all_data_passive_final))

    #     # Plot the metric values
    #     plt.figure(figsize=(3.5,3))
    #     x = np.arange(len(mean_trace_active))
        
    #     # Plot active condition
    #     active_line, = plt.plot(mean_trace_active, color=colors[0], label='Active')
    #     plt.fill_between(x, mean_trace_active - sem_trace_active, mean_trace_active + sem_trace_active, alpha=0.3, color=colors[0]) #, label='Active SEM'
        
    #     # Plot passive condition
    #     passive_line, = plt.plot(mean_trace_passive, color=colors[1], label='Passive')
    #     plt.fill_between(x, mean_trace_passive - sem_trace_passive, mean_trace_passive + sem_trace_passive, alpha=0.3, color=colors[1]) #, label='Passive SEM'
        
    #     # Add event markers
    #     for frame in event_frames:
    #         if frame < len(mean_trace_active):
    #             plt.axvline(x=frame - start_frame, color='k', linestyle=':', alpha=0.5)
        
    #     # Formatting
    #     plt.title(title)
    #     plt.xlabel(xlabel)
    #     if ylabel:
    #         plt.ylabel(ylabel)
    #     else:
    #         plt.ylabel('Bits' if 'information' in metric else 'Fraction Correct')
        
    #     plt.xlim(xlim if xlim else (0+ start_frame, len(mean_trace_active)- start_frame))
    #     if ylim:
    #         plt.ylim(ylim)
        
    #     # Add text annotations for the labels
    #     plt.text(xlim[1]-20, ylim[1], 'Active', color=colors[0], verticalalignment='center')
    #     plt.text(xlim[1]-20, ylim[1] - ylim[1]*.1, 'Passive', color=colors[1], verticalalignment='center')

    #     # # Custom legend with colored text
    #     # legend_labels = ['Active', 'Passive']
    #     # legend_colors = colors
    #     # handles = [plt.Line2D([0], [0], color='w', markerfacecolor=color, markersize=10, marker='o') for color in legend_colors]

    #     # #handles = [active_line, passive_line]
    #     # legend = plt.legend(handles, legend_labels,shadow = None, frameon = False,   loc='center left', bbox_to_anchor=(1, 0.5)) #loc='upper right',
    #     # for text, color in zip(legend.get_texts(), legend_colors):
    #     #     text.set_color(color)

    #     # plt.legend(shadow = None, frameon = False,  loc='upper right')

    #     plt.tight_layout()

    #     # Clean up the appearance
    #     ax = plt.gca()  # get current axis  
    #     ax.spines['top'].set_visible(False)
    #     ax.spines['right'].set_visible(False)
        
    #     # Save the plot if save_dir is provided
    #     if save_dir:
    #         plt.savefig(f"{save_dir}/{metric}_context_comparison_traces.png")
        
    #     plt.show()
    #     return mean_trace_active, sem_trace_active, mean_trace_passive, sem_trace_passive

    # def plot_single_neuron_analysis(results_dict, decoder_type='sound_category', start_frame= 14, end_frame = None):
    #     print(start_frame)
    #     """Comprehensive single neuron decoding visualization"""
        
    #     # 1. Neuron Performance Heatmap
    #     plt.figure(figsize=(12, 8))
    #     for dataset in results_dict:
    #         data = results_dict[dataset][decoder_type]['sc_cumulative_information_mean']
    #         celltype_array = results_dict[dataset]['celltype_array']
            
    #         # Sort neurons by cell type and performance
    #         max_info = np.max(data[start_frame:, :], axis=0)
    #         sort_idx = np.argsort(max_info)
            
    #         plt.subplot(len(results_dict), 1, list(results_dict.keys()).index(dataset) + 1)
    #         sns.heatmap(data[:, sort_idx].T, 
    #                 cmap='viridis',
    #                 xticklabels=20,
    #                 yticklabels=False)
    #         plt.title(f'{dataset} Single Neuron Decoding')
    #     plt.tight_layout()
        
    #     # 2. Best Neurons Analysis
    #     # Dictionary to store neuron IDs for each dataset and cell type
    #     neuron_ids_by_dataset = {}
    #     fig, axes = plt.subplots(1, 2, figsize=(6, 3))  # 1 row, 2 columns
    #     for cel_index,(celltype, color) in enumerate(plotter.celltypecolors.items()):
    #         all_peaks = []
    #         all_peaks_locs = []
    #         for dataset in results_dict:
    #             # Initialize a dictionary for this dataset if not already present
    #             if dataset not in neuron_ids_by_dataset:
    #                 neuron_ids_by_dataset[dataset] = {}

    #             # Initialize a list for this cell type in the current dataset
    #             if celltype not in neuron_ids_by_dataset[dataset]:
    #                 neuron_ids_by_dataset[dataset][celltype] = []
                
    #             peaks_by_celltype = analyze_peaks_by_celltype( results_dict, decoder_type=decoder_type, start_frame=start_frame, end_frame = end_frame)
    #             peaks = peaks_by_celltype[dataset][celltype]['sc']['sc_instantaneous_information_mean']['peak_values']
    #             peaks_locs = peaks_by_celltype[dataset][celltype]['sc']['sc_instantaneous_information_mean']['peak_frames']    
    #             if len(peaks) > 0:
    #                 max_peaks = sorted(peaks)  # Sorted in ascending order
    #                 top_5 = max_peaks[-5:]     # Slice the last 5 elements (highest values)
    #                 all_peaks.extend(top_5)

    #                 # Get the corresponding peak locations  
    #                 top_5_locs = [peaks_locs[max_peaks.index(p)] for p in top_5]
    #                 # Use the indices from the sorting step to ensure uniqueness
    #                 sorted_indices = np.argsort(peaks)
    #                 top_5_indices = sorted_indices[-5:]  # Indices of the top 5 values
    #                 neuron_ids_by_dataset[dataset][celltype].extend(top_5_indices.tolist())


    #                 # Add neuron IDs (indices) to the dictionary
    #                 #neuron_ids_by_dataset[dataset][celltype].extend(top_5_ids)

    #                 all_peaks_locs.extend(top_5_locs)
    #         # Plotting histograms on subplots
    #         axes[0].hist(all_peaks, alpha=1.0, color=color, label=celltype, histtype='step', linewidth=2)
    #         axes[0].set_xlabel('Information (bits)')  # Correct method to set the x-axis label
    #         axes[0].spines['top'].set_visible(False)
    #         axes[0].spines['right'].set_visible(False)
    #         axes[0].set_box_aspect(1)

    #         axes[1].hist(all_peaks_locs, alpha=1.0, color=color, label=celltype, histtype='step', linewidth=2)
    #         axes[1].set_xlabel('Peak Frame')  # Correct method to set the x-axis label
    #         axes[1].spines['top'].set_visible(False)
    #         axes[1].spines['right'].set_visible(False)
    #         axes[1].set_box_aspect(1)

    #     # Add legend and title
    #     #fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)  # Adjust legend location
    #     fig.suptitle('Top 5 Neurons Distribution by Cell Type')

    #     # Adjust layout to avoid overlap
    #     plt.tight_layout() #(rect=[0, 0.03, .8, 0.95]) #(rect=[0, 0.03, 1, 0.95])  # Leave space for the title and legend

    #     # Show the plot
    #     plt.show()
        
    #     # 3. Time Course by Cell Type
    #     plt.figure(figsize=(3, 3))
    #     for cel_index,(celltype, color) in enumerate(plotter.celltypecolors.items()):
    #         all_traces = []
    #         for dataset in results_dict:
    #             if end_frame is None:
    #                 end_frame = len(data)

    #             traces = results_dict[dataset][decoder_type]['sc_instantaneous_information_mean'][0:end_frame,:]

    #             celltype_idx = results_dict[dataset]['celltype_array'] == cel_index
    #             if np.any(celltype_idx):
    #                 mean_trace = np.mean(traces[:, celltype_idx], axis=1)
    #                 all_traces.append(mean_trace)
            
    #         mean = np.mean(all_traces, axis=0)
    #         sem = np.std(all_traces, axis=0) / np.sqrt(len(all_traces))
    #         plt.plot(mean, color=color, label=celltype)
    #         # Get the current Axes object
    #         ax = plt.gca()   
    #         ax.axvline(x = start_frame, color='k', linestyle=':', alpha=0.5)
    #         plt.fill_between(range(len(mean)), mean-sem, mean+sem, alpha=0.2, color=color)
    #         ax.spines['top'].set_visible(False)
    #         ax.spines['right'].set_visible(False)
            
    #         ax.set_box_aspect(1)
        
    #     plt.legend()
    #     plt.title('Average Information Time Course by Cell Type')
    #     plt.xlabel('Time (frames)')
    #     plt.ylabel('Information (bits)')
        
    #     plt.show()

    #     return neuron_ids_by_dataset
