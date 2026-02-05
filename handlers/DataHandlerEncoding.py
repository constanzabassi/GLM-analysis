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

#from .Plotter import Plotter as plotter

class DataHandlerEncoding:
    def __init__(self, data):
        self.data = data

    #FUNCTION TO LOAD SAVED RESULTS (SAVES TIME)
    def load_pkls(self,directory, keyword):
        """
        Load a specific pickle file from a directory based on a keyword.
        
        Parameters:
        directory (str): The directory containing the pickle files.
        keyword (str): The keyword to identify the correct file (e.g., "iti", "pre", "pass").
        
        Returns:
        object: The loaded pickle data if the file is found.
        
        Raises:
        FileNotFoundError: If no file with the given keyword is found.
        """
        # List all files in the directory
        files = os.listdir(directory)
        
        # Search for the file that contains the keyword
        for file in files:
            if keyword in file and file.endswith('.pkl'):
                filepath = os.path.join(directory, file)
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                print(f"Loaded file: {file}")
                return data
        
        # Raise an error if no matching file is found
        raise FileNotFoundError(f"No file with keyword '{keyword}' found in directory {directory}.")
        
    def save_pkls(self, directory, data, keyword, overwrite=False):
        """
        Save data to a pickle file in a directory using a keyword in the filename.

        Parameters:
        directory (str): Directory to save the pickle file.
        data (object): Python object to pickle.
        keyword (str): Keyword to include in the filename (e.g., "iti", "pre", "pass").
        overwrite (bool): Whether to overwrite an existing matching file.

        Returns:
        str: Path to the saved pickle file.
        """
        os.makedirs(directory, exist_ok=True)

        # Check for existing files with same keyword
        existing_files = [
            f for f in os.listdir(directory)
            if keyword in f and f.endswith(".pkl")
        ]

        if existing_files and not overwrite:
            raise FileExistsError(
                f"File(s) with keyword '{keyword}' already exist: {existing_files}"
            )

        # Optional timestamp to avoid accidental overwrites
        
        filename = f"results_{keyword}.pkl"
        filepath = os.path.join(directory, filename)

        with open(filepath, "wb") as f:
            pickle.dump(data, f)

        print(f"Saved file: {filename}")
        return filepath

    #FUNCTIONS TO LOAD DATASETS!
    def load_GLM_results_cluster(self,save_directory, save_string):
        os.chdir(save_directory)
        # Identify all neuron files
        neuron_files = [filename for filename in os.listdir(save_directory) if filename.startswith(save_string)]
        
        # Extract neuron IDs and sort filenames numerically
        neuron_files_sorted = sorted(neuron_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))

        model_output_all = {}  # Create dictionary structure
        for filename in neuron_files_sorted: #can change to specific neurons in here [0:381]
            print(filename)
            neuron_id = filename.split('_')[-1].split('.')[0]  # Extract neuron ID from filename
            neuron_path = os.path.join(save_directory, filename)

            with open(neuron_path, 'rb') as file:
                model_data = pickle.load(file)
            

            for fold, model_output in model_data.items():

                if fold not in model_output_all:
                    model_output_all[fold] = {}
                for key, value in model_output.items():
                    if key not in model_output_all[fold]:
                        model_output_all[fold][key] = []

                    if key in ['B_weights','y_pred','loss_trace','lambda_trace']:
                        model_output_all[fold][key].append(value)
                    elif isinstance(value, list):
                        model_output_all[fold][key].extend(value)
                    else:
                        model_output_all[fold][key].append(value)

        # Convert lists of arrays to single NumPy arrays where applicable
        for fold, outputs in model_output_all.items():
            for key, value in outputs.items():
                if isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                    if key == 'y_pred':
                        model_output_all[fold][key] = np.stack(value, axis=1)  # Transpose to neurons x frames
                        model_output_all[fold][key] = np.squeeze(model_output_all[fold][key])  # Remove extra dimension
                    elif key == 'B_weights':
                        print(f'size beta {np.shape(value)}')
                        model_output_all[fold][key] = np.stack(value, axis=2)  # Transpose to features x neurons
                        model_output_all[fold][key] = np.squeeze(model_output_all[fold][key])  # Remove extra dimension
                    else:
                        model_output_all[fold][key] = np.concatenate(value, axis=0)
                else:
                    model_output_all[fold][key] = value

                # if fold not in model_output_all:
                #     model_output_all[fold] = {'frac_dev_expl': []}  # Initialize frac_dev_expl list
                # if 'frac_dev_expl' in model_output:  # Check if 'frac_dev_expl' exists in model_output
                #     model_output_all[fold]['frac_dev_expl'].extend(model_output['frac_dev_expl'])

                
                # if fold not in model_output_all:
                #     model_output_all[fold] = {}
                # model_output_all[fold][neuron_id] = model_output

                # print(f"Neuron {neuron_id}, Fold {fold}:")
                # print(model_output['frac_dev_expl'][:5])  # To make sure no outputs are empty
        return model_output_all


    ## load cell types! - INPUT MOUSE NAME AND DATE
    def load_celltypes(self,server,animalID,date):
        base_path = f"{server}/Connie/ProcessedData/{animalID}/{date}/"


        path = os.path.join(base_path, 'red_variables/')
        pyr_str = scipy.io.loadmat(path+'pyr_cells.mat')
        pyr = pyr_str['pyr_cells']-1 #convert to python indices
        pyr = np.transpose(pyr)

        som_str = scipy.io.loadmat(path+'mcherry_cells.mat')
        som = som_str['mcherry_cells']-1

        pv_str = scipy.io.loadmat(path+'tdtom_cells.mat')
        pv = pv_str['tdtom_cells']-1


        neuron_groups = {
            'pyr': pyr,
            'som': som,
            'pv': pv}

        # Define colors for each group
        colors = {'pyr': (0.37, 0.75, 0.49),   # pyr = 0
                'som': (0.17, 0.35, 0.8),    # som = 1
                'pv': (0.82, 0.04, 0.04)}    # pv = 2

        #combine celltypes into different indices
        celltype_array = np.zeros(np.shape(np.concatenate((pyr,som,pv)))[0])
        celltype_array[som] = 1
        celltype_array[pv] = 2

        return celltype_array, neuron_groups, colors

    def load_data(self, animalID, date, server, model_type, results_type='results', models_to_load=None):
        """
        Load and process GLM results for a given dataset.
        
        Parameters:
            animalID (str): The animal ID.
            date (str): The date of the dataset.
            server (str): The server location.
            model_type (str): The type of the GLM model.
            results_type (str): The type of results to load (default is 'results').
            models_to_load (list): List of model indices to load (default [0,1,2,3,4])

        Returns:
            dict: A dictionary containing the mean deviance explained for each model.
        """

        if models_to_load is None:
            models_to_load = [0, 1, 2, 3, 4]

        save_directory_v1 = os.path.join(f'{server}/Connie/ProcessedData/{animalID}/{date}/{model_type}', results_type)

        # Mapping model indices to names and file prefixes
        model_info = {
            0: ("model_output_behav", "poss_model_0_data_cluster_"), #behavior only model
            1: ("model_output_all", "poss_model_1_data_cluster_"), #using factors for coupling predictors
            2: ("model_output_no_pyr", "poss_model_2_data_cluster_"),
            3: ("model_output_no_som", "poss_model_3_data_cluster_"),
            4: ("model_output_no_pv", "poss_model_4_data_cluster_"),
            5: ("model_output_all_neurons", "poss_model_5_data_cluster_") #using indiividual neurons instead of factors for coupling predictors
        }

        loaded_models = {}
        mean_devs = {}

        # Load only requested models
        for model_idx in models_to_load:
            if model_idx not in model_info:
                raise ValueError(f"Model {model_idx} not recognized")

            name, prefix = model_info[model_idx]

            output = self.load_GLM_results_cluster(save_directory_v1, prefix)
            loaded_models[name] = output

            # Compute mean deviance explained
            mean_devs[f"mean_dev_{model_idx}"] = np.mean(
                [model_output['frac_dev_expl'] for model_output in output.values()],
                axis=0
            )

        print("Loaded models:", list(loaded_models.keys()))

        # Load cell types
        celltype_array, neuron_groups, colors = self.load_celltypes(server, animalID, date)

        results = {
            **mean_devs,
            **loaded_models,
            'celltype_array': celltype_array,
            'neuron_groups': neuron_groups,
            'colors': colors
        }

        return results



    def process_multiple_datasets(self,datasets, model_type, results_type = 'results', models_to_load = None):
        """
        Process multiple datasets and calculate mean deviance explained for each.

        Parameters:
            datasets (list of tuples): List of tuples containing (animalID, date, server).
            model_type (str): The type of the GLM model.

        Returns:
            dict: A dictionary where keys are dataset identifiers and values are results.
        """
        all_results = {}

        for animalID, date, server in datasets:
            key = f'{animalID}_{date}'
            print(f'Processing dataset: {key}')
            results = self.load_data(animalID, date, server, model_type, results_type =results_type,  models_to_load= models_to_load)
            all_results[key] = results

        return all_results


    #LOAD INFO STRUCTURE
    def load_info(self,directory):
        """
        Load mouse dates and associated keys from the specified directory.

        Parameters:
            directory (str): The directory containing the info.mat file.

        Returns:
            list: A list of tuples containing (animalID, date, server).
            list: A list of mouse date keys.
        """
        opto_dir = directory #'V:/Connie/results/active/mod' #

        # Load the condition_array_trials structure
        mat_data = scipy.io.loadmat(os.path.join(opto_dir,'info.mat'))
        info = mat_data['info'][0][0]

        # Assuming your mouse_date structure is loaded as a numpy array
        mouse_dates_keys = [
            item[0].replace('\\', '_').replace('/', '_')  # Replace slashes with underscores for consistency
            for item in info['mouse_date'][0]
        ]

        mouse_dates = []
        for item,server in zip(info['mouse_date'][0],info['serverid'][0]):
            current_item = item[0].replace('\\', '_').replace('/', '_')  # Replace both slashes with underscores
            
            parts = current_item.split('_')  # Split the modified string by underscore
            
            # Assuming animalID is the first part and date is the last part
            animalID = parts[0]  # Assuming animal ID is the first part
            date = parts[-1]  # Assuming date is the last part

            # Append as a tuple in the format (animalID, date, server)
            mouse_dates.append((animalID, date, server[0]))

        return mouse_dates, mouse_dates_keys

    #FUNCTIONS TO AGGREGATE DATASETS TOGETHER
    def aggregate_model_data_from_results(self,all_results, no_abs=1, significant_neurons=None):
        """
        Aggregates model data from the all_results dictionary across multiple datasets.

        Parameters:
        - all_results: Dictionary containing model results from multiple datasets.
        - no_abs: Whether to use absolute values for weights.
        - significant_neurons: Optional dictionary of significant neurons for each dataset.

        Returns:
        - Aggregated model data including weights, feature names, cell types, and indices.
        """
        aggregated_data = {
            'B_weights': [],
            'feature_names': None,
            'coupling_indices': None,
            'celltype_array': [],
            'neuron_groups': {},
            'start_indices': []
        }

        # Load actual response data (should be the same across datasets!!!)
        behav_big_matrix_ids_mat = scipy.io.loadmat(
            os.path.join('V:/Connie/ProcessedData/HA11-1R/2023-05-05/GLM_3nmf_iti/prepost trial cv 73 #1', 
                        'behav_big_matrix_ids.mat')
        )
        behav_big_matrix_ids = behav_big_matrix_ids_mat['behav_big_matrix_ids']
        feature_names = [name[0] for name in behav_big_matrix_ids[0]]  # Flatten the structure

        current_index = 0

        for dataset_key, dataset in all_results.items():
            print(dataset_key)
            model_output_all = dataset['model_output_all']
            celltype_array = dataset['celltype_array']
            
            if aggregated_data['feature_names'] is None:
                aggregated_data['feature_names'] = feature_names
            
            # Retrieve significant neurons for this dataset if provided
            if significant_neurons and dataset_key in significant_neurons:
                sig_neurons = significant_neurons[dataset_key][0]
                sig_neurons = np.array(sig_neurons, dtype=np.uint16)
            else:
                sig_neurons = None

            # Aggregate B_weights across folds for coupling predictors
            B_weights_behavior_coupling = np.concatenate(
                [model_output_all[fold]['B_weights'] for fold in model_output_all.keys()], axis=1
            )

            num_neurons = B_weights_behavior_coupling.shape[1]

            # Apply significant neurons filtering if applicable
            if sig_neurons is not None:
                B_weights_behavior_coupling = B_weights_behavior_coupling[:,sig_neurons]
                celltype_array = np.array(celltype_array)[sig_neurons].tolist()
                num_neurons = len(sig_neurons)
            # Update indices for cell types
            
            start_index = current_index
            end_index = current_index + num_neurons

            # Aggregate cell type data
            aggregated_data['celltype_array'].extend(celltype_array)
            
            # Store the start index for each dataset
            aggregated_data['start_indices'].append(start_index)
            
            # Append to aggregated data
            aggregated_data['B_weights'].append(B_weights_behavior_coupling)
            
            # Assuming coupling_indices are the same across datasets
            if aggregated_data['coupling_indices'] is None:
                coupling_indices = range(183, B_weights_behavior_coupling.shape[0])
                aggregated_data['coupling_indices'] = coupling_indices
            
            current_index += num_neurons

        # Combine B_weights from all datasets
        combined_B_weights = np.concatenate(aggregated_data['B_weights'], axis=1)
        
        # Process weights for plotting
        if no_abs == 1:
            other_weights = combined_B_weights.mean(axis=1)
            coupling_weights = combined_B_weights[aggregated_data['coupling_indices'], :].mean(axis=1)
        else:
            other_weights = np.abs(combined_B_weights).mean(axis=1)
            coupling_weights = np.abs(combined_B_weights[aggregated_data['coupling_indices'], :]).mean(axis=1)

        # Map cell types to labels
        cell_type_labels = {0: 'pyr', 1: 'som', 2: 'pv'}  # Update if necessary
        labeled_cell_types = [cell_type_labels.get(cell_type, 'unknown') for cell_type in aggregated_data['celltype_array']]

        # Create neuron_groups as a dictionary
        neuron_groups = {}
        for index, label in enumerate(labeled_cell_types):
            if label not in neuron_groups:
                neuron_groups[label] = []
            neuron_groups[label].append(index)

        # Sort the neuron_groups dictionary by a predefined order of keys
        desired_order = ['pyr', 'som', 'pv']
        neuron_groups = {key: np.array(neuron_groups[key], dtype=np.uint16).reshape(-1, 1)
                        for key in desired_order if key in neuron_groups}
        aggregated_data['neuron_groups'] = neuron_groups

        return combined_B_weights, aggregated_data['feature_names'], other_weights, coupling_weights, aggregated_data['coupling_indices'], aggregated_data['celltype_array'], aggregated_data['neuron_groups'], aggregated_data['start_indices']

    def combine_model_output_all(self,all_results):
        combined_model_output_all = {}
        current_index = 0

        for dataset_key, dataset in all_results.items():
            model_output_all = dataset['model_output_all']

            for fold, metrics in model_output_all.items():
                if fold not in combined_model_output_all:
                    combined_model_output_all[fold] = {
                        'frac_dev_expl': [],
                        'dev_model': [],
                        'dev_null': [],
                        'dev_expl': [],
                        'B_weights': [],
                        'intercept_weight': [],
                        'y_pred': [],
                        'selec_lambda': []
                    }

                # Append or concatenate the data
                # Append the data to lists, but skip 'y_pred'
                for key in combined_model_output_all[fold].keys():
                    if key == 'y_pred':
                        combined_model_output_all[fold][key].append(metrics[key][:1999,:]) #append first 2000 frames
                    else: 
                        combined_model_output_all[fold][key].append(metrics[key])

        # Convert lists to numpy arrays
        for fold, metrics in combined_model_output_all.items():
            for key in metrics.keys():
                if isinstance(metrics[key][0], np.ndarray) and metrics[key][0].ndim == 1:
                    # Concatenate along axis 0 for 1D arrays
                    combined_model_output_all[fold][key] = np.concatenate(metrics[key], axis=0)
                else:
                    # Concatenate along axis 1 for 2D or higher arrays
                    combined_model_output_all[fold][key] = np.concatenate(metrics[key], axis=1)

        return combined_model_output_all
    
    def load_sig_neurons_modindex(self,opto_or_not, opto_dir, sound_dir, act_dir):
        """
        Loads neuron IDs and their modulation indices based on the provided directories and optogenetic condition.

        Parameters:
        opto: int
            Indicator for optogenetic condition. If 1, loads opto data; otherwise, loads passive or sound data.
        opto_dir: str
            Directory for opto data.
        sound_dir: str
            Directory for sound data (for non-opto condition).
        act_dir: str
            Directory for activity data (for non-opto condition).

        Returns:
        significant_neurons: dict
            A dictionary mapping mouse_date to its significant neurons.
        mod_index_neurons: dict
            A dictionary mapping mouse_date to its corresponding modulation index for significant neurons.
        mouse_dates: list
            A list of mouse dates after formatting.
        """
        
        # Load the condition_array_trials structure
        mat_data = scipy.io.loadmat(os.path.join(opto_dir, 'info.mat'))
        info = mat_data['info'][0][0]

        # Assuming your mouse_date structure is loaded as a numpy array
        mouse_dates = [
            item[0].replace('\\', '_').replace('/', '_')  # Replace slashes with underscores for consistency
            for item in info['mouse_date'][0]
        ]

        # Load data based on opto condition
        if opto_or_not == 1:
            # Load opto-specific modulation indices and significant cells
            mod_idx = scipy.io.loadmat(os.path.join(opto_dir, 'mod_indexm.mat'))
            mod_index = mod_idx['mod_indexm'][0]
            sig_cels = scipy.io.loadmat(os.path.join(opto_dir, 'sig_mod_boot_thr.mat'))
            sig_cells = sig_cels['sig_mod_boot_thr'][0] - 1  # Adjust for MATLAB indexing
        else:
            # Load passive or sound data
            mod_idx = scipy.io.loadmat(os.path.join(act_dir, 'combined_mod_index.mat'))
            mod_index = mod_idx['combined_mod_index'][0]
            sig_cels = scipy.io.loadmat(os.path.join(sound_dir, 'sig_mod_boot_thr01.mat'))
            sig_cells = sig_cels['sig_mod_boot_thr01'][0] - 1  # Adjust for MATLAB indexing

            sig_cels = scipy.io.loadmat(os.path.join(sound_dir,'combined_thres.mat'))
            sig_cells = sig_cels['combined_thres'][0]-1 #minus one bc of MATLAB indexing

        # Initialize dictionaries to store results
        significant_neurons = {}
        mod_index_neurons = {}

        # Iterate over mouse_dates and map to corresponding neurons in sig_cells by index
        for idx, mouse_date in enumerate(mouse_dates):
            if idx < len(sig_cells):
                significant_neurons_al = sig_cells[idx]  # Significant neurons for the current mouse
                significant_neurons[mouse_date] = significant_neurons_al
                current_mod = mod_index[idx][0]
                mod_index_neurons[mouse_date] = current_mod[significant_neurons_al[0]]
            else:
                print(f"No significant neurons found for {mouse_date}")

        return significant_neurons, mod_index_neurons, mouse_dates
    

    def unwrap_matlab(self,data):

        for field in data.dtype.names:

            arr = data[field]

            if isinstance(arr, np.ndarray):

                fixed = np.empty(arr.shape, dtype=object)

                for idx in np.ndindex(arr.shape):

                    val = arr[idx]

                    # FIRST unwrap 2D arrays with one row
                    if isinstance(val, np.ndarray) and val.ndim == 2 and val.shape[0] == 1:
                        val = val[0]

                    # THEN if that result is still a (1,1) object array, unwrap again
                    if isinstance(val, np.ndarray) and val.shape == (1,):
                        val = val[0]

                    fixed[idx] = val

                data[field] = fixed

        return data
    
    def load_sound_opto_data(self,dir, set_diff = True, exclude=None):
        """
        Loads neuron IDs and their modulation indices based on the provided directories and optogenetic condition.

        Parameters:
        dir: str
            Directory for opto data.


        Returns:
        significant_neurons: dict
            A dictionary mapping mouse_date to its significant neurons.
        mod_index_neurons: dict
            A dictionary mapping mouse_date to its corresponding modulation index for significant neurons.
        mouse_dates: list
            A list of mouse dates after formatting.
        """
        
        # Load the condition_array_trials structure
        mat_data = scipy.io.loadmat(os.path.join(dir, 'opto.mat'))
        opto = mat_data['opto'][0][0]
        opto = self.unwrap_matlab(opto)

        mat_data = scipy.io.loadmat(os.path.join(dir, 'sound.mat'))
        sound = mat_data['sound'][0][0]
        sound = self.unwrap_matlab(sound)

            # Load the info structure to get mouse_date

        mat_data = scipy.io.loadmat(os.path.join(dir, 'info.mat'))
        info = mat_data['info'][0][0]    

        # Assuming your mouse_date structure is loaded as a numpy array
        mouse_dates = [
            item[0].replace('\\', '_').replace('/', '_')  # Replace slashes with underscores for consistency
            for item in info['mouse_date'][0]
        ]

        # ---- NEW: process exclusion argument ----
        if exclude is not None:
            # convert single value to list
            if not isinstance(exclude, (list, tuple)):
                exclude = [exclude]
            exclude_indices = []
            for ex in exclude:
                if isinstance(ex, int):
                    exclude_indices.append(ex)
                elif isinstance(ex, str) and ex in mouse_dates:
                    exclude_indices.append(mouse_dates.index(ex))
            # remove duplicates
            exclude_indices = list(set(exclude_indices))
        else:
            exclude_indices = []

        significant_neurons = {}
        mod_indices = {}
        # Iterate over mouse_dates and map to corresponding neurons in sig_cells by index
        for idx, mouse_date in enumerate(mouse_dates):
            # ---- NEW: skip excluded datasets ----
            if idx in exclude_indices:
                continue

            significant_neurons[mouse_date] = {}
            opto_neurons = opto['sig_cells'][idx,0]-1 # Adjust for MATLAB indexing
            sound_neurons = sound['sig_cells'][0,idx]-1 # Adjust for MATLAB indexing

            mod_indices[mouse_date] = {}

            for context in range(opto['mod'].shape[1]):
                mod_indices[mouse_date][context] = {}   # initialize context dictionary
                mod_indices[mouse_date][context]['opto'] = opto['mod'][idx, context]
            for context in range(sound['mod'].shape[1]):
                if context not in mod_indices[mouse_date]:
                    mod_indices[mouse_date][context] = {}
                mod_indices[mouse_date][context]['sound'] = sound['mod'][idx, context]

            #get total nuerons
            all_neurons = list(range(opto['mod'][idx,0].shape[0]))#range(0,opto['mod'][idx,0].shape[0])
            
            #get significant neurons
            if set_diff:
                significant_neurons[mouse_date]['opto'] = list(set(opto_neurons) - set(sound_neurons))
                significant_neurons[mouse_date]['sound'] = list(set(sound_neurons) - set(opto_neurons))
                significant_neurons[mouse_date]['both'] = list(set(opto_neurons).intersection(set(sound_neurons)))
                significant_neurons[mouse_date]['unmod'] = list(set(all_neurons) - set(opto_neurons) - set(sound_neurons))
            else:   
                significant_neurons[mouse_date]['opto'] = opto_neurons
                significant_neurons[mouse_date]['sound'] = sound_neurons
                significant_neurons[mouse_date]['both'] = list(set(opto_neurons).intersection(set(sound_neurons)))
                significant_neurons[mouse_date]['unmod'] = list(set(all_neurons) - set(opto_neurons) - set(sound_neurons))

        return opto, sound, mouse_dates, significant_neurons, mod_indices

