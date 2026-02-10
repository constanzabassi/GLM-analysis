import numpy as np
import os
import pickle
import scipy
import random
import h5py
from pathlib import Path
import argparse
import re
import scipy.io

class DataHandlerDecoding:
    """
    Class for handling decoding results from GLM analysis across multiple datasets.
    
    Attributes:
        decoded_variables (set): Set of variables to decode
        cat_results (dict): Dictionary to store concatenated results
        mean_results (dict): Dictionary to store mean results per split
        mean_results_all (dict): Dictionary to store overall mean results
    """
        
    def __init__(self, decoded_variables = None):
        # if decoded_variables is None:
        #     decoded_variables = {'sound_category', 'choice', 'photostim', 'outcome',
        #                         'shuffled/sound_category', 'shuffled/choice', 'shuffled/photostim', 'shuffled/outcome'} 
        # else:  
        #     # self.decoded_variables = decoded_variables  
        #     # Convert input to list if it's a set
        #     self.decoded_variables = (
        #         list(decoded_variables) if isinstance(decoded_variables, set) 
        #         else decoded_variables
        #     )

        # Default variables in a specific order
        default_variables = [
            'sound_category', 'choice', 'photostim', 'outcome',
            'shuffled/sound_category', 'shuffled/choice', 'shuffled/photostim', 'shuffled/outcome'
        ]
        
        if decoded_variables is None:
            self.decoded_variables = default_variables
        else:
            # Convert set to ordered list, ensuring shuffled versions come after non-shuffled
            if isinstance(decoded_variables, set):
                regular_vars = sorted([v for v in decoded_variables if not v.startswith('shuffled/')])
                shuffled_vars = sorted([v for v in decoded_variables if v.startswith('shuffled/')])
                self.decoded_variables = regular_vars + shuffled_vars
            else:
                self.decoded_variables = list(decoded_variables)
        self.cat_results = {}
        self.mean_results = {}
        self.mean_results_all = {}
        self.celltype_info = {}

    #FUNCTION TO LOAD SAVED RESULTS (SAVES TIME)
    def load_all_decoder_results(filepath):
            def process_reference(ref, file):
                try:
                    if isinstance(ref, h5py.h5r.Reference):
                        referenced_obj = file[ref]
                        if isinstance(referenced_obj, h5py.Dataset):
                            return referenced_obj[()]
                        return process_group(referenced_obj, file)
                    return ref
                except Exception as e:
                    # Silent error handling
                    return None

            def process_dataset(dataset, file):
                try:
                    if dataset.dtype == np.dtype('O'):
                        refs = dataset[()]
                        if isinstance(refs, np.ndarray):
                            return [process_reference(ref, file) for ref in refs.flat]
                        return process_reference(refs, file)
                    return dataset[()]
                except Exception as e:
                    return None

            def process_group(group, file):
                result = {}
                for key in group.keys():
                    try:
                        item = group[key]
                        if isinstance(item, h5py.Group):
                            result[key] = process_group(item, file)
                        elif isinstance(item, h5py.Dataset):
                            result[key] = process_dataset(item, file)
                    except Exception as e:
                        pass
                return result

            try:
                with h5py.File(filepath, 'r') as file:            
                    if 'decoder_results' not in file:
                        return None
                        
                    decoder_group = file['decoder_results']
                    result = process_group(decoder_group, file)
                    return result
                    
            except Exception as e:
                print(f"File loading error: {e}")
                return None

        # try:
        #     mat_path = Path('decoder_results_regular_choice.mat')
        #     results = load_all_decoder_results(mat_path)
        #     if results:
        #         print("\nSuccessfully loaded results")
        #         print("Top level keys:", list(results.keys()))
                
        # except Exception as e:
        #     print(f"Error: {e}")

        # np.shape(results['inputs']['choice']['sc_cumulative_decoded_variables'][0]) # frames x trials x neurons
        # np.shape(results['aligned']['choice']['cat_results'][0]['sc_cumulative_fraction_correct']) # frames x neurons x shuffles



    def load_cat_results(self, filepath, decoded_variables):
        def process_reference(ref, file):
            if isinstance(ref, h5py.h5r.Reference):
                referenced_obj = file[ref]
                if isinstance(referenced_obj, h5py.Dataset):
                    return referenced_obj[()]
                return process_group(referenced_obj, file)
            return ref

        def process_dataset(dataset, file):
            if dataset.dtype == np.dtype('O'):  # Object references
                refs = dataset[()]
                if isinstance(refs, np.ndarray):
                    return [process_reference(ref, file) for ref in refs.flat]
                return process_reference(refs, file)
            return dataset[()]

        def process_group(group, file):
            return {
                key: process_group(item, file) if isinstance(item := group[key], h5py.Group)
                else process_dataset(item, file)
                for key in group.keys()
            }

        try:
            # with h5py.File(filepath, 'r') as file:
            #     target_path = f'decoder_results/aligned/{decoded_variables}/cat_results'
            #     if target_path not in file:
            #         print(f"[WARN] '{target_path}' path not found in {filepath}")
            #         return None

            #     cat_results = file[target_path]

            #     result = (
            #         process_dataset(cat_results, file)
            #         if isinstance(cat_results, h5py.Dataset)
            #         else process_group(cat_results, file)
            #     )

            #     # Ensure consistent return type
            #     if isinstance(result, dict):
            #         return [result]   # wrap single struct in list

            #     return result

            with h5py.File(filepath, 'r') as file:
                # Construct the dynamic path based on the variable
                target_path = f'decoder_results/aligned/{decoded_variables}/cat_results'
                if target_path not in file:
                    print(f"'{target_path}' path not found")
                    return None

                cat_results = file[target_path]
                if isinstance(cat_results, h5py.Dataset):
                    return process_dataset(cat_results, file)
                return process_group(cat_results, file)
        except Exception as e:
            print(f"Error loading decoded results for '{decoded_variables}': {e}")
            return None

    # Example usage
    # mat_path = Path('decoder_results_regular_choice.mat')
    # variable = 'choice'  # Change to 'sound_category', 'photostim', etc.
    # cat_results = load_cat_results(mat_path, variable)
    # cat_results = cat_results[0]  # to get rid of the first dimension


    def get_cat_results_across_datasets(self,decoding_dir,decoded_variables, single_balanced=False):    
        cat_results = {}
        for splits in range(0,10):
            #decoding_dir =f'V:/Connie\ProcessedData\HA11-1R/2023-04-13\GLM_3nmf_pre\decoding/{splits+1}/'
            if single_balanced is True:
                os.chdir(f'{decoding_dir}{splits+1}_1/')
                print(f'{decoding_dir}{splits+1}_1/')
            else:
                {'Loading regular results without single balance'}
                os.chdir(f'{decoding_dir}{splits+1}/')
                print(f'{decoding_dir}{splits+1}/')
            for variable in decoded_variables:
                if variable.startswith('shuffled/'):
                    new_variable = variable[9:]
                    mat_path = Path(f'decoder_results_shuffled_{new_variable}.mat')
                    
                    # Try loading from current directory first
                    if not mat_path.exists() and 'pre' in decoding_dir:
                        # If file doesn't exist and we're in pre, try regular directory
                        fallback_dir = f'{decoding_dir}{splits+1}/'
                        os.chdir(fallback_dir)
                        print(f'Shuffled file not found, trying fallback directory: {fallback_dir}')
                else:
                    mat_path = Path(f'decoder_results_regular_{variable}.mat')
                    # variable = variable.split('_')[0]

                
                print(mat_path)
                variable = variable  # Change to 'sound_category', 'photostim', etc.
                # temp_results = self.load_cat_results(mat_path, variable)
                # temp_results = temp_results[0]  # to get rid of the first dimension
                # ## skip this variable if file doesn’t exist
                # if not mat_path.exists():
                #     print(f"[WARN] Missing {variable} for split {splits} → skipping")
                #     continue

                # Define variables to extract
                variables_to_load = [
                    'pop_instantaneous_information',
                    'pop_cumulative_information',
                    'pop_instantaneous_fraction_correct',
                    'pop_cumulative_fraction_correct',
                    'sc_instantaneous_information',
                    'sc_cumulative_information',
                    'sc_instantaneous_fraction_correct',
                    'sc_cumulative_fraction_correct',
                    'event_frame' #have to subtract one to this value because it is in MATLAB indexing
                ]

                # Initialize the dictionary for the variable
                if variable not in cat_results:
                    cat_results[variable] = {}
                
                # Initialize the dictionary for the splits
                if splits not in cat_results[variable]:
                    cat_results[variable][splits] = {}

                # If file is missing → insert EMPTY fields
                if not mat_path.exists():
                    print(f"[WARN] Missing {mat_path}, inserting empty structure")
                    for key in variables_to_load:
                        if key == 'event_frame':
                            cat_results[variable][splits][key] = np.array([])
                        else:
                            cat_results[variable][splits][key] = np.array([])
                    continue  # move on to next variable

                # Load .mat file
                temp_results = self.load_cat_results(mat_path, variable)[0]
                # try:
                #     temp_results = self.load_cat_results(mat_path, variable)[0]
                # except Exception as e:
                #     print(f"[ERROR] Failed loading {mat_path}: {e} → skipping variable")
                #     continue
                            
                
                
                for key in variables_to_load:
                    if key == 'event_frame':
                        cat_results[variable][splits][key] = temp_results[key] - 1
                    else:
                        cat_results[variable][splits][key] = temp_results[key]

        return cat_results 

    def calculate_mean_across_shuffles(self,cat_results):
        mean_results = {}
        mean_results_all = {}
        
        for variable in cat_results:
            mean_results[variable] = {}
            mean_results_all[variable] = {}
            
            # Store data across splits for final mean
            split_data = {measure: [] for measure in cat_results[variable][0].keys()}
            
            for split in cat_results[variable]:
                mean_results[variable][split] = {}
                
                for measure in cat_results[variable][split]:
                    data = cat_results[variable][split][measure]
                    
                    # Handle sc vs pop data
                    # If data is empty, just propagate empty
                    if data is None or len(data) == 0:
                        mean_data = np.array([])
                        std_data = np.array([])
                    elif 'sc_' in measure:
                        mean_data = np.mean(data, axis=2)  # frames x neurons
                        std_data = np.std(data, axis=2)
                    elif 'pop_' in measure:   
                        mean_data = np.mean(data, axis=1)  # frames
                        std_data = np.std(data, axis=1)
                    else:
                        mean_data = data
                        std_data = None
                    
                    mean_results[variable][split][f'{measure}_mean'] = mean_data
                    mean_results[variable][split][f'{measure}_std'] = std_data
                    
                    # Collect means across splits
                    split_data[measure].append(mean_data)
            
            # Calculate means across splits
            for measure in split_data:
                split_means = np.array(split_data[measure])
                if 'sc_' in measure:
                    mean_results_all[variable][f'{measure}_mean'] = np.mean(split_means, axis=0)  # mean across splits
                    mean_results_all[variable][f'{measure}_std'] = np.std(split_means, axis=0)   # std across splits
                else:
                    mean_results_all[variable][f'{measure}_mean'] = np.mean(split_means, axis=0)
                    mean_results_all[variable][f'{measure}_std'] = np.std(split_means, axis=0)

        return mean_results, mean_results_all
    
    def correct_artifact_in_data(self, cat_results, method='zero', artifact_start=4, artifact_end=13):
        """
        Correct artifact in original data before averaging across shuffles.
        
        Parameters:
        -----------
        cat_results : dict
            Original results dictionary containing raw decoder outputs
        method : str
            'zero' or 'interpolate'
        artifact_start : int
            First frame of artifact
        artifact_end : int
            Last frame of artifact
        
        Returns:
        --------
        dict
            Copy of results with corrected data
        """
        corrected_results = {}
        
        for variable in cat_results:
            corrected_results[variable] = {}
            
            for split in cat_results[variable]:
                corrected_results[variable][split] = {}
                
                for measure in cat_results[variable][split]:
                    if measure == 'event_frame':
                        corrected_results[variable][split][measure] = cat_results[variable][split][measure]
                        continue
                        
                    data = cat_results[variable][split][measure].copy()

                    # Store original cumulative data if present
                    if 'cumulative' in measure:
                        corrected_results[variable][split][f'{measure}_original'] = data.copy()
                    
                    if method == 'zero':
                        artifact_start = 0
                        # Zero out artifact frames
                        if 'sc_' in measure:  # Single cell data
                            data[artifact_start:artifact_end+1, :, :] = 0
                        elif 'pop_' in measure:  # Population data
                            data[artifact_start:artifact_end+1, :] = 0
                            
                    else:  # interpolate
                        if 'sc_' in measure:  # Single cell data
                            for neuron in range(data.shape[1]):
                                for shuffle in range(data.shape[2]):
                                    values = data[:, neuron, shuffle]
                                    x_known = [artifact_start-1, artifact_end+1]
                                    y_known = [values[artifact_start-1], values[artifact_end+1]]
                                    x_interp = np.arange(artifact_start, artifact_end+1)
                                    y_interp = np.interp(x_interp, x_known, y_known)
                                    data[artifact_start:artifact_end+1, neuron, shuffle] = y_interp
                                    
                        elif 'pop_' in measure:  # Population data
                            for shuffle in range(data.shape[1]):
                                values = data[:, shuffle]
                                x_known = [artifact_start-1, artifact_end+1]
                                y_known = [values[artifact_start-1], values[artifact_end+1]]
                                x_interp = np.arange(artifact_start, artifact_end+1)
                                y_interp = np.interp(x_interp, x_known, y_known)
                                data[artifact_start:artifact_end+1, shuffle] = y_interp
                    
                    # Recalculate cumulative metrics if needed
                    if 'cumulative' in measure:
                        if 'information' in measure:
                            data = np.cumsum(data, axis=0)
                        else:  # fraction correct
                            data = np.cumsum(data, axis=0) / np.arange(1, len(data) + 1)[:, None, None if 'sc_' in measure else None]
                    
                    corrected_results[variable][split][measure] = data
        
        return corrected_results

    def process_multiple_datasets(self, datasets, model_type, single_balanced=False):
        """Process multiple datasets and calculate mean decoding."""
        for animalID, date, server in datasets:
            try:
                key = f'{animalID}_{date}'
                print(f'Processing dataset: {key}')
                
                # Process all splits for this dataset
                self.cat_results[key] = {}
                self.celltype_info[key] = {}
                
                decoding_dir = f'{server}/Connie/ProcessedData/{animalID}/{date}/{model_type}/decoding/'
                split_results = self.get_cat_results_across_datasets(decoding_dir, self.decoded_variables,single_balanced=single_balanced)
                celltype_array, neuron_groups, colors = self.load_celltypes(server,animalID,date)
                self.cat_results[key] = split_results

                # Calculate means for this dataset
                self.mean_results[key], self.mean_results_all[key] = self.calculate_mean_across_shuffles(self.cat_results[key])
                self.mean_results_all[key]['celltype_array'] = celltype_array
                self.mean_results_all[key]['neuron_groups'] = neuron_groups
                self.celltype_info[key]['celltype_array'] = celltype_array
                self.celltype_info[key]['neuron_groups'] = neuron_groups

            except Exception as e:
                print(f"Error processing {key}: {e}")
                continue

        return self.mean_results, self.mean_results_all,self.cat_results,self.celltype_info

    def create_shuffled_distribution_structure(self, decoder_type='sound_category', metric = 'sc_instantaneous_information'):
        """Create a structure for frames x neurons x all shuffles (500 total shuffles across 10 folds)."""
        shuffled_structure = {}
        for dataset in self.cat_results:
            print(dataset)
            shuffled_structure[dataset] = []
            try:
                # Process each fold
                for fold_num in range(1, 11):  # Assuming folds are labeled 1 to 10
                    fold_key = f'fold_{fold_num}'
                    # Extract the shuffled data for the specific decoder type and fold
                    shuffled_data = self.cat_results[dataset][f'shuffled/{decoder_type}'][fold_num - 1][metric]
                    # Append the shuffled data (50 shuffles per fold) to the list
                    shuffled_structure[dataset].append(shuffled_data)
                # After all folds are processed, concatenate the data into a single array of shape (frames x neurons x all_shuffles)
                if shuffled_data.ndim == 2:
                    shuffled_structure[dataset] = np.concatenate(shuffled_structure[dataset], axis=1)
                else:
                    shuffled_structure[dataset] = np.concatenate(shuffled_structure[dataset], axis=2)
            except Exception as e:
                print(f"Error processing {dataset} on shuffle {fold_num}: {e}")
                continue

        return shuffled_structure
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
    #def aggregate_cat_from_results(self,all_results, no_abs=1, significant_neurons=None):

    
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

    def save_process_multiple_datasets_output(self,output, filename):
        """
        Save the output of process_multiple_datasets to a file using pickle.
        
        Parameters:
        - output: The output of process_multiple_datasets
        - filename: str, the filename to save the output
        """
        with open(filename, 'wb') as file:
            pickle.dump(output, file)
        print(f"Output saved to {filename}")