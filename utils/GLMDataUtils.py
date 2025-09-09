import numpy as np
import os
import h5py
import scipy

class GLMDataUtils:
    def __init__(self):
        pass
    def get_testing_trial_frames(self,combined_frames_included, frames, condition_array_trials):
        """
        Extracting trial/frame relationships from imaging test trials.
        Parameters
        ----------
        combined_frames_included : np.ndarray
            Array containing all possible frame indices.
        frames : array-like
            Indices to select from combined_frames_included.
        condition_array_trials : np.ndarray
            2D array where the 5th column (index 4 in Python) holds frame IDs.
        Returns
        -------
        frame_relative_to_all : np.ndarray
            Subset of combined_frames_included corresponding to the selected frames.
        trials_included : np.ndarray
            Indices (0-based) of trials in condition_array_trials that match frame_relative_to_all.
        relative_trial_starts : np.ndarray
            Indices (0-based) of frames in frame_relative_to_all that match condition_array_trials[:, 4].
        """
        # Subset frames from combined_frames_included
        frame_relative_to_all = combined_frames_included[frames]

        # Find which trials (rows) in condition_array_trials[:, 4] are in frame_relative_to_all
        trials_included = np.where(np.isin(condition_array_trials[:, 4], frame_relative_to_all))[0]

        # Find which indices in frame_relative_to_all are present in condition_array_trials[:, 4]
        relative_trial_starts = np.where(np.isin(frame_relative_to_all, condition_array_trials[:, 4]))[0]

        return frame_relative_to_all, trials_included, relative_trial_starts
    
    def load_decoder_and_glm_variables(self,mouse_dates_keys, session_idx, model_type, fold_number, frames, decoder_type = 'choice', shuffle_num = 0, server = 'V'):
        """
        Load decoder (decoded variables, trials used, information) and GLM variables (predictors) for a specific session and fold.
        """

        example_dataset = [mouse_dates_keys[session_idx]]
        animalID, date = example_dataset[0].split('_')
        save_directory = f'{server}:/Connie/ProcessedData/{animalID}/{date}/GLM_3nmf_{model_type}/'
        path = os.path.join(save_directory, f"prepost trial cv 73 #{fold_number+1}/test") #get testing folder

        # Load behavioral matrices
        behav = scipy.io.loadmat(os.path.join(path, 'behav_the_matrix.mat'))
        behav_matrix = behav['behav_the_matrix']
        behav_ids = scipy.io.loadmat(os.path.join(path, 'behav_the_matrix_ids.mat'))
        behav_matrix_ids_raw = behav_ids['behav_the_matrix_ids'][0]

        behav = scipy.io.loadmat(os.path.join(path, 'behav_big_matrix.mat'))
        behav_big_matrix = behav['behav_big_matrix']
        behav_big_ids = scipy.io.loadmat(os.path.join(path, 'behav_big_matrix_ids.mat'))
        behav_big_matrix_ids = behav_big_ids['behav_big_matrix_ids'][0]

        # Load condition array trials and frames
        condition_array = scipy.io.loadmat(os.path.join(path, 'condition_array_trials.mat'))
        condition_array_trials = condition_array['condition_array_trials']
        combined_frames = scipy.io.loadmat(os.path.join(path, 'combined_frames_included.mat'))
        combined_frames_included = combined_frames['combined_frames_included'].squeeze()

        # Load decoder
        decoder_path = os.path.join(save_directory, f"decoding/{fold_number+1}_1/decoder_results_regular_{decoder_type}.mat")
        with h5py.File(decoder_path, 'r') as f:
            decoder = f['decoder_results']
            decoder_group = decoder['inputs'][decoder_type]
            decoder_info = decoder['aligned'][decoder_type]

            i = shuffle_num  # shuffle_num - 1
            #get shuffle i
            pop_ref = decoder_group['pop_instantaneous_decoded_variables'][i, 0]
            sc_ref = decoder_group['sc_instantaneous_decoded_variables'][i, 0]
            pop_ref_cumulative = decoder_group['pop_cumulative_decoded_variables'][i, 0]
            sc_ref_cumulative = decoder_group['sc_cumulative_decoded_variables'][i, 0]
            decoder_info_ref = decoder_info['results'][i, 0]

            decoder_info_group = f[decoder_info_ref]
            info_data = decoder_info_group['sc_instantaneous_information'][()]
            info_data_cumulative = decoder_info_group['sc_cumulative_information'][()]
            testing_trials_used = decoder_info_group['alignment']['trials_used']['test'][()] #decoder uses data only form testing folder
            testing_trials_used = np.where(testing_trials_used)[0]
                # Access the actual data
            decoder_population = f[pop_ref][()]
            decoder_singlecell = f[sc_ref][()]
            decoder_population_cumulative = f[pop_ref_cumulative][()]
            decoder_singlecell_cumulative = f[sc_ref_cumulative][()]

        return {
            'frames': frames,
            'behav_matrix': behav_matrix,
            'behav_matrix_ids_raw': behav_matrix_ids_raw,
            'behav_big_matrix': behav_big_matrix,
            'behav_big_matrix_ids': behav_big_matrix_ids,
            'condition_array_trials': condition_array_trials,
            'combined_frames_included': combined_frames_included,
            'info_data': info_data,
            'info_data_cumulative': info_data_cumulative,
            'testing_trials_used': testing_trials_used,
            'decoder_population': decoder_population,
            'decoder_singlecell': decoder_singlecell,
            'decoder_population_cumulative': decoder_population_cumulative,
            'decoder_singlecell_cumulative': decoder_singlecell_cumulative,
            'example_dataset': example_dataset,
            'save_directory': save_directory,
            'fold_number': fold_number
        }

    def get_sorted_neuron_indices(self,info_data):
        #sort neurons by max information value
        max_per_neuron = np.max(info_data, axis=0)
        sorted_idx = np.argsort(max_per_neuron)[::-1]
        return sorted_idx

    def get_highlight_trial_indices(self,trials_included, testing_trials_used):
        """
        Returns the indices and values of trials_included that are also in testing_trials_used.
        """
        mask = np.isin(trials_included, testing_trials_used)
        highlight_indices = np.where(mask)[0]
        highlight_values = trials_included[highlight_indices]
        return highlight_indices, highlight_values