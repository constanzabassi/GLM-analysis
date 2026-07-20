
#create functions to generate supplemental figure 2 panels
import os
import numpy as np
import scipy.io
import h5py
from scipy.stats import sem
from scipy import stats

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.stats import sem

class GLMPredictorProcessor:
    def __init__(self, neuron_groups):
        self.neuron_groups = neuron_groups
        self.align_info = None

    def load_and_align_predictors_datasets(self,datasets, model_type,alignment,save_suffix = 'prepost trial cv 73 #'):
        """
        Process multiple datasets and calculate mean deviance explained for each.

        Parameters:
            datasets (list of tuples): List of tuples containing (animalID, date, server).
            model_type (str): The type of the GLM model.

        Returns:
            dict: A dictionary where keys are dataset identifiers and values are results.
        """
        all_predictor_var = {}
        aligned_predictors_all = {}
        aligned_predictors_coupling = {}
        for animalID, date, server in datasets:
            key = f'{animalID}_{date}'
            print(f'Processing dataset: {key}')
            predictor_var = self.load_glm_variables(animalID, date, server, model_type,save_suffix=save_suffix) #load varaibles
            
            aligned_predictors_all[key] = {}
            aligned_predictors_coupling[key] = {}

            for fold_number in range(10):
                print(f'  Processing fold: {fold_number}')
                relative_trial_starts = self.get_trial_frames_from_combined_frames(predictor_var[fold_number]['combined_frames_included'])
                _,alignment_frames_global, alignment_frames, left_padding, right_padding = self.find_align_info_from_behav(
                                            behav_matrix=predictor_var[fold_number]['behav_matrix'],
                                            condition_array_trials=predictor_var[fold_number]['condition_array_trials'],
                                            trial_start_frames= relative_trial_starts,
                                            trial_start_col= 4,  # MATLAB 5th col -> python index 4
                                            alternative_alignment = False,
                                            behav_cols= None,
                                            behav_big_matrix=predictor_var[fold_number]['behav_big_matrix'],
                                            no_reward_big_row= 182,  # e.g. behav_big_matrix[182,:] marks "no reward / pure"
                                        )
                frames = self.alignment_frames( alignment_frames_global, left_padding, right_padding, alignment)
                aligned_behav_this_fold, valid_trials = self.align_behav_predictors(frames, predictor_var[fold_number]['behav_matrix'])
                aligned_coupling_this_fold, valid_trials_c = self.align_behav_predictors(frames, predictor_var[fold_number]['coupling_predictors'])
            
            # if key not in aligned_predictors_all:
            #     aligned_predictors_all[key] = {}
            #     aligned_predictors_coupling[key] = {}

            # FILTER condition_array_trials to match aligned trials
                predictor_var[fold_number]['condition_array_trials'] = (
                predictor_var[fold_number]['condition_array_trials'][valid_trials, :]
)

                aligned_predictors_all[key][fold_number] = aligned_behav_this_fold
                aligned_predictors_coupling[key][fold_number] = aligned_coupling_this_fold

            all_predictor_var[key] = predictor_var

        return all_predictor_var,aligned_predictors_all, aligned_predictors_coupling
    
    def load_and_align_predictors_datasets_running(self,datasets, model_type,alignment,save_suffix = 'prepost trial cv 73 #'):
        """
        Process multiple datasets and calculate mean deviance explained for each.

        Parameters:
            datasets (list of tuples): List of tuples containing (animalID, date, server).
            model_type (str): The type of the GLM model.

        Returns:
            dict: A dictionary where keys are dataset identifiers and values are results.
        """
        all_predictor_var = {}
        aligned_predictors_all = {}
        aligned_predictors_coupling = {}
        alignment_frames_all = {}

        behav_cols = {
            "vel_y": 78,
            "vel_x": 94,
            "view_angle": 110,
            "left_turn": 0,
            "right_turn": 39,
            "reward": 122,
            "no_reward": 130,
            "left_sound_rep1": 140,
            "right_sound_rep1": 146,
            "left_sound_rep2": 152,
            "right_sound_rep2": 158,
            "left_sound_rep3": 164,
            "right_sound_rep3": 170,
            "photostim": 176,
        }
        for animalID, date, server in datasets:
            key = f'{animalID}_{date}'
            print(f'Processing dataset: {key}')
            predictor_var = self.load_glm_variables(animalID, date, server, model_type,do_test = True,behav_matrix_to_load = 'behav_big_matrix_original',model_name = '',load_coupling = False, load_trial_ids = True, save_suffix = save_suffix) #load varaibles

            aligned_predictors_all[key] = {}
            aligned_predictors_coupling[key] = {}
            alignment_frames_all[key] = {}

            for fold_number in range(10):
                # print(f'  Processing fold: {fold_number}')
                relative_trial_starts = self.find_trial_start_alignment_frames(predictor_var[fold_number]['trial_start'])
                align_info,alignment_frames_global, alignment_frames, left_padding, right_padding = self.find_align_info_from_behav(
                                            behav_matrix=predictor_var[fold_number]['behav_big_matrix_raw'],
                                            condition_array_trials=predictor_var[fold_number]['condition_array_trials'],
                                            trial_start_frames= relative_trial_starts,
                                            trial_start_col= 4,  # MATLAB 5th col -> python index 4
                                            alternative_alignment = False,
                                            behav_cols= behav_cols,
                                            behav_big_matrix=predictor_var[fold_number]['behav_big_matrix'],
                                            no_reward_big_row= 182,  # e.g. behav_big_matrix[182,:] marks "no reward / pure"
                                        )
                self.align_info = align_info
                frames = self.alignment_frames( alignment_frames_global, left_padding, right_padding, alignment)
                aligned_behav_this_fold, valid_trials = self.align_behav_predictors(frames, predictor_var[fold_number]['behav_matrix'])
            
            # if key not in aligned_predictors_all:
            #     aligned_predictors_all[key] = {}
            #     aligned_predictors_coupling[key] = {}

            # FILTER condition_array_trials to match aligned trials
                predictor_var[fold_number]['condition_array_trials'] = (
                predictor_var[fold_number]['condition_array_trials'][valid_trials, :]
            )

                aligned_predictors_all[key][fold_number] = aligned_behav_this_fold
                aligned_predictors_coupling[key][fold_number] = []  # Initialize as an empty list
                alignment_frames_all[key][fold_number] = frames

            all_predictor_var[key] = predictor_var

        return all_predictor_var,aligned_predictors_all, aligned_predictors_coupling, alignment_frames_all
    
    def load_glm_variables(self,animalID, date, server, model_type, do_test = False, model_name = 'GLM_3nmf', load_coupling = True, load_trial_ids = False, save_suffix = 'prepost trial cv 73 #', behav_matrix_to_load = 'behav_big_matrix'):
        """
        Load GLM variables from specified directory.
        Parameters
        ----------
        animalID : str
            Animal identifier.
        date : str
            Date of the session.
        server : str
            Server path.
        model_type : str
            Type of the GLM model.
        fold_number : int
            Fold number for cross-validation.
        Returns
        -------
        dict
            Dictionary containing loaded variables.
        """
        predictor_vars = {}
        save_directory = f'{server}/Connie/ProcessedData/{animalID}/{date}/{model_name}{model_type}/'
        for fold_number in range(10):
            path = os.path.join(save_directory, f"{save_suffix}{fold_number+1}") 
            if do_test:
                path_nonpredictors = path
                path = os.path.join(path,'test')
                # print(f'Loading predictors from: {path}')
            else:
                path_nonpredictors = path
            
            # Load behavioral matrices
            behav = scipy.io.loadmat(os.path.join(path, 'behav_the_matrix.mat'))
            behav_matrix = behav['behav_the_matrix']
            behav_ids = scipy.io.loadmat(os.path.join(path, 'behav_the_matrix_ids.mat'))
            behav_matrix_ids_raw = behav_ids['behav_the_matrix_ids'][0]
 
            behav = scipy.io.loadmat(os.path.join(path, f'{behav_matrix_to_load}.mat'))
            behav_big_matrix = behav[behav_matrix_to_load]
            behav_big_matrix_raw = behav[behav_matrix_to_load]
            behav_big_matrix = self.safe_zscore(behav_big_matrix.T)
            behav_big_matrix = behav_big_matrix.T
            if behav_matrix_to_load == 'behav_big_matrix_original':
                ids_string = 'behav_big_matrix_ids_original'
            else:
                ids_string = f'{behav_matrix_to_load}_ids'
            behav_big_ids = scipy.io.loadmat(os.path.join(path, f'{ids_string}.mat'))
            behav_big_matrix_ids = behav_big_ids[ids_string][0]

            # Load condition array trials and frames
            condition_array = scipy.io.loadmat(os.path.join(path_nonpredictors, 'condition_array_trials.mat'))
            condition_array_trials = condition_array['condition_array_trials']
            combined_frames = scipy.io.loadmat(os.path.join(path_nonpredictors, 'combined_frames_included.mat'))
            combined_frames_included = combined_frames['combined_frames_included'].squeeze()
            
        
            # Load coupling matrix
            # coupling_matrix = scipy.io.loadmat(os.path.join(path, 'cells_big_matrix.mat'))
            # coupling_predictors = coupling_matrix['cells_big_matrix'] #really large matrix so it would be nice to keep it small
            if load_coupling:
                with h5py.File(os.path.join(path,'cells_big_matrix.mat'), 'r') as file:
                    # Get the data
                    cel_matrix = file['cells_big_matrix'][()] 
                    cel_matrix = cel_matrix.T
                    cel_matrix = self.safe_zscore(cel_matrix)

                # Now stack into a 2D array: shape = (n_trials, n_predictors)
                coupling_predictors = cel_matrix
                coupling_predictors, _ = self.load_general_coupling_predictors(coupling_predictors)
            else:
                coupling_predictors = None

            test_trials = None
            train_trials = None
            trial_start = None

            if load_trial_ids:
                test_trials = scipy.io.loadmat(
                    os.path.join(path_nonpredictors, 'test_trials_all.mat')
                )['test_trials_all'][0] - 1

                train_trials = scipy.io.loadmat(
                    os.path.join(path_nonpredictors, 'train_trials_all.mat')
                )['train_trials_all'][0] - 1

                trial_start = scipy.io.loadmat(os.path.join(path, 'this_trial_start.mat'))['this_trial_start'].squeeze() - 1  


            predictor_vars[fold_number] = {
                'behav_matrix': behav_matrix,
                'behav_matrix_ids_raw': behav_matrix_ids_raw,
                'behav_big_matrix': behav_big_matrix,
                'behav_big_matrix_ids': behav_big_matrix_ids,
                'behav_big_matrix_raw' : behav_big_matrix_raw,
                'condition_array_trials': condition_array_trials,
                'combined_frames_included': combined_frames_included,
                'coupling_predictors': coupling_predictors,
                'test_trials': test_trials,
                'train_trials': train_trials,
                'trial_start': trial_start
            }

        return predictor_vars
    
    def load_general_coupling_predictors(self, coupling_predictors):
        if coupling_predictors.shape[0] < coupling_predictors.shape[1]:
            coupling_predictors = coupling_predictors.T

        predictors_per_cell = 9
        pyr_indices = slice(0, 3)
        som_indices = slice(3, 6)
        pv_indices  = slice(6, 9)
        celltype_slices = {'pyr': pyr_indices, 'som': som_indices, 'pv': pv_indices}

        first_indices = {}
        for cell_type in ['pyr', 'som', 'pv']:
            if cell_type in self.neuron_groups and len(self.neuron_groups[cell_type]) > 0:
                first_indices[cell_type] = self.neuron_groups[cell_type][0][0]
            else:
                first_indices[cell_type] = None

        general_predictors = []

        # Just get one set of 3 from each type
        for cell_type, first_idx in first_indices.items():
            if first_idx is None:
                continue
            start = first_idx * predictors_per_cell
            end = (first_idx + 1) * predictors_per_cell
            neuron_block = coupling_predictors[:, start:end]  # (frames, 9)
            sl = celltype_slices[cell_type]
            general_predictors.append(neuron_block[:, sl])  # (frames, 3)

        final_predictors = np.hstack(general_predictors)  # (frames, 9)
        return final_predictors.T, first_indices
    


    def find_trial_start_alignment_frames(self,
            this_trial_start
        ):
            """
            Build alignment frame indices around trial start.

            Parameters
            ----------
            this_trial_start : array
                First frame of each trial, e.g. [1, 350, 750, ...]
        
            matlab_indexing : bool
                True if this_trial_start or test_trials_all came from MATLAB and starts at 1.

            Returns
            -------
            this_trial_start : array
                Trial start frames, converted to 0-based indexing if needed.
            """

            this_trial_start = np.asarray(this_trial_start).squeeze()

            # Convert trial start frames from MATLAB 1-based to Python 0-based
            this_trial_start = this_trial_start #- 1

            return this_trial_start

    def align_matrix_to_frames(self,frames, matrix_to_align, fill_value=np.nan):
        """
        Align any frame-based matrix to trial/frame windows.

        Parameters
        ----------
        frames : array, shape (n_trials, n_align_frames)
            Frame indices to extract.
        matrix_to_align : array
            Shape should be either (n_vars, n_frames) or (n_frames, n_vars).
        fill_value : float
            Value for invalid/out-of-bounds frames.

        Returns
        -------
        aligned : array, shape (n_trials, n_vars, n_align_frames)
        valid_trials : array
            Trials that had at least one valid frame.
        """

        matrix_to_align = np.asarray(matrix_to_align)

        # Make sure matrix is vars x frames
        if matrix_to_align.shape[0] > matrix_to_align.shape[1]:
            matrix_to_align = matrix_to_align.T

        n_vars, n_total_frames = matrix_to_align.shape
        n_trials, n_align_frames = frames.shape

        aligned = np.full(
            (n_trials, n_vars, n_align_frames),
            fill_value,
            dtype=float
        )

        valid_trials = []

        for t in range(n_trials):
            valid_frame_mask = (frames[t] >= 0) & (frames[t] < n_total_frames)

            if np.any(valid_frame_mask):
                valid_trials.append(t)

            aligned[t, :, valid_frame_mask] = matrix_to_align[:, frames[t, valid_frame_mask]]

        return aligned, np.asarray(valid_trials)

    
    # def load_general_coupling_predictors(self, coupling_predictors):
    #     """
    #     Load general coupling predictors for each cell type from first neurons of each type.
        
    #     Parameters
    #     ----------
    #     coupling_predictors : np.ndarray
    #         Array of shape (n_frames, n_total_predictors) or (n_total_predictors, n_frames),
    #         where each neuron has 9 predictors: 3 pyr, 3 som, 3 pv.
        
    #     Returns
    #     -------
    #     final_predictors : np.ndarray
    #         Array of shape (n_total_predictors_to_plot, n_frames)
    #     first_indices : dict
    #         Dictionary with first neuron index for each cell type
    #     """

    #     # Ensure shape is (frames, predictors)
    #     if coupling_predictors.shape[0] < coupling_predictors.shape[1]:
    #         coupling_predictors = coupling_predictors.T

    #     predictors_per_cell = 9  # 3 pyr, 3 som, 3 pv
    #     # Slices within a neuron
    #     pyr_indices = slice(0, 3)
    #     som_indices = slice(3, 6)
    #     pv_indices  = slice(6, 9)

    #     celltype_slices = {'pyr': pyr_indices, 'som': som_indices, 'pv': pv_indices}

    #     # Find first neuron index for each cell type
    #     first_indices = {}
    #     for cell_type in ['pyr','som','pv']:
    #         if cell_type in self.neuron_groups and len(self.neuron_groups[cell_type]) > 0:
    #             first_indices[cell_type] = self.neuron_groups[cell_type][0][0]
    #         else:
    #             first_indices[cell_type] = None

    #     # Collect predictors
    #     general_predictors = []

    #     for cell_type, first_idx in first_indices.items():
    #         if first_idx is None:
    #             continue  # skip if no neurons of this type

    #         # Extract this neuron's full predictor block
    #         start = first_idx * predictors_per_cell
    #         end   = (first_idx + 1) * predictors_per_cell
    #         neuron_block = coupling_predictors[:, start:end]  # shape: (frames, 9)

    #         # Add predictors for other cell types only
    #         for other_type, sl in celltype_slices.items():
    #             if other_type != cell_type:
    #                 general_predictors.append(neuron_block[:, sl])  # shape: (frames, 3)

    #     # Stack horizontally: shape (frames, n_factors)
    #     final_predictors = np.hstack(general_predictors)

    #     # Return in original format (predictors x frames)
    #     return final_predictors.T, first_indices
    
    def match_coupling_factors(avg_A, avg_B):

        # Similarity
        similarity = 1 - cdist(avg_A, avg_B, metric='cosine')
        
        # Match
        row_ind, col_ind = linear_sum_assignment(-similarity)

        # Return reordered B to match A
        reordered_B = avg_B[col_ind, :]

        return reordered_B, similarity, row_ind, col_ind


    def get_trial_frames_from_combined_frames(self,combined_frames_included):
        """
       
       
        """
        # Subset frames from combined_frames_included
        # Assume combined_frames_included is a 1D binary array like [0, 0, 1, 1, 1, 0, 1, 1]
        included = combined_frames_included

        # Compute differences between adjacent frames
        diffs = np.diff(included)

        # Trial starts where the diff goes from 0 to 1
        trial_starts = np.where(diffs > 1)[0] + 1  # +1 because diff shifts index by 1
        # Prepend 0 to it (only if needed)
        relative_trial_starts = np.concatenate(([0], trial_starts))

        return relative_trial_starts
    
    def get_trial_frames(self,combined_frames_included, condition_array_trials, frames = None):
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
        if frames is None:
            frame_relative_to_all = combined_frames_included
        else:
            frame_relative_to_all = combined_frames_included[frames]

        # Find which trials (rows) in condition_array_trials[:, 4] are in frame_relative_to_all
        trials_included = np.where(np.isin(condition_array_trials[:, 4], frame_relative_to_all))[0]

        # Find which indices in frame_relative_to_all are present in condition_array_trials[:, 4]
        relative_trial_starts = np.where(np.isin(frame_relative_to_all, condition_array_trials[:, 4]))[0]

        return frame_relative_to_all, trials_included, relative_trial_starts

 
    def find_align_info_from_behav(self,
        behav_matrix: np.ndarray,
        condition_array_trials: np.ndarray | None = None,
        trial_start_frames: np.ndarray | None = None,
        trial_start_col: int = 4,  # MATLAB 5th col -> python index 4
        alternative_alignment: bool = False,
        behav_cols: dict | None = None,
        behav_big_matrix: np.ndarray | None = None,
        no_reward_big_row: int | None = 182,  # e.g. behav_big_matrix[182,:] marks "no reward / pure"
    ):
        """
        Like find_align_info, but computed directly from concatenated behavior matrices.
    
        Events (per trial):
        0 sound1 onset  (left OR right sound rep1)
        1 sound2 onset  (left OR right sound rep2)
        2 sound3 onset  (left OR right sound rep3)
        3 turn onset    (left_turn OR right_turn)
        4 reward onset  (reward; if missing, falls back to "no reward/pure" from behav_big_matrix[no_reward_big_row])
    
        alignment_frames are 0-based indices *within-trial* (NaN if missing).
    
        Returns: align_info, alignment_frames, left_padding, right_padding
        """
        if behav_cols is None:
            behav_cols = {
                "vel_y": 0,
                "vel_x": 1,
                "view_angle": 2,
                "left_turn": 3,
                "right_turn": 4,
                "reward": 5,
                "no_reward": 6,
                "left_sound_rep1": 7,
                "right_sound_rep1": 8,
                "left_sound_rep2": 9,
                "right_sound_rep2": 10,
                "left_sound_rep3": 11,
                "right_sound_rep3": 12,
                "photostim": 13,
            }
    
        if trial_start_frames is None:
            if condition_array_trials is None:
                raise ValueError("Provide either trial_start_frames or condition_array_trials.")
            trial_start_frames = np.asarray(condition_array_trials[:, trial_start_col]).ravel()
        else:
            trial_start_frames = np.asarray(trial_start_frames).ravel()
    
        # Clean/sort/unique trial starts
        # trial_start_frames = trial_start_frames[~np.isnan(trial_start_frames)].astype(int)
        # trial_start_frames = np.unique(trial_start_frames)
        # trial_start_frames = trial_start_frames[(trial_start_frames >= 0) & (trial_start_frames < behav_matrix.shape[1])]
        # trial_start_frames.sort()

        # keep order, remove NaNs only
        trial_start_frames = trial_start_frames[~np.isnan(trial_start_frames)].astype(int)

        # clip invalid starts instead of dropping trials
        trial_start_frames = np.clip(
            trial_start_frames,
            0,
            behav_matrix.shape[1] - 1
        )

    
        # Build (start,end) global frame segments per trial
        trial_segments = []
        for i, s in enumerate(trial_start_frames):
            e = (trial_start_frames[i + 1] - 1) if i < len(trial_start_frames) - 1 else (behav_matrix.shape[1] - 1)
            if e >= s:
                trial_segments.append((int(s), int(e)))
    
        def first_onset(x_1d):
            idx = np.where(x_1d)[0]
            return int(idx[0]) if idx.size else None
    
        # Per-trial onsets (0-based within-trial)
        sound_onsets = []  # list of [s1,s2,s3] per trial
        turn_onsets = []
        reward_or_pure_onsets = []
    
        for (s, e) in trial_segments:
            seg = behav_matrix[:, s : e + 1]
            # print(f'Processing trial segment: start={s}, end={e}, segment shape={seg.shape}')
    
            # sound repeats: combine left+right for each repeat
            s1 = first_onset((seg[behav_cols["left_sound_rep1"], :] > 0) | (seg[behav_cols["right_sound_rep1"], :] > 0))
            s2 = first_onset((seg[behav_cols["left_sound_rep2"], :] > 0) | (seg[behav_cols["right_sound_rep2"], :] > 0))
            s3 = first_onset((seg[behav_cols["left_sound_rep3"], :] > 0) | (seg[behav_cols["right_sound_rep3"], :] > 0))
            sound_onsets.append([s1, s2, s3])
    
            # turn: first onset of left OR right
            t_on = first_onset((seg[behav_cols["left_turn"], :] > 0) | (seg[behav_cols["right_turn"], :] > 0))
            turn_onsets.append(t_on)
    
            # reward: first reward; if missing and no_reward_big_row is available, use that instead
            r_on = first_onset(seg[behav_cols["reward"], :] > 0)
    
            if r_on is None and behav_big_matrix is not None and no_reward_big_row is not None:
                big_seg = behav_big_matrix[no_reward_big_row, s : e + 1]
                # treat any >0 as "on"/onset (works even if it's convolved)
                r_on = first_onset(seg[behav_cols["no_reward"], :] > 0) #first_onset(big_seg > 0)
                # r_on = first_onset(big_seg > 0)
    
            reward_or_pure_onsets.append(r_on)
    
        n_trials = len(trial_segments)
        sound_onsets_arr = np.array(
            [[np.nan if x is None else x for x in row] for row in sound_onsets],
            dtype=float,
        )  # (trials,3)
        turn_onsets_arr = np.array([np.nan if x is None else x for x in turn_onsets], dtype=float)
        reward_onsets_arr = np.array([np.nan if x is None else x for x in reward_or_pure_onsets], dtype=float)

        # print('sound_onsets_arr:', sound_onsets_arr, 'turn_onsets_arr:', turn_onsets_arr, 'reward_onsets_arr:', reward_onsets_arr)
    
        # alignment_frames (events x trials)
        event_names = ["S1", "S2", "S3", "turn", "reward"]
        alignment_frames = np.full((len(event_names), n_trials), np.nan, dtype=float)
        alignment_frames[0, :] = sound_onsets_arr[:, 0]
        alignment_frames[1, :] = sound_onsets_arr[:, 1]
        alignment_frames[2, :] = sound_onsets_arr[:, 2]
        alignment_frames[3, :] = turn_onsets_arr
        alignment_frames[4, :] = reward_onsets_arr
    
        # Padding windows (copied from your find_align_info defaults)
        left_padding = {}
        right_padding = {}
        for ev in range(len(event_names)):
            if ev == 0:
                left_padding[ev] = 6
                right_padding[ev] = 30
            elif ev in (1, 2):
                left_padding[ev] = 1
                right_padding[ev] = 30
            elif ev == 3:
                left_padding[ev] = 90 if alternative_alignment else 30
                right_padding[ev] = 60 if alternative_alignment else 12
            elif ev == 4:
                left_padding[ev] = 1
                right_padding[ev] = 23
    
        align_info = {
            "event_names": event_names,
            "trial_start_frames": trial_start_frames,
            "trial_segments": trial_segments,
            "stimulus_repeats_onsets": sound_onsets_arr,   # (trials,3)
            "turn_onset": turn_onsets_arr,                 # (trials,)
            "reward_or_pure_onset": reward_onsets_arr,     # (trials,)
            "alignment_frames": alignment_frames,
            "left_padding": left_padding,
            "right_padding": right_padding,
            "no_reward_big_row": no_reward_big_row,
        }

        # print('alignment_frames:', alignment_frames)
        # print('left_padding:', left_padding)
        # print('right_padding:', right_padding)

        # Convert alignment_frames from within-trial to global frame indices
        n_events, n_trials = alignment_frames.shape
        alignment_frames_global = np.full_like(alignment_frames, np.nan)

        for trial_idx in range(n_trials):
            trial_start = trial_segments[trial_idx][0]
            for ev in range(n_events):
                frame_within = alignment_frames[ev, trial_idx]
                if not np.isnan(frame_within):
                    alignment_frames_global[ev, trial_idx] = trial_start + frame_within

        return align_info,alignment_frames_global, alignment_frames, left_padding, right_padding
    
    def alignment_frames(self, alignment_frames, left_padding, right_padding, alignment):
        
        if alignment['type'] == 'stimulus':
            frames = self.find_alignment_frames(alignment_frames, list(range(3)), 
                                        left_padding, right_padding)
        
        elif alignment['type'] == 'turn':
            frames = self.find_alignment_frames(alignment_frames, [3], 
                                        left_padding, right_padding)
        
        elif alignment['type'] == 'all':
            frames = self.find_alignment_frames(alignment_frames, list(range(6)), 
                                        left_padding, right_padding)
        
        elif alignment['type'] == 'pre':
            frames = self.find_alignment_frames(alignment_frames, list(range(5)), 
                                        left_padding, right_padding)
        return frames
    
    def find_alignment_frames(self, alignment_frames: np.ndarray, event_id: list, left_padding: np.ndarray, right_padding: np.ndarray):
        """
        Align frames based on events and padding.
        Args:
            alignment_frames: Array of frame indices for each event
            event_id: List of event indices to align
            left_padding: Padding before each event
            right_padding: Padding after each event
        Returns:
            frames: Array of aligned frame indices
        """
        
        num_trials = len(alignment_frames[0])
        # print(num_trials)
        total_frame_length = (
            np.sum([left_padding[event] for event in event_id]) +
            np.sum([right_padding[event] for event in event_id]) +
            len(event_id)
        )

        frames = np.zeros((num_trials, total_frame_length), dtype=int)
        # print(frames.shape) 

        for i in range(num_trials):
            temp_frames = []
            for event in event_id:
                left_pad = -left_padding[event]
                right_pad = right_padding[event]
                event_frames = alignment_frames[event, i] + np.arange(left_pad, right_pad + 1)
                temp_frames.extend(event_frames)
            frames[i, :] = temp_frames

        # Remove zero frames (for passive trials)
        # if np.any(frames == 0):
        #     zero_frame_indices = np.where(frames[0, :] == 0)[0]
        #     frames = np.delete(frames, zero_frame_indices, axis=1)

        # print('frames aligned:',frames)
        # print('frames shape:',frames.shape , 'n trials x n frames')
        return frames
    
    def align_behav_predictors(self,frames, predictors_to_align):
        """
        frames: array of shape ( n_trials, n_align_frames)
        predictors_to_align: array of shape (n_vars, n_total_frames)

        Returns:
            aligned_predictors: (n_trials, n_vars, n_align_frames)
        """
        n_trials, n_align_frames = frames.shape
        n_vars = predictors_to_align.shape[0]

        valid_trials = []
        for trial in range(n_trials):
            if np.max(frames[trial, :]) < predictors_to_align.shape[1]:
                valid_trials.append(trial)

        # Preallocate only for valid trials
        aligned_predictors = np.zeros((len(valid_trials), n_vars, n_align_frames))

        for i, trial in enumerate(valid_trials):
            aligned_predictors[i, :, :] = predictors_to_align[:, frames[trial, :]]

        # aligned_predictors = np.zeros((n_trials, n_vars, n_align_frames))

        # for trial in range(n_trials):
        #     aligned_predictors[trial, :, :] = predictors_to_align[:, frames[trial,:]]

        return aligned_predictors, valid_trials
    
    def align_neural_data(self, frames, neural_data_to_align):
        """
        frames: array of shape ( n_trials, n_align_frames)
        neural_data_to_align: array of shape (n_neurons, n_total_frames)

        Returns:
            aligned_neural_data: (n_trials, n_neurons, n_align_frames)
        """

        
        n_trials, n_align_frames = frames.shape
        neural_data_to_align = neural_data_to_align.T
        n_neurons = neural_data_to_align.shape[0]

        aligned_trials = []
        skipped_trials = []

        for trial in range(n_trials):
            trial_frames = frames[trial, :]

            if np.any(trial_frames < 0) or np.any(trial_frames >= neural_data_to_align.shape[1]):
                skipped_trials.append(trial)
                continue

            aligned_trials.append(neural_data_to_align[:, trial_frames])

        if skipped_trials:
            print(
                f"Skipped {len(skipped_trials)} / {n_trials} trials due to "
                f"out-of-bounds frame indices: {skipped_trials}"
            )

        aligned_neural_data = np.stack(aligned_trials, axis=0)

        return aligned_neural_data

        # aligned_neural_data = np.zeros((n_trials, n_neurons, n_align_frames))

        # for trial in range(n_trials):
        #     aligned_neural_data[trial, :, :] = neural_data_to_align[:,frames[trial,:]]

        # return aligned_neural_data
    
    def align_model_outputs_across_folds(self, datasets, frames, model_outputs_dict, model_type, data_dir = None):
        aligned_data_all = {}
        aligned_velocity_all = {}
            
        for animalID, date, server in datasets:
            key = f'{animalID}_{date}'
            print(f'Aligning neural or model outputs for {key}...')
            
            aligned_data_all[key] = {}
            aligned_velocity_all[key] = {}

            for fold_number in range(10):
                save_directory = f'{server}/Connie/ProcessedData/{animalID}/{date}/GLM_running/'

                if data_dir is None:
                    #load predicted neural responses
                    model_outputs = model_outputs_dict[key][model_type][fold_number]['y_pred'] #[{to_align}]
                else: #load true neural responses
                    path = os.path.join(save_directory, f"{data_dir}{fold_number+1}",'test') 
                    model_outputs1 = scipy.io.loadmat(os.path.join(path, 'combined_response.mat'))
                    model_outputs =  model_outputs1['combined_response'].T
                aligned_data = self.align_neural_data(frames[key][fold_number], model_outputs)
                aligned_data_all[key][fold_number] = aligned_data

                #load predictors (no convolution)
                behav_the_matrix1 = scipy.io.loadmat(os.path.join(path, 'velocity.mat')) #y is row 0, x is row 1
                behav_the_matrix =  behav_the_matrix1['velocity'].T
                aligned_data_predictors = self.align_neural_data(frames[key][fold_number], behav_the_matrix)
                aligned_velocity_all[key][fold_number] = aligned_data_predictors
            
        return aligned_data_all, aligned_velocity_all
    

    def get_trial_conditions_from_array(self, condition_array_trials,
                                    fields_to_separate=['correct']):
        """
        Extracts trial indices for each condition combo from condition_array_trials,
        including inferred sound side (left/right) from correctness and turn direction.

        Parameters:
        -----------
        condition_array_trials : np.ndarray
            Array of shape (n_trials, N) where columns 1–3 (MATLAB) or 0–2 (Python) are:
            correct (col 1), left_turn (col 2), is_stim_trial (col 3)
        fields_to_separate : List[str]
            List of fields to split conditions by. Can include 'sound_left' (derived).

        Returns:
        --------
        all_conditions : List[Tuple[np.ndarray, np.ndarray, str]]
            List of (trial_indices, binary_condition_array, label_string)

        condition_matrix : np.ndarray
            Array of shape (n_trials, len(fields_to_separate)) with binary condition values
        """
        n_trials = condition_array_trials.shape[0]
        # -----------------------------
        # SPECIAL CASE: no separation
        # -----------------------------
        if fields_to_separate is None or len(fields_to_separate) == 0:

            # Return a dummy condition matrix (n_trials × 1)
            condition_matrix =  condition_array_trials[:,1:-1]

            all_trials = np.arange(n_trials)

            all_conditions = [
                (all_trials, np.array([]), 'All trials')
            ]

            return all_conditions, condition_matrix

        field_to_col = {
            'correct': 1,
            'left_turn': 2,
            'is_stim_trial': 3
        }

        # Extract base columns
        raw_conditions = {}
        for field in fields_to_separate:
            if field != 'sound_left':  # handled separately
                raw_conditions[field] = condition_array_trials[:, field_to_col[field]].astype(int)

        # Derive sound side if requested
        if 'sound_left' in fields_to_separate:
            correct = raw_conditions['correct']
            left_turn = raw_conditions['left_turn']
            sound_left = (correct & left_turn) | ((1 - correct) & (1 - left_turn))
            raw_conditions['sound_left'] = sound_left.astype(int)

        # Create matrix of just the selected fields
        condition_matrix = np.column_stack([raw_conditions[field] for field in fields_to_separate])

        # Generate all binary combinations
        num_fields = len(fields_to_separate)
        all_combinations = np.array([
            list(map(int, format(i, f'0{num_fields}b')))
            for i in range(2**num_fields)
        ])

        # Human-readable labels
        label_map = {
            'correct': {1: 'Correct', 0: 'Incorrect'},
            'left_turn': {1: 'Left Turn', 0: 'Right Turn'},
            'is_stim_trial': {1: 'Stim', 0: 'Control'},
            'sound_left': {1: 'Sound Left', 0: 'Sound Right'}
        }

        def get_label(comb):
            return '/'.join([label_map[field][bit] for field, bit in zip(fields_to_separate, comb)])

        all_conditions = []
        for comb in all_combinations:
            matching = np.all(condition_matrix == comb, axis=1)
            matching_trials = np.where(matching)[0]
            if len(matching_trials) > 0:
                label = get_label(comb)
                all_conditions.append((matching_trials, comb, label))

        return all_conditions, condition_matrix
    
    def concatenate_folds(self, aligned_predictors_dict):
        """
        Concatenate aligned predictors across folds within each dataset.
        
        Parameters:
            aligned_predictors_dict: dict
                Dictionary of shape {dataset_key: {fold_number: aligned_predictors}}
                Each aligned_predictors is of shape (n_trials, n_vars, n_frames)

        Returns:
            dict: {dataset_key: concatenated_array}, shape (total_trials, n_vars, n_frames)
        """
        concatenated_predictors = {}

        for dataset_key, folds in aligned_predictors_dict.items():
            all_trials = []
            for fold_number, predictors in folds.items():
                all_trials.append(predictors)  # shape: (n_trials, n_vars, n_frames)

            if len(all_trials) > 0:
                concatenated_predictors[dataset_key] = np.concatenate(all_trials, axis=0)
            else:
                print(f'No predictors found for {dataset_key}')
                concatenated_predictors[dataset_key] = None

        return concatenated_predictors
    
    def average_folds(self, aligned_predictors_dict):
        """
        Average aligned predictors across folds (if trial structure is consistent).
        Returns:
            dict: {dataset_key: averaged_predictors}, shape (n_trials, n_vars, n_frames)
        """
        averaged_predictors = {} 
        result = {}
        for dataset_key, folds in aligned_predictors_dict.items():
            fold_arrays = list(folds.values())
            stacked = np.stack(fold_arrays, axis=0)  # shape: (n_folds, n_trials, n_vars, n_frames)
            mean_predictors = np.mean(stacked, axis=0)
            averaged_predictors[dataset_key] = mean_predictors

            # Compute mean and SEM per label
            mean_list = []
            sem_list = []
            label_list = ['All Trials']

            mean_vals = mean_predictors
            sem_vals = sem(stacked, axis=0, nan_policy='omit')  # (n_vars, n_frames)

            mean_list.append(mean_vals)
            sem_list.append(sem_vals)

            result[dataset_key] = {
                'labels': label_list,
                'mean': mean_list,
                'sem': sem_list
            }

        return averaged_predictors

    def average_folds_by_condition(self, aligned_predictors_dict,
                                condition_array_dict,
                                fields_to_separate):
        """
        Averages aligned predictors across folds, split by specified trial conditions.

        Parameters:
        -----------
        aligned_predictors_dict : dict
            {dataset_key: {fold_number: np.array of shape (n_trials, n_vars, n_frames)}}
        
        condition_array_dict : dict
            {dataset_key: {fold_number: condition_array_trials}} — shape (n_trials, >=4)

        fields_to_separate : list of str
            Fields to split trials by (e.g., ['sound_left', 'is_stim_trial'])

        Returns:
        --------
        Dict:
            {dataset_key: {
                'labels': list of condition labels,
                'mean': list of arrays (n_vars x n_frames),
                'sem': list of arrays (n_vars x n_frames)
            }}
        """
        

        result = {}
        
        for dataset_key in aligned_predictors_dict:
            fold_data = aligned_predictors_dict[dataset_key]
            fold_conditions = condition_array_dict[dataset_key]

            # Map label → list of arrays from folds
            condition_trials_by_label = {}

            for fold_number in fold_data:
                predictors = fold_data[fold_number]  # (n_trials, n_vars, n_frames)
                condition_array = fold_conditions[fold_number]['condition_array_trials']

                all_conditions, _ = self.get_trial_conditions_from_array(
                    condition_array, fields_to_separate=fields_to_separate
                )

                for trial_indices, comb, label in all_conditions:
                    if label not in condition_trials_by_label:
                        condition_trials_by_label[label] = []

                    if len(trial_indices) > 0:
                        trials = predictors[trial_indices, :, :]  # (n_trials, n_vars, n_frames)
                        condition_trials_by_label[label].append(trials)

            # Compute mean and SEM per label
            mean_list = []
            sem_list = []
            label_list = []

            for label, trials_list in condition_trials_by_label.items():
                all_trials = np.concatenate(trials_list, axis=0)  # (total_trials, n_vars, n_frames)
                mean_vals = np.nanmean(all_trials, axis=0)        # (n_vars, n_frames)
                sem_vals = sem(all_trials, axis=0, nan_policy='omit')  # (n_vars, n_frames)

                mean_list.append(mean_vals)
                sem_list.append(sem_vals)
                label_list.append(label)

            result[dataset_key] = {
                'labels': label_list,
                'mean': mean_list,
                'sem': sem_list
            }

        return result
    
    def average_folds_by_condition_intervals(self,
                                aligned_predictors_dict,
                                condition_array_dict,
                                fields_to_separate,
                                event_frames):
        """
        Averages aligned predictors across folds, split by specified trial conditions,
        computing mean activity between consecutive events.
        """

        result = {}

        for dataset_key in aligned_predictors_dict:
            fold_data = aligned_predictors_dict[dataset_key]
            fold_conditions = condition_array_dict[dataset_key]

            condition_trials_by_label = {}

            # Build event intervals using number of frames from first fold
            example_fold = next(iter(fold_data.values()))
            n_frames = example_fold.shape[2]
            event_intervals = self.build_event_intervals(event_frames, n_frames,101)
            n_events = len(event_intervals)

            for fold_number in fold_data:
                predictors = fold_data[fold_number]  # (n_trials, n_vars, n_frames)
                condition_array = fold_conditions[fold_number]['condition_array_trials']

                all_conditions, _ = self.get_trial_conditions_from_array(
                    condition_array, fields_to_separate=fields_to_separate
                )

                for trial_indices, comb, label in all_conditions:
                    if label not in condition_trials_by_label:
                        condition_trials_by_label[label] = []

                    if len(trial_indices) > 0:
                        trials = predictors[trial_indices, :, :]  # (n_trials, n_vars, n_frames)
                        condition_trials_by_label[label].append(trials)

            mean_list = []
            sem_list = []
            label_list = []

            for label, trials_list in condition_trials_by_label.items():
                all_trials = np.concatenate(trials_list, axis=0)
                # (total_trials, n_vars, n_frames)

                # Allocate event-averaged arrays
                mean_vals = np.full((all_trials.shape[1], n_events), np.nan)
                sem_vals  = np.full((all_trials.shape[1], n_events), np.nan)

                for ev, frames in enumerate(event_intervals):
                    if len(frames) == 0:
                        continue

                    # Mean over trials AND frames in event
                    frames = np.asarray(frames, dtype=int)
                    event_data = all_trials[:, :, frames]  # (trials, vars, frames)

                    mean_vals[:, ev] = np.nanmean(event_data, axis=(0, 2))
                    # Average over frames per trial
                    trial_means = np.nanmean(event_data, axis=2)  # shape: (n_trials, n_vars)
                    sem_vals[:, ev]  = sem( trial_means, axis=0, nan_policy='omit')

                mean_list.append(mean_vals)
                sem_list.append(sem_vals)
                label_list.append(label)

            result[dataset_key] = {
                'labels': label_list,
                'mean': mean_list,  # (n_vars × n_events)
                'sem': sem_list
            }

        return result
    
    def build_event_intervals(self,event_frames, n_frames, split_frame=None):
        """
        Build event frame intervals, excluding the event onset frame.
        
        Special case:
        - Event index 2 (3rd event): ends at split_frame
        - Event index 3 (4th event): starts at split_frame + 1
        """
        event_frames = np.asarray(event_frames)
        intervals = []

        n_events = len(event_frames) 

        for ev in range(n_events):
            if ev < n_events - 1:
                start = event_frames[ev] + 1
                end   = event_frames[ev + 1]
            else:
                start = event_frames[ev] + 1
                end   = n_frames

            # MATLAB: ev == 3  → Python: ev == 2
            if ev == 2 and split_frame is not None:
                end = split_frame

            # MATLAB: ev == 4  → Python: ev == 3
            elif ev == 3 and split_frame is not None:
                start = split_frame + 1

            # Safety
            start = max(start, 0)
            end   = min(end, n_frames)

            if start < end:
                intervals.append(np.arange(start, end))
            else:
                intervals.append(np.array([], dtype=int))

        return intervals
    
    def safe_zscore(self, X):
        z_scored = np.zeros_like(X)
        stds = np.std(X, axis=0)
        non_zero = stds != 0
        z_scored[:, non_zero] = stats.zscore(X[:, non_zero], axis=0)
        return z_scored
    
    

    def _match_factors(
        self,
        reference,
        target,
        is_data=False,
        return_indices=False,
        index_override=None
    ):
        """
        Match factors in target to reference using correlation.

        reference: (n_factors, n_frames)
        target:
            if is_data=False → (n_factors, n_frames)
            if is_data=True  → (n_trials, n_factors, n_frames)

        index_override: optional ordering to apply directly
        """

        # If indices already provided → just reorder
        if index_override is not None:
            idx = np.asarray(index_override)

            if not is_data:
                matched = target[idx, :]
            else:
                matched = target[:, idx, :]

            if return_indices:
                return matched, idx
            return matched

        # --- compute matching ---
        if not is_data:
            corr = np.corrcoef(reference, target)[:reference.shape[0],
                                                reference.shape[0]:]
        else:
            target_avg = np.nanmean(target, axis=0)
            corr = np.corrcoef(reference, target_avg)[:reference.shape[0],
                                                    reference.shape[0]:]

        row_ind, col_ind = linear_sum_assignment(-np.abs(corr))

        if not is_data:
            matched = target[col_ind, :]
        else:
            matched = target[:, col_ind, :]

        if return_indices:
            return matched, col_ind

        return matched


    def match_and_aggregate_factors(
        self,
        aligned_predictors_dict,
        condition_array_dict,
        fields_to_separate,
        event_frames=None
    ):

        results = {}
        results_interval = {}

        # ---- factor blocks (within‑celltype matching) ----
        factor_groups = {
            'pyr': slice(0, 3),
            'som': slice(3, 6),
            'pv':  slice(6, 9)
        }

        # =========================================================
        # FIRST PASS — per dataset aggregation
        # =========================================================

        for dataset_key in aligned_predictors_dict:

            fold_data = aligned_predictors_dict[dataset_key]
            fold_conditions = condition_array_dict[dataset_key]

            condition_trials_by_label = {}

            for fold in fold_data:

                predictors = fold_data[fold]
                condition_array = fold_conditions[fold]['condition_array_trials']

                all_conditions, _ = self.get_trial_conditions_from_array(
                    condition_array, fields_to_separate
                )

                for trial_inds, _, label in all_conditions:

                    if len(trial_inds) == 0:
                        continue

                    trials = predictors[trial_inds, :, :]
                    condition_trials_by_label.setdefault(label, []).append(trials)

            labels, means, sems, raw_data = [], [], [], []

            for label, trial_blocks in condition_trials_by_label.items():

                all_trials = np.concatenate(trial_blocks, axis=0)

                mean_val = np.nanmean(all_trials, axis=0)
                sem_val = sem(all_trials, axis=0, nan_policy='omit')

                labels.append(label)
                means.append(mean_val)
                sems.append(sem_val)
                raw_data.append(all_trials)

            results[dataset_key] = {
                'labels': labels,
                'mean': means,
                'sem': sems,
                'data': raw_data
            }

        # =========================================================
        # MATCHING ACROSS DATASETS (WITHIN CELLTYPE)
        # =========================================================

        ref_key = next(iter(results.keys()))
        ref_means = results[ref_key]['mean']

        for dataset_key in results:

            if dataset_key == ref_key:
                results[dataset_key]['match_indices'] = [0,1,2,3,4,5,6,7,8]
                continue

            matched_means = []
            matched_sems = []
            matched_data = []
            match_indices = []

            for ref_mat, tgt_mat, tgt_sem, tgt_data in zip(
                ref_means,
                results[dataset_key]['mean'],
                results[dataset_key]['sem'],
                results[dataset_key]['data']
            ):

                blocks_mean = []
                blocks_sem = []
                blocks_data = []
                idx_all = []

                for _, sl in factor_groups.items():

                    mm, idx = self._match_factors(
                        ref_mat[sl, :],
                        tgt_mat[sl, :],
                        return_indices=True
                    )

                    ms = self._match_factors(
                        ref_mat[sl, :],
                        tgt_sem[sl, :],
                        index_override=idx
                    )

                    md = self._match_factors(
                        ref_mat[sl, :],
                        tgt_data[:, sl, :],
                        is_data=True,
                        index_override=idx
                    )

                    blocks_mean.append(mm)
                    blocks_sem.append(ms)
                    blocks_data.append(md)

                    idx_all.extend(sl.start + idx)

                matched_means.append(np.vstack(blocks_mean))
                matched_sems.append(np.vstack(blocks_sem))
                matched_data.append(np.concatenate(blocks_data, axis=1))
                match_indices.append(idx_all)

            results[dataset_key]['mean'] = matched_means
            results[dataset_key]['sem'] = matched_sems
            results[dataset_key]['data'] = matched_data
            results[dataset_key]['match_indices'] = match_indices

        # =========================================================
        # AGGREGATE ACROSS DATASETS
        # =========================================================

        all_means_stack = []

        for dataset_key in results:
            all_means_stack.append(
                np.stack(results[dataset_key]['mean'], axis=0)
            )

        all_means_stack = np.stack(all_means_stack, axis=0)

        mean_across = np.nanmean(all_means_stack, axis=0)
        sem_across = sem(all_means_stack, axis=0, nan_policy='omit')

        results['all_datasets'] = {
            'labels': results[ref_key]['labels'],
            'mean': list(mean_across),
            'sem': list(sem_across)
        }

        # =========================================================
        # OPTIONAL INTERVAL AVERAGING
        # =========================================================

        if event_frames is not None:

            example = np.asarray(results['all_datasets']['mean'])
            n_frames = example.shape[2]

            intervals = self.build_event_intervals(
                event_frames, n_frames, 101
            )

            n_events = len(intervals)

            for key in results:

                interval_means = []
                interval_sems = []

                for mean_mat, sem_mat in zip(
                    results[key]['mean'],
                    results[key]['sem']
                ):

                    im = np.full((mean_mat.shape[0], n_events), np.nan)
                    isem = np.full_like(im, np.nan)

                    for ev, frames in enumerate(intervals):

                        if len(frames) == 0:
                            continue

                        frames = np.asarray(frames, dtype=int)

                        im[:, ev] = np.nanmean(
                            mean_mat[:, frames], axis=1
                        )

                        isem[:, ev] = np.nanmean(
                            sem_mat[:, frames], axis=1
                        )

                    interval_means.append(im)
                    interval_sems.append(isem)

                results[key]['interval_mean'] = interval_means
                results[key]['interval_sem'] = interval_sems

                results_interval[key] = {
                    'labels': results[key]['labels'],
                    'mean': interval_means,
                    'sem': interval_sems
                }

        return results, results_interval

    def load_and_align_predictors_datasets(self,datasets, model_type,alignment,save_suffix = 'prepost trial cv 73 #'):
        """
        Process multiple datasets and calculate mean deviance explained for each.

        Parameters:
            datasets (list of tuples): List of tuples containing (animalID, date, server).
            model_type (str): The type of the GLM model.

        Returns:
            dict: A dictionary where keys are dataset identifiers and values are results.
        """
        all_predictor_var = {}
        aligned_predictors_all = {}
        aligned_predictors_coupling = {}
        for animalID, date, server in datasets:
            key = f'{animalID}_{date}'
            print(f'Processing dataset: {key}')
            predictor_var = self.load_glm_variables(animalID, date, server, model_type,save_suffix=save_suffix) #load varaibles
            
            aligned_predictors_all[key] = {}
            aligned_predictors_coupling[key] = {}

            for fold_number in range(10):
                print(f'  Processing fold: {fold_number}')
                relative_trial_starts = self.get_trial_frames_from_combined_frames(predictor_var[fold_number]['combined_frames_included'])
                _,alignment_frames_global, alignment_frames, left_padding, right_padding = self.find_align_info_from_behav(
                                            behav_matrix=predictor_var[fold_number]['behav_matrix'],
                                            condition_array_trials=predictor_var[fold_number]['condition_array_trials'],
                                            trial_start_frames= relative_trial_starts,
                                            trial_start_col= 4,  # MATLAB 5th col -> python index 4
                                            alternative_alignment = False,
                                            behav_cols= None,
                                            behav_big_matrix=predictor_var[fold_number]['behav_big_matrix'],
                                            no_reward_big_row= 182,  # e.g. behav_big_matrix[182,:] marks "no reward / pure"
                                        )
                frames = self.alignment_frames( alignment_frames_global, left_padding, right_padding, alignment)
                aligned_behav_this_fold, valid_trials = self.align_behav_predictors(frames, predictor_var[fold_number]['behav_matrix'])
                aligned_coupling_this_fold, valid_trials_c = self.align_behav_predictors(frames, predictor_var[fold_number]['coupling_predictors'])
            
            # if key not in aligned_predictors_all:
            #     aligned_predictors_all[key] = {}
            #     aligned_predictors_coupling[key] = {}

            # FILTER condition_array_trials to match aligned trials
                predictor_var[fold_number]['condition_array_trials'] = (
                predictor_var[fold_number]['condition_array_trials'][valid_trials, :]
)

                aligned_predictors_all[key][fold_number] = aligned_behav_this_fold
                aligned_predictors_coupling[key][fold_number] = aligned_coupling_this_fold

            all_predictor_var[key] = predictor_var

        return all_predictor_var,aligned_predictors_all, aligned_predictors_coupling
    
    def load_and_align_predictors_datasets_running(self,datasets, model_type,alignment,save_suffix = 'prepost trial cv 73 #'):
        """
        Process multiple datasets and calculate mean deviance explained for each.

        Parameters:
            datasets (list of tuples): List of tuples containing (animalID, date, server).
            model_type (str): The type of the GLM model.

        Returns:
            dict: A dictionary where keys are dataset identifiers and values are results.
        """
        all_predictor_var = {}
        aligned_predictors_all = {}
        aligned_predictors_coupling = {}
        alignment_frames_all = {}

        behav_cols = {
            "vel_y": 78,
            "vel_x": 94,
            "view_angle": 110,
            "left_turn": 0,
            "right_turn": 39,
            "reward": 122,
            "no_reward": 130,
            "left_sound_rep1": 140,
            "right_sound_rep1": 146,
            "left_sound_rep2": 152,
            "right_sound_rep2": 158,
            "left_sound_rep3": 164,
            "right_sound_rep3": 170,
            "photostim": 176,
        }
        for animalID, date, server in datasets:
            key = f'{animalID}_{date}'
            print(f'Processing dataset: {key}')
            predictor_var = self.load_glm_variables(animalID, date, server, model_type,do_test = True,behav_matrix_to_load = 'behav_big_matrix_original',model_name = '',load_coupling = False, load_trial_ids = True, save_suffix = save_suffix) #load varaibles

            aligned_predictors_all[key] = {}
            aligned_predictors_coupling[key] = {}
            alignment_frames_all[key] = {}

            for fold_number in range(10):
                # print(f'  Processing fold: {fold_number}')
                relative_trial_starts = self.find_trial_start_alignment_frames(predictor_var[fold_number]['trial_start'])
                align_info,alignment_frames_global, alignment_frames, left_padding, right_padding = self.find_align_info_from_behav(
                                            behav_matrix=predictor_var[fold_number]['behav_big_matrix_raw'],
                                            condition_array_trials=predictor_var[fold_number]['condition_array_trials'],
                                            trial_start_frames= relative_trial_starts,
                                            trial_start_col= 4,  # MATLAB 5th col -> python index 4
                                            alternative_alignment = False,
                                            behav_cols= behav_cols,
                                            behav_big_matrix=predictor_var[fold_number]['behav_big_matrix'],
                                            no_reward_big_row= 182,  # e.g. behav_big_matrix[182,:] marks "no reward / pure"
                                        )
                self.align_info = align_info
                frames = self.alignment_frames( alignment_frames_global, left_padding, right_padding, alignment)
                aligned_behav_this_fold, valid_trials = self.align_behav_predictors(frames, predictor_var[fold_number]['behav_matrix'])
            
            # if key not in aligned_predictors_all:
            #     aligned_predictors_all[key] = {}
            #     aligned_predictors_coupling[key] = {}

            # FILTER condition_array_trials to match aligned trials
                predictor_var[fold_number]['condition_array_trials'] = (
                predictor_var[fold_number]['condition_array_trials'][valid_trials, :]
            )

                aligned_predictors_all[key][fold_number] = aligned_behav_this_fold
                aligned_predictors_coupling[key][fold_number] = []  # Initialize as an empty list
                alignment_frames_all[key][fold_number] = frames

            all_predictor_var[key] = predictor_var

        return all_predictor_var,aligned_predictors_all, aligned_predictors_coupling, alignment_frames_all
    
    def load_glm_variables(self,animalID, date, server, model_type, do_test = False, model_name = 'GLM_3nmf', load_coupling = True, load_trial_ids = False, save_suffix = 'prepost trial cv 73 #', behav_matrix_to_load = 'behav_big_matrix'):
        """
        Load GLM variables from specified directory.
        Parameters
        ----------
        animalID : str
            Animal identifier.
        date : str
            Date of the session.
        server : str
            Server path.
        model_type : str
            Type of the GLM model.
        fold_number : int
            Fold number for cross-validation.
        Returns
        -------
        dict
            Dictionary containing loaded variables.
        """
        predictor_vars = {}
        save_directory = f'{server}/Connie/ProcessedData/{animalID}/{date}/{model_name}{model_type}/'
        for fold_number in range(10):
            path = os.path.join(save_directory, f"{save_suffix}{fold_number+1}") 
            if do_test:
                path_nonpredictors = path
                path = os.path.join(path,'test')
                # print(f'Loading predictors from: {path}')
            else:
                path_nonpredictors = path
            
            # Load behavioral matrices
            behav = scipy.io.loadmat(os.path.join(path, 'behav_the_matrix.mat'))
            behav_matrix = behav['behav_the_matrix']
            behav_ids = scipy.io.loadmat(os.path.join(path, 'behav_the_matrix_ids.mat'))
            behav_matrix_ids_raw = behav_ids['behav_the_matrix_ids'][0]
 
            behav = scipy.io.loadmat(os.path.join(path, f'{behav_matrix_to_load}.mat'))
            behav_big_matrix = behav[behav_matrix_to_load]
            behav_big_matrix_raw = behav[behav_matrix_to_load]
            behav_big_matrix = self.safe_zscore(behav_big_matrix.T)
            behav_big_matrix = behav_big_matrix.T
            if behav_matrix_to_load == 'behav_big_matrix_original':
                ids_string = 'behav_big_matrix_ids_original'
            else:
                ids_string = f'{behav_matrix_to_load}_ids'
            behav_big_ids = scipy.io.loadmat(os.path.join(path, f'{ids_string}.mat'))
            behav_big_matrix_ids = behav_big_ids[ids_string][0]

            # Load condition array trials and frames
            condition_array = scipy.io.loadmat(os.path.join(path_nonpredictors, 'condition_array_trials.mat'))
            condition_array_trials = condition_array['condition_array_trials']
            combined_frames = scipy.io.loadmat(os.path.join(path_nonpredictors, 'combined_frames_included.mat'))
            combined_frames_included = combined_frames['combined_frames_included'].squeeze()
            
        
            # Load coupling matrix
            # coupling_matrix = scipy.io.loadmat(os.path.join(path, 'cells_big_matrix.mat'))
            # coupling_predictors = coupling_matrix['cells_big_matrix'] #really large matrix so it would be nice to keep it small
            if load_coupling:
                with h5py.File(os.path.join(path,'cells_big_matrix.mat'), 'r') as file:
                    # Get the data
                    cel_matrix = file['cells_big_matrix'][()] 
                    cel_matrix = cel_matrix.T
                    cel_matrix = self.safe_zscore(cel_matrix)

                # Now stack into a 2D array: shape = (n_trials, n_predictors)
                coupling_predictors = cel_matrix
                coupling_predictors, _ = self.load_general_coupling_predictors(coupling_predictors)
            else:
                coupling_predictors = None

            test_trials = None
            train_trials = None
            trial_start = None

            if load_trial_ids:
                test_trials = scipy.io.loadmat(
                    os.path.join(path_nonpredictors, 'test_trials_all.mat')
                )['test_trials_all'][0] - 1

                train_trials = scipy.io.loadmat(
                    os.path.join(path_nonpredictors, 'train_trials_all.mat')
                )['train_trials_all'][0] - 1

                trial_start = scipy.io.loadmat(os.path.join(path, 'this_trial_start.mat'))['this_trial_start'].squeeze() - 1  


            predictor_vars[fold_number] = {
                'behav_matrix': behav_matrix,
                'behav_matrix_ids_raw': behav_matrix_ids_raw,
                'behav_big_matrix': behav_big_matrix,
                'behav_big_matrix_ids': behav_big_matrix_ids,
                'behav_big_matrix_raw' : behav_big_matrix_raw,
                'condition_array_trials': condition_array_trials,
                'combined_frames_included': combined_frames_included,
                'coupling_predictors': coupling_predictors,
                'test_trials': test_trials,
                'train_trials': train_trials,
                'trial_start': trial_start
            }

        return predictor_vars
    
    def load_general_coupling_predictors(self, coupling_predictors):
        if coupling_predictors.shape[0] < coupling_predictors.shape[1]:
            coupling_predictors = coupling_predictors.T

        predictors_per_cell = 9
        pyr_indices = slice(0, 3)
        som_indices = slice(3, 6)
        pv_indices  = slice(6, 9)
        celltype_slices = {'pyr': pyr_indices, 'som': som_indices, 'pv': pv_indices}

        first_indices = {}
        for cell_type in ['pyr', 'som', 'pv']:
            if cell_type in self.neuron_groups and len(self.neuron_groups[cell_type]) > 0:
                first_indices[cell_type] = self.neuron_groups[cell_type][0][0]
            else:
                first_indices[cell_type] = None

        general_predictors = []

        # Just get one set of 3 from each type
        for cell_type, first_idx in first_indices.items():
            if first_idx is None:
                continue
            start = first_idx * predictors_per_cell
            end = (first_idx + 1) * predictors_per_cell
            neuron_block = coupling_predictors[:, start:end]  # (frames, 9)
            sl = celltype_slices[cell_type]
            general_predictors.append(neuron_block[:, sl])  # (frames, 3)

        final_predictors = np.hstack(general_predictors)  # (frames, 9)
        return final_predictors.T, first_indices
    


    def find_trial_start_alignment_frames(self,
            this_trial_start
        ):
            """
            Build alignment frame indices around trial start.

            Parameters
            ----------
            this_trial_start : array
                First frame of each trial, e.g. [1, 350, 750, ...]
        
            matlab_indexing : bool
                True if this_trial_start or test_trials_all came from MATLAB and starts at 1.

            Returns
            -------
            this_trial_start : array
                Trial start frames, converted to 0-based indexing if needed.
            """

            this_trial_start = np.asarray(this_trial_start).squeeze()

            # Convert trial start frames from MATLAB 1-based to Python 0-based
            this_trial_start = this_trial_start #- 1

            return this_trial_start

    def align_matrix_to_frames(self,frames, matrix_to_align, fill_value=np.nan):
        """
        Align any frame-based matrix to trial/frame windows.

        Parameters
        ----------
        frames : array, shape (n_trials, n_align_frames)
            Frame indices to extract.
        matrix_to_align : array
            Shape should be either (n_vars, n_frames) or (n_frames, n_vars).
        fill_value : float
            Value for invalid/out-of-bounds frames.

        Returns
        -------
        aligned : array, shape (n_trials, n_vars, n_align_frames)
        valid_trials : array
            Trials that had at least one valid frame.
        """

        matrix_to_align = np.asarray(matrix_to_align)

        # Make sure matrix is vars x frames
        if matrix_to_align.shape[0] > matrix_to_align.shape[1]:
            matrix_to_align = matrix_to_align.T

        n_vars, n_total_frames = matrix_to_align.shape
        n_trials, n_align_frames = frames.shape

        aligned = np.full(
            (n_trials, n_vars, n_align_frames),
            fill_value,
            dtype=float
        )

        valid_trials = []

        for t in range(n_trials):
            valid_frame_mask = (frames[t] >= 0) & (frames[t] < n_total_frames)

            if np.any(valid_frame_mask):
                valid_trials.append(t)

            aligned[t, :, valid_frame_mask] = matrix_to_align[:, frames[t, valid_frame_mask]]

        return aligned, np.asarray(valid_trials)

    
    # def load_general_coupling_predictors(self, coupling_predictors):
    #     """
    #     Load general coupling predictors for each cell type from first neurons of each type.
        
    #     Parameters
    #     ----------
    #     coupling_predictors : np.ndarray
    #         Array of shape (n_frames, n_total_predictors) or (n_total_predictors, n_frames),
    #         where each neuron has 9 predictors: 3 pyr, 3 som, 3 pv.
        
    #     Returns
    #     -------
    #     final_predictors : np.ndarray
    #         Array of shape (n_total_predictors_to_plot, n_frames)
    #     first_indices : dict
    #         Dictionary with first neuron index for each cell type
    #     """

    #     # Ensure shape is (frames, predictors)
    #     if coupling_predictors.shape[0] < coupling_predictors.shape[1]:
    #         coupling_predictors = coupling_predictors.T

    #     predictors_per_cell = 9  # 3 pyr, 3 som, 3 pv
    #     # Slices within a neuron
    #     pyr_indices = slice(0, 3)
    #     som_indices = slice(3, 6)
    #     pv_indices  = slice(6, 9)

    #     celltype_slices = {'pyr': pyr_indices, 'som': som_indices, 'pv': pv_indices}

    #     # Find first neuron index for each cell type
    #     first_indices = {}
    #     for cell_type in ['pyr','som','pv']:
    #         if cell_type in self.neuron_groups and len(self.neuron_groups[cell_type]) > 0:
    #             first_indices[cell_type] = self.neuron_groups[cell_type][0][0]
    #         else:
    #             first_indices[cell_type] = None

    #     # Collect predictors
    #     general_predictors = []

    #     for cell_type, first_idx in first_indices.items():
    #         if first_idx is None:
    #             continue  # skip if no neurons of this type

    #         # Extract this neuron's full predictor block
    #         start = first_idx * predictors_per_cell
    #         end   = (first_idx + 1) * predictors_per_cell
    #         neuron_block = coupling_predictors[:, start:end]  # shape: (frames, 9)

    #         # Add predictors for other cell types only
    #         for other_type, sl in celltype_slices.items():
    #             if other_type != cell_type:
    #                 general_predictors.append(neuron_block[:, sl])  # shape: (frames, 3)

    #     # Stack horizontally: shape (frames, n_factors)
    #     final_predictors = np.hstack(general_predictors)

    #     # Return in original format (predictors x frames)
    #     return final_predictors.T, first_indices
    
    def match_coupling_factors(avg_A, avg_B):

        # Similarity
        similarity = 1 - cdist(avg_A, avg_B, metric='cosine')
        
        # Match
        row_ind, col_ind = linear_sum_assignment(-similarity)

        # Return reordered B to match A
        reordered_B = avg_B[col_ind, :]

        return reordered_B, similarity, row_ind, col_ind


    def get_trial_frames_from_combined_frames(self,combined_frames_included):
        """
       
       
        """
        # Subset frames from combined_frames_included
        # Assume combined_frames_included is a 1D binary array like [0, 0, 1, 1, 1, 0, 1, 1]
        included = combined_frames_included

        # Compute differences between adjacent frames
        diffs = np.diff(included)

        # Trial starts where the diff goes from 0 to 1
        trial_starts = np.where(diffs > 1)[0] + 1  # +1 because diff shifts index by 1
        # Prepend 0 to it (only if needed)
        relative_trial_starts = np.concatenate(([0], trial_starts))

        return relative_trial_starts
    
    def get_trial_frames(self,combined_frames_included, condition_array_trials, frames = None):
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
        if frames is None:
            frame_relative_to_all = combined_frames_included
        else:
            frame_relative_to_all = combined_frames_included[frames]

        # Find which trials (rows) in condition_array_trials[:, 4] are in frame_relative_to_all
        trials_included = np.where(np.isin(condition_array_trials[:, 4], frame_relative_to_all))[0]

        # Find which indices in frame_relative_to_all are present in condition_array_trials[:, 4]
        relative_trial_starts = np.where(np.isin(frame_relative_to_all, condition_array_trials[:, 4]))[0]

        return frame_relative_to_all, trials_included, relative_trial_starts

 
    def find_align_info_from_behav(self,
        behav_matrix: np.ndarray,
        condition_array_trials: np.ndarray | None = None,
        trial_start_frames: np.ndarray | None = None,
        trial_start_col: int = 4,  # MATLAB 5th col -> python index 4
        alternative_alignment: bool = False,
        behav_cols: dict | None = None,
        behav_big_matrix: np.ndarray | None = None,
        no_reward_big_row: int | None = 182,  # e.g. behav_big_matrix[182,:] marks "no reward / pure"
    ):
        """
        Like find_align_info, but computed directly from concatenated behavior matrices.
    
        Events (per trial):
        0 sound1 onset  (left OR right sound rep1)
        1 sound2 onset  (left OR right sound rep2)
        2 sound3 onset  (left OR right sound rep3)
        3 turn onset    (left_turn OR right_turn)
        4 reward onset  (reward; if missing, falls back to "no reward/pure" from behav_big_matrix[no_reward_big_row])
    
        alignment_frames are 0-based indices *within-trial* (NaN if missing).
    
        Returns: align_info, alignment_frames, left_padding, right_padding
        """
        if behav_cols is None:
            behav_cols = {
                "vel_y": 0,
                "vel_x": 1,
                "view_angle": 2,
                "left_turn": 3,
                "right_turn": 4,
                "reward": 5,
                "no_reward": 6,
                "left_sound_rep1": 7,
                "right_sound_rep1": 8,
                "left_sound_rep2": 9,
                "right_sound_rep2": 10,
                "left_sound_rep3": 11,
                "right_sound_rep3": 12,
                "photostim": 13,
            }
    
        if trial_start_frames is None:
            if condition_array_trials is None:
                raise ValueError("Provide either trial_start_frames or condition_array_trials.")
            trial_start_frames = np.asarray(condition_array_trials[:, trial_start_col]).ravel()
        else:
            trial_start_frames = np.asarray(trial_start_frames).ravel()
    
        # Clean/sort/unique trial starts
        # trial_start_frames = trial_start_frames[~np.isnan(trial_start_frames)].astype(int)
        # trial_start_frames = np.unique(trial_start_frames)
        # trial_start_frames = trial_start_frames[(trial_start_frames >= 0) & (trial_start_frames < behav_matrix.shape[1])]
        # trial_start_frames.sort()

        # keep order, remove NaNs only
        trial_start_frames = trial_start_frames[~np.isnan(trial_start_frames)].astype(int)

        # clip invalid starts instead of dropping trials
        trial_start_frames = np.clip(
            trial_start_frames,
            0,
            behav_matrix.shape[1] - 1
        )

    
        # Build (start,end) global frame segments per trial
        trial_segments = []
        for i, s in enumerate(trial_start_frames):
            e = (trial_start_frames[i + 1] - 1) if i < len(trial_start_frames) - 1 else (behav_matrix.shape[1] - 1)
            if e >= s:
                trial_segments.append((int(s), int(e)))
    
        def first_onset(x_1d):
            idx = np.where(x_1d)[0]
            return int(idx[0]) if idx.size else None
    
        # Per-trial onsets (0-based within-trial)
        sound_onsets = []  # list of [s1,s2,s3] per trial
        turn_onsets = []
        reward_or_pure_onsets = []
    
        for (s, e) in trial_segments:
            seg = behav_matrix[:, s : e + 1]
            # print(f'Processing trial segment: start={s}, end={e}, segment shape={seg.shape}')
    
            # sound repeats: combine left+right for each repeat
            s1 = first_onset((seg[behav_cols["left_sound_rep1"], :] > 0) | (seg[behav_cols["right_sound_rep1"], :] > 0))
            s2 = first_onset((seg[behav_cols["left_sound_rep2"], :] > 0) | (seg[behav_cols["right_sound_rep2"], :] > 0))
            s3 = first_onset((seg[behav_cols["left_sound_rep3"], :] > 0) | (seg[behav_cols["right_sound_rep3"], :] > 0))
            sound_onsets.append([s1, s2, s3])
    
            # turn: first onset of left OR right
            t_on = first_onset((seg[behav_cols["left_turn"], :] > 0) | (seg[behav_cols["right_turn"], :] > 0))
            turn_onsets.append(t_on)
    
            # reward: first reward; if missing and no_reward_big_row is available, use that instead
            r_on = first_onset(seg[behav_cols["reward"], :] > 0)
    
            if r_on is None and behav_big_matrix is not None and no_reward_big_row is not None:
                big_seg = behav_big_matrix[no_reward_big_row, s : e + 1]
                # treat any >0 as "on"/onset (works even if it's convolved)
                r_on = first_onset(seg[behav_cols["no_reward"], :] > 0) #first_onset(big_seg > 0)
                # r_on = first_onset(big_seg > 0)
    
            reward_or_pure_onsets.append(r_on)
    
        n_trials = len(trial_segments)
        sound_onsets_arr = np.array(
            [[np.nan if x is None else x for x in row] for row in sound_onsets],
            dtype=float,
        )  # (trials,3)
        turn_onsets_arr = np.array([np.nan if x is None else x for x in turn_onsets], dtype=float)
        reward_onsets_arr = np.array([np.nan if x is None else x for x in reward_or_pure_onsets], dtype=float)

        # print('sound_onsets_arr:', sound_onsets_arr, 'turn_onsets_arr:', turn_onsets_arr, 'reward_onsets_arr:', reward_onsets_arr)
    
        # alignment_frames (events x trials)
        event_names = ["S1", "S2", "S3", "turn", "reward"]
        alignment_frames = np.full((len(event_names), n_trials), np.nan, dtype=float)
        alignment_frames[0, :] = sound_onsets_arr[:, 0]
        alignment_frames[1, :] = sound_onsets_arr[:, 1]
        alignment_frames[2, :] = sound_onsets_arr[:, 2]
        alignment_frames[3, :] = turn_onsets_arr
        alignment_frames[4, :] = reward_onsets_arr
    
        # Padding windows (copied from your find_align_info defaults)
        left_padding = {}
        right_padding = {}
        for ev in range(len(event_names)):
            if ev == 0:
                left_padding[ev] = 6
                right_padding[ev] = 30
            elif ev in (1, 2):
                left_padding[ev] = 1
                right_padding[ev] = 30
            elif ev == 3:
                left_padding[ev] = 90 if alternative_alignment else 30
                right_padding[ev] = 60 if alternative_alignment else 12
            elif ev == 4:
                left_padding[ev] = 1
                right_padding[ev] = 23
    
        align_info = {
            "event_names": event_names,
            "trial_start_frames": trial_start_frames,
            "trial_segments": trial_segments,
            "stimulus_repeats_onsets": sound_onsets_arr,   # (trials,3)
            "turn_onset": turn_onsets_arr,                 # (trials,)
            "reward_or_pure_onset": reward_onsets_arr,     # (trials,)
            "alignment_frames": alignment_frames,
            "left_padding": left_padding,
            "right_padding": right_padding,
            "no_reward_big_row": no_reward_big_row,
        }

        # print('alignment_frames:', alignment_frames)
        # print('left_padding:', left_padding)
        # print('right_padding:', right_padding)

        # Convert alignment_frames from within-trial to global frame indices
        n_events, n_trials = alignment_frames.shape
        alignment_frames_global = np.full_like(alignment_frames, np.nan)

        for trial_idx in range(n_trials):
            trial_start = trial_segments[trial_idx][0]
            for ev in range(n_events):
                frame_within = alignment_frames[ev, trial_idx]
                if not np.isnan(frame_within):
                    alignment_frames_global[ev, trial_idx] = trial_start + frame_within

        return align_info,alignment_frames_global, alignment_frames, left_padding, right_padding
    
    def alignment_frames(self, alignment_frames, left_padding, right_padding, alignment):
        
        if alignment['type'] == 'stimulus':
            frames = self.find_alignment_frames(alignment_frames, list(range(3)), 
                                        left_padding, right_padding)
        
        elif alignment['type'] == 'turn':
            frames = self.find_alignment_frames(alignment_frames, [3], 
                                        left_padding, right_padding)
        
        elif alignment['type'] == 'all':
            frames = self.find_alignment_frames(alignment_frames, list(range(6)), 
                                        left_padding, right_padding)
        
        elif alignment['type'] == 'pre':
            frames = self.find_alignment_frames(alignment_frames, list(range(5)), 
                                        left_padding, right_padding)
        return frames
    
    def find_alignment_frames(self, alignment_frames: np.ndarray, event_id: list, left_padding: np.ndarray, right_padding: np.ndarray):
        """
        Align frames based on events and padding.
        Args:
            alignment_frames: Array of frame indices for each event
            event_id: List of event indices to align
            left_padding: Padding before each event
            right_padding: Padding after each event
        Returns:
            frames: Array of aligned frame indices
        """
        
        num_trials = len(alignment_frames[0])
        # print(num_trials)
        total_frame_length = (
            np.sum([left_padding[event] for event in event_id]) +
            np.sum([right_padding[event] for event in event_id]) +
            len(event_id)
        )

        frames = np.zeros((num_trials, total_frame_length), dtype=int)
        # print(frames.shape) 

        for i in range(num_trials):
            temp_frames = []
            for event in event_id:
                left_pad = -left_padding[event]
                right_pad = right_padding[event]
                event_frames = alignment_frames[event, i] + np.arange(left_pad, right_pad + 1)
                temp_frames.extend(event_frames)
            frames[i, :] = temp_frames

        # Remove zero frames (for passive trials)
        # if np.any(frames == 0):
        #     zero_frame_indices = np.where(frames[0, :] == 0)[0]
        #     frames = np.delete(frames, zero_frame_indices, axis=1)

        # print('frames aligned:',frames)
        # print('frames shape:',frames.shape , 'n trials x n frames')
        return frames
    
    def align_behav_predictors(self,frames, predictors_to_align):
        """
        frames: array of shape ( n_trials, n_align_frames)
        predictors_to_align: array of shape (n_vars, n_total_frames)

        Returns:
            aligned_predictors: (n_trials, n_vars, n_align_frames)
        """
        n_trials, n_align_frames = frames.shape
        n_vars = predictors_to_align.shape[0]

        valid_trials = []
        for trial in range(n_trials):
            if np.max(frames[trial, :]) < predictors_to_align.shape[1]:
                valid_trials.append(trial)

        # Preallocate only for valid trials
        aligned_predictors = np.zeros((len(valid_trials), n_vars, n_align_frames))

        for i, trial in enumerate(valid_trials):
            aligned_predictors[i, :, :] = predictors_to_align[:, frames[trial, :]]

        # aligned_predictors = np.zeros((n_trials, n_vars, n_align_frames))

        # for trial in range(n_trials):
        #     aligned_predictors[trial, :, :] = predictors_to_align[:, frames[trial,:]]

        return aligned_predictors, valid_trials
    
    def align_neural_data(self, frames, neural_data_to_align):
        """
        frames: array of shape ( n_trials, n_align_frames)
        neural_data_to_align: array of shape (n_neurons, n_total_frames)

        Returns:
            aligned_neural_data: (n_trials, n_neurons, n_align_frames)
        """

        
        n_trials, n_align_frames = frames.shape
        neural_data_to_align = neural_data_to_align.T
        n_neurons = neural_data_to_align.shape[0]

        aligned_trials = []
        skipped_trials = []

        for trial in range(n_trials):
            trial_frames = frames[trial, :]

            if np.any(trial_frames < 0) or np.any(trial_frames >= neural_data_to_align.shape[1]):
                skipped_trials.append(trial)
                continue

            aligned_trials.append(neural_data_to_align[:, trial_frames])

        if skipped_trials:
            print(
                f"Skipped {len(skipped_trials)} / {n_trials} trials due to "
                f"out-of-bounds frame indices: {skipped_trials}"
            )

        aligned_neural_data = np.stack(aligned_trials, axis=0)

        return aligned_neural_data

        # aligned_neural_data = np.zeros((n_trials, n_neurons, n_align_frames))

        # for trial in range(n_trials):
        #     aligned_neural_data[trial, :, :] = neural_data_to_align[:,frames[trial,:]]

        # return aligned_neural_data
    

    def get_trial_conditions_from_array(self, condition_array_trials,
                                    fields_to_separate=['correct']):
        """
        Extracts trial indices for each condition combo from condition_array_trials,
        including inferred sound side (left/right) from correctness and turn direction.

        Parameters:
        -----------
        condition_array_trials : np.ndarray
            Array of shape (n_trials, N) where columns 1–3 (MATLAB) or 0–2 (Python) are:
            correct (col 1), left_turn (col 2), is_stim_trial (col 3)
        fields_to_separate : List[str]
            List of fields to split conditions by. Can include 'sound_left' (derived).

        Returns:
        --------
        all_conditions : List[Tuple[np.ndarray, np.ndarray, str]]
            List of (trial_indices, binary_condition_array, label_string)

        condition_matrix : np.ndarray
            Array of shape (n_trials, len(fields_to_separate)) with binary condition values
        """
        n_trials = condition_array_trials.shape[0]
        # -----------------------------
        # SPECIAL CASE: no separation
        # -----------------------------
        if fields_to_separate is None or len(fields_to_separate) == 0:

            # Return a dummy condition matrix (n_trials × 1)
            condition_matrix =  condition_array_trials[:,1:-1]

            all_trials = np.arange(n_trials)

            all_conditions = [
                (all_trials, np.array([]), 'All trials')
            ]

            return all_conditions, condition_matrix

        field_to_col = {
            'correct': 1,
            'left_turn': 2,
            'is_stim_trial': 3
        }

        # Extract base columns
        raw_conditions = {}
        for field in fields_to_separate:
            if field != 'sound_left':  # handled separately
                raw_conditions[field] = condition_array_trials[:, field_to_col[field]].astype(int)

        # Derive sound side if requested
        if 'sound_left' in fields_to_separate:
            correct = raw_conditions['correct']
            left_turn = raw_conditions['left_turn']
            sound_left = (correct & left_turn) | ((1 - correct) & (1 - left_turn))
            raw_conditions['sound_left'] = sound_left.astype(int)

        # Create matrix of just the selected fields
        condition_matrix = np.column_stack([raw_conditions[field] for field in fields_to_separate])

        # Generate all binary combinations
        num_fields = len(fields_to_separate)
        all_combinations = np.array([
            list(map(int, format(i, f'0{num_fields}b')))
            for i in range(2**num_fields)
        ])

        # Human-readable labels
        label_map = {
            'correct': {1: 'Correct', 0: 'Incorrect'},
            'left_turn': {1: 'Left Turn', 0: 'Right Turn'},
            'is_stim_trial': {1: 'Stim', 0: 'Control'},
            'sound_left': {1: 'Sound Left', 0: 'Sound Right'}
        }

        def get_label(comb):
            return '/'.join([label_map[field][bit] for field, bit in zip(fields_to_separate, comb)])

        all_conditions = []
        for comb in all_combinations:
            matching = np.all(condition_matrix == comb, axis=1)
            matching_trials = np.where(matching)[0]
            if len(matching_trials) > 0:
                label = get_label(comb)
                all_conditions.append((matching_trials, comb, label))

        return all_conditions, condition_matrix
    
    def concatenate_folds(self, aligned_predictors_dict):
        """
        Concatenate aligned predictors across folds within each dataset.
        
        Parameters:
            aligned_predictors_dict: dict
                Dictionary of shape {dataset_key: {fold_number: aligned_predictors}}
                Each aligned_predictors is of shape (n_trials, n_vars, n_frames)

        Returns:
            dict: {dataset_key: concatenated_array}, shape (total_trials, n_vars, n_frames)
        """
        concatenated_predictors = {}

        for dataset_key, folds in aligned_predictors_dict.items():
            all_trials = []
            for fold_number, predictors in folds.items():
                all_trials.append(predictors)  # shape: (n_trials, n_vars, n_frames)

            if len(all_trials) > 0:
                concatenated_predictors[dataset_key] = np.concatenate(all_trials, axis=0)
            else:
                print(f'No predictors found for {dataset_key}')
                concatenated_predictors[dataset_key] = None

        return concatenated_predictors
    
    def average_folds(self, aligned_predictors_dict):
        """
        Average aligned predictors across folds (if trial structure is consistent).
        Returns:
            dict: {dataset_key: averaged_predictors}, shape (n_trials, n_vars, n_frames)
        """
        averaged_predictors = {} 
        result = {}
        for dataset_key, folds in aligned_predictors_dict.items():
            fold_arrays = list(folds.values())
            stacked = np.stack(fold_arrays, axis=0)  # shape: (n_folds, n_trials, n_vars, n_frames)
            mean_predictors = np.mean(stacked, axis=0)
            averaged_predictors[dataset_key] = mean_predictors

            # Compute mean and SEM per label
            mean_list = []
            sem_list = []
            label_list = ['All Trials']

            mean_vals = mean_predictors
            sem_vals = sem(stacked, axis=0, nan_policy='omit')  # (n_vars, n_frames)

            mean_list.append(mean_vals)
            sem_list.append(sem_vals)

            result[dataset_key] = {
                'labels': label_list,
                'mean': mean_list,
                'sem': sem_list
            }

        return averaged_predictors

    def average_folds_by_condition(self, aligned_predictors_dict,
                                condition_array_dict,
                                fields_to_separate):
        """
        Averages aligned predictors across folds, split by specified trial conditions.

        Parameters:
        -----------
        aligned_predictors_dict : dict
            {dataset_key: {fold_number: np.array of shape (n_trials, n_vars, n_frames)}}
        
        condition_array_dict : dict
            {dataset_key: {fold_number: condition_array_trials}} — shape (n_trials, >=4)

        fields_to_separate : list of str
            Fields to split trials by (e.g., ['sound_left', 'is_stim_trial'])

        Returns:
        --------
        Dict:
            {dataset_key: {
                'labels': list of condition labels,
                'mean': list of arrays (n_vars x n_frames),
                'sem': list of arrays (n_vars x n_frames)
            }}
        """
        

        result = {}
        
        for dataset_key in aligned_predictors_dict:
            fold_data = aligned_predictors_dict[dataset_key]
            fold_conditions = condition_array_dict[dataset_key]

            # Map label → list of arrays from folds
            condition_trials_by_label = {}

            for fold_number in fold_data:
                predictors = fold_data[fold_number]  # (n_trials, n_vars, n_frames)
                condition_array = fold_conditions[fold_number]['condition_array_trials']

                all_conditions, _ = self.get_trial_conditions_from_array(
                    condition_array, fields_to_separate=fields_to_separate
                )

                for trial_indices, comb, label in all_conditions:
                    if label not in condition_trials_by_label:
                        condition_trials_by_label[label] = []

                    if len(trial_indices) > 0:
                        trials = predictors[trial_indices, :, :]  # (n_trials, n_vars, n_frames)
                        condition_trials_by_label[label].append(trials)

            # Compute mean and SEM per label
            mean_list = []
            sem_list = []
            label_list = []

            for label, trials_list in condition_trials_by_label.items():
                all_trials = np.concatenate(trials_list, axis=0)  # (total_trials, n_vars, n_frames)
                mean_vals = np.nanmean(all_trials, axis=0)        # (n_vars, n_frames)
                sem_vals = sem(all_trials, axis=0, nan_policy='omit')  # (n_vars, n_frames)

                mean_list.append(mean_vals)
                sem_list.append(sem_vals)
                label_list.append(label)

            result[dataset_key] = {
                'labels': label_list,
                'mean': mean_list,
                'sem': sem_list
            }

        return result
    
    def average_folds_by_condition_intervals(self,
                                aligned_predictors_dict,
                                condition_array_dict,
                                fields_to_separate,
                                event_frames):
        """
        Averages aligned predictors across folds, split by specified trial conditions,
        computing mean activity between consecutive events.
        """

        result = {}

        for dataset_key in aligned_predictors_dict:
            fold_data = aligned_predictors_dict[dataset_key]
            fold_conditions = condition_array_dict[dataset_key]

            condition_trials_by_label = {}

            # Build event intervals using number of frames from first fold
            example_fold = next(iter(fold_data.values()))
            n_frames = example_fold.shape[2]
            event_intervals = self.build_event_intervals(event_frames, n_frames,101)
            n_events = len(event_intervals)

            for fold_number in fold_data:
                predictors = fold_data[fold_number]  # (n_trials, n_vars, n_frames)
                condition_array = fold_conditions[fold_number]['condition_array_trials']

                all_conditions, _ = self.get_trial_conditions_from_array(
                    condition_array, fields_to_separate=fields_to_separate
                )

                for trial_indices, comb, label in all_conditions:
                    if label not in condition_trials_by_label:
                        condition_trials_by_label[label] = []

                    if len(trial_indices) > 0:
                        trials = predictors[trial_indices, :, :]  # (n_trials, n_vars, n_frames)
                        condition_trials_by_label[label].append(trials)

            mean_list = []
            sem_list = []
            label_list = []

            for label, trials_list in condition_trials_by_label.items():
                all_trials = np.concatenate(trials_list, axis=0)
                # (total_trials, n_vars, n_frames)

                # Allocate event-averaged arrays
                mean_vals = np.full((all_trials.shape[1], n_events), np.nan)
                sem_vals  = np.full((all_trials.shape[1], n_events), np.nan)

                for ev, frames in enumerate(event_intervals):
                    if len(frames) == 0:
                        continue

                    # Mean over trials AND frames in event
                    frames = np.asarray(frames, dtype=int)
                    event_data = all_trials[:, :, frames]  # (trials, vars, frames)

                    mean_vals[:, ev] = np.nanmean(event_data, axis=(0, 2))
                    # Average over frames per trial
                    trial_means = np.nanmean(event_data, axis=2)  # shape: (n_trials, n_vars)
                    sem_vals[:, ev]  = sem( trial_means, axis=0, nan_policy='omit')

                mean_list.append(mean_vals)
                sem_list.append(sem_vals)
                label_list.append(label)

            result[dataset_key] = {
                'labels': label_list,
                'mean': mean_list,  # (n_vars × n_events)
                'sem': sem_list
            }

        return result
    
    def build_event_intervals(self,event_frames, n_frames, split_frame=None):
        """
        Build event frame intervals, excluding the event onset frame.
        
        Special case:
        - Event index 2 (3rd event): ends at split_frame
        - Event index 3 (4th event): starts at split_frame + 1
        """
        event_frames = np.asarray(event_frames)
        intervals = []

        n_events = len(event_frames) 

        for ev in range(n_events):
            if ev < n_events - 1:
                start = event_frames[ev] + 1
                end   = event_frames[ev + 1]
            else:
                start = event_frames[ev] + 1
                end   = n_frames

            # MATLAB: ev == 3  → Python: ev == 2
            if ev == 2 and split_frame is not None:
                end = split_frame

            # MATLAB: ev == 4  → Python: ev == 3
            elif ev == 3 and split_frame is not None:
                start = split_frame + 1

            # Safety
            start = max(start, 0)
            end   = min(end, n_frames)

            if start < end:
                intervals.append(np.arange(start, end))
            else:
                intervals.append(np.array([], dtype=int))

        return intervals
    
    def safe_zscore(self, X):
        z_scored = np.zeros_like(X)
        stds = np.std(X, axis=0)
        non_zero = stds != 0
        z_scored[:, non_zero] = stats.zscore(X[:, non_zero], axis=0)
        return z_scored
    
    

    def _match_factors(
        self,
        reference,
        target,
        is_data=False,
        return_indices=False,
        index_override=None
    ):
        """
        Match factors in target to reference using correlation.

        reference: (n_factors, n_frames)
        target:
            if is_data=False → (n_factors, n_frames)
            if is_data=True  → (n_trials, n_factors, n_frames)

        index_override: optional ordering to apply directly
        """

        # If indices already provided → just reorder
        if index_override is not None:
            idx = np.asarray(index_override)

            if not is_data:
                matched = target[idx, :]
            else:
                matched = target[:, idx, :]

            if return_indices:
                return matched, idx
            return matched

        # --- compute matching ---
        if not is_data:
            corr = np.corrcoef(reference, target)[:reference.shape[0],
                                                reference.shape[0]:]
        else:
            target_avg = np.nanmean(target, axis=0)
            corr = np.corrcoef(reference, target_avg)[:reference.shape[0],
                                                    reference.shape[0]:]

        row_ind, col_ind = linear_sum_assignment(-np.abs(corr))

        if not is_data:
            matched = target[col_ind, :]
        else:
            matched = target[:, col_ind, :]

        if return_indices:
            return matched, col_ind

        return matched


    def match_and_aggregate_factors(
        self,
        aligned_predictors_dict,
        condition_array_dict,
        fields_to_separate,
        event_frames=None
    ):

        results = {}
        results_interval = {}

        # ---- factor blocks (within‑celltype matching) ----
        factor_groups = {
            'pyr': slice(0, 3),
            'som': slice(3, 6),
            'pv':  slice(6, 9)
        }

        # =========================================================
        # FIRST PASS — per dataset aggregation
        # =========================================================

        for dataset_key in aligned_predictors_dict:

            fold_data = aligned_predictors_dict[dataset_key]
            fold_conditions = condition_array_dict[dataset_key]

            condition_trials_by_label = {}

            for fold in fold_data:

                predictors = fold_data[fold]
                condition_array = fold_conditions[fold]['condition_array_trials']

                all_conditions, _ = self.get_trial_conditions_from_array(
                    condition_array, fields_to_separate
                )

                for trial_inds, _, label in all_conditions:

                    if len(trial_inds) == 0:
                        continue

                    trials = predictors[trial_inds, :, :]
                    condition_trials_by_label.setdefault(label, []).append(trials)

            labels, means, sems, raw_data = [], [], [], []

            for label, trial_blocks in condition_trials_by_label.items():

                all_trials = np.concatenate(trial_blocks, axis=0)

                mean_val = np.nanmean(all_trials, axis=0)
                sem_val = sem(all_trials, axis=0, nan_policy='omit')

                labels.append(label)
                means.append(mean_val)
                sems.append(sem_val)
                raw_data.append(all_trials)

            results[dataset_key] = {
                'labels': labels,
                'mean': means,
                'sem': sems,
                'data': raw_data
            }

        # =========================================================
        # MATCHING ACROSS DATASETS (WITHIN CELLTYPE)
        # =========================================================

        ref_key = next(iter(results.keys()))
        ref_means = results[ref_key]['mean']

        for dataset_key in results:

            if dataset_key == ref_key:
                results[dataset_key]['match_indices'] = [0,1,2,3,4,5,6,7,8]
                continue

            matched_means = []
            matched_sems = []
            matched_data = []
            match_indices = []

            for ref_mat, tgt_mat, tgt_sem, tgt_data in zip(
                ref_means,
                results[dataset_key]['mean'],
                results[dataset_key]['sem'],
                results[dataset_key]['data']
            ):

                blocks_mean = []
                blocks_sem = []
                blocks_data = []
                idx_all = []

                for _, sl in factor_groups.items():

                    mm, idx = self._match_factors(
                        ref_mat[sl, :],
                        tgt_mat[sl, :],
                        return_indices=True
                    )

                    ms = self._match_factors(
                        ref_mat[sl, :],
                        tgt_sem[sl, :],
                        index_override=idx
                    )

                    md = self._match_factors(
                        ref_mat[sl, :],
                        tgt_data[:, sl, :],
                        is_data=True,
                        index_override=idx
                    )

                    blocks_mean.append(mm)
                    blocks_sem.append(ms)
                    blocks_data.append(md)

                    idx_all.extend(sl.start + idx)

                matched_means.append(np.vstack(blocks_mean))
                matched_sems.append(np.vstack(blocks_sem))
                matched_data.append(np.concatenate(blocks_data, axis=1))
                match_indices.append(idx_all)

            results[dataset_key]['mean'] = matched_means
            results[dataset_key]['sem'] = matched_sems
            results[dataset_key]['data'] = matched_data
            results[dataset_key]['match_indices'] = match_indices

        # =========================================================
        # AGGREGATE ACROSS DATASETS
        # =========================================================

        all_means_stack = []

        for dataset_key in results:
            all_means_stack.append(
                np.stack(results[dataset_key]['mean'], axis=0)
            )

        all_means_stack = np.stack(all_means_stack, axis=0)

        mean_across = np.nanmean(all_means_stack, axis=0)
        sem_across = sem(all_means_stack, axis=0, nan_policy='omit')

        results['all_datasets'] = {
            'labels': results[ref_key]['labels'],
            'mean': list(mean_across),
            'sem': list(sem_across)
        }

        # =========================================================
        # OPTIONAL INTERVAL AVERAGING
        # =========================================================

        if event_frames is not None:

            example = np.asarray(results['all_datasets']['mean'])
            n_frames = example.shape[2]

            intervals = self.build_event_intervals(
                event_frames, n_frames, 101
            )

            n_events = len(intervals)

            for key in results:

                interval_means = []
                interval_sems = []

                for mean_mat, sem_mat in zip(
                    results[key]['mean'],
                    results[key]['sem']
                ):

                    im = np.full((mean_mat.shape[0], n_events), np.nan)
                    isem = np.full_like(im, np.nan)

                    for ev, frames in enumerate(intervals):

                        if len(frames) == 0:
                            continue

                        frames = np.asarray(frames, dtype=int)

                        im[:, ev] = np.nanmean(
                            mean_mat[:, frames], axis=1
                        )

                        isem[:, ev] = np.nanmean(
                            sem_mat[:, frames], axis=1
                        )

                    interval_means.append(im)
                    interval_sems.append(isem)

                results[key]['interval_mean'] = interval_means
                results[key]['interval_sem'] = interval_sems

                results_interval[key] = {
                    'labels': results[key]['labels'],
                    'mean': interval_means,
                    'sem': interval_sems
                }

        return results, results_interval



    def get_fold_test_path(self, server, animalID, date, model_type, fold_number):
        """
        Build path to the test folder for a given dataset/fold.

        fold_number is zero-indexed.
        """
        save_directory = f'{server}/Connie/ProcessedData/{animalID}/{date}/{model_type}/'
        return os.path.join(save_directory, f"VR_{fold_number + 1}", "test")


    def load_test_response(self, path):
        """
        Load combined_response.mat.

        Expected output:
            response_matrix: neurons x frames
        """
        response_path = os.path.join(path, "combined_response.mat")

        if not os.path.exists(response_path):
            raise FileNotFoundError(f"Could not find combined_response.mat at: {response_path}")

        response = scipy.io.loadmat(response_path)

        if "combined_response" not in response:
            raise KeyError(
                f"'combined_response' not found in combined_response.mat. "
                f"Available keys: {list(response.keys())}"
            )

        response_matrix = np.asarray(response["combined_response"])

        if response_matrix.ndim != 2:
            raise ValueError(
                f"Expected combined_response to be 2D, got shape {response_matrix.shape}"
            )

        return response_matrix


    def load_running_predictors(self, path, expected_n_predictors=32):
        """
        Load behav_big_matrix.mat.

        Expected output:
            running_predictors: predictors x frames

        If saved as frames x predictors, transpose automatically.
        """
        predictor_path = os.path.join(path, "velocity.mat")

        if not os.path.exists(predictor_path):
            raise FileNotFoundError(f"Could not find velocity.mat at: {predictor_path}")

        pred_file = scipy.io.loadmat(predictor_path)

        if "velocity" not in pred_file:
            raise KeyError(
                f"'velocity' not found in velocity.mat. "
                f"Available keys: {list(pred_file.keys())}"
            )

        running_predictors = np.asarray(pred_file["velocity"])

        if running_predictors.ndim != 2:
            raise ValueError(
                f"Expected behav_big_matrix to be 2D, got shape {running_predictors.shape}"
            )

        # If saved as frames x predictors, transpose it.
        if (
            running_predictors.shape[0] != expected_n_predictors
            and running_predictors.shape[1] == expected_n_predictors
        ):
            running_predictors = running_predictors.T

        if running_predictors.shape[0] != expected_n_predictors:
            raise ValueError(
                f"Expected running_predictors to be "
                f"{expected_n_predictors} x frames, got {running_predictors.shape}"
            )

        return running_predictors


    def load_velocity(self, path):
        """
        Load velocity.mat.

        Expected output:
            velocity: 2 x frames
                velocity[0, :] = x velocity
                velocity[1, :] = y velocity
        """
        velocity_path = os.path.join(path, "velocity.mat")

        if not os.path.exists(velocity_path):
            raise FileNotFoundError(f"Could not find velocity.mat at: {velocity_path}")

        velocity_file = scipy.io.loadmat(velocity_path)

        if "velocity" not in velocity_file:
            raise KeyError(
                f"'velocity' not found in velocity.mat. "
                f"Available keys: {list(velocity_file.keys())}"
            )

        velocity = np.asarray(velocity_file["velocity"]).squeeze()

        if velocity.ndim != 2:
            raise ValueError(f"Expected velocity to be 2D, got shape {velocity.shape}")

        # Make sure velocity is 2 x frames.
        if velocity.shape[0] != 2 and velocity.shape[1] == 2:
            velocity = velocity.T

        if velocity.shape[0] != 2:
            raise ValueError(f"Expected velocity to be 2 x frames, got {velocity.shape}")

        return velocity


    def split_velocity(self, velocity):
        """
        Split raw x/y velocity into positive and negative components.

        Returns dictionary with:
            x_raw, y_raw, pos_x, neg_x, pos_y, neg_y
        """
        velocity = np.asarray(velocity)

        if velocity.ndim != 2 or velocity.shape[0] != 2:
            raise ValueError(f"Expected velocity to be 2 x frames, got {velocity.shape}")

        x_velocity_raw = velocity[0, :]
        y_velocity_raw = velocity[1, :]

        velocity_dict = {
            "x_raw": x_velocity_raw,
            "y_raw": y_velocity_raw,
            "pos_x": np.maximum(x_velocity_raw, 0),
            "neg_x": np.maximum(-x_velocity_raw, 0),
            "pos_y": np.maximum(y_velocity_raw, 0),
            "neg_y": np.maximum(-y_velocity_raw, 0),
        }

        return velocity_dict


    def check_velocity_split_overlap(self, velocity_dict):
        """
        Check that raw positive and negative velocity components are mutually exclusive.
        """
        x_overlap = np.sum((velocity_dict["pos_x"] > 0) & (velocity_dict["neg_x"] > 0))
        y_overlap = np.sum((velocity_dict["pos_y"] > 0) & (velocity_dict["neg_y"] > 0))

        return {
            "x_overlap_frames": int(x_overlap),
            "y_overlap_frames": int(y_overlap),
        }


    def get_running_predictor_groups(self, group_mode="4_groups"):
        """
        Return predictor group indices based on MATLAB construction order.

        MATLAB order:
            0:4    +y retro
            4:8    +y prospective
            8:12   -y retro
            12:16  -y prospective
            16:20  +x retro
            20:24  +x prospective
            24:28  -x retro
            28:32  -x prospective
        """
        predictor_groups_4 = {
            "+y": np.arange(0, 8),
            "-y": np.arange(8, 16),
            "+x": np.arange(16, 24),
            "-x": np.arange(24, 32),
        }

        predictor_groups_8 = {
            "+y retro": np.arange(0, 4),
            "+y pro": np.arange(4, 8),
            "-y retro": np.arange(8, 12),
            "-y pro": np.arange(12, 16),
            "+x retro": np.arange(16, 20),
            "+x pro": np.arange(20, 24),
            "-x retro": np.arange(24, 28),
            "-x pro": np.arange(28, 32),
        }

        if group_mode == "4_groups":
            return predictor_groups_4

        if group_mode == "8_groups":
            return predictor_groups_8

        raise ValueError("group_mode must be '4_groups' or '8_groups'")


    def zscore_for_display(self, x, axis=1, eps=1e-12):
        """
        Display-only z-score.
        Useful for visualizing predictors. Do not use this for model fitting
        unless intentionally matching the model normalization.
        """
        x = np.asarray(x).astype(float).copy()
        mu = np.nanmean(x, axis=axis, keepdims=True)
        sigma = np.nanstd(x, axis=axis, keepdims=True)
        sigma[sigma < eps] = 1.0
        return (x - mu) / sigma


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


    def summarize_running_predictors_for_display(self, running_predictors, group_mode="4_groups"):
        """
        Create display-only grouped summaries of convolved running predictors.

        Input:
            running_predictors: 32 x frames

        Output:
            dict of group_name -> 0-to-1 display trace
        """
        running_predictors = np.asarray(running_predictors)

        if running_predictors.ndim != 2 or running_predictors.shape[0] != 32:
            raise ValueError(
                f"Expected running_predictors to be 32 x frames, got {running_predictors.shape}"
            )

        running_pred_z = self.zscore_for_display(running_predictors, axis=1)
        predictor_groups = self.get_running_predictor_groups(group_mode=group_mode)

        running_group_summaries = {}

        for group_name, group_idx in predictor_groups.items():
            group_trace = np.nanmean(np.abs(running_pred_z[group_idx, :]), axis=0)
            running_group_summaries[group_name] = self.minmax_for_display(group_trace)

        return running_group_summaries


    def make_raw_velocity_display_traces(self, velocity_dict, mode="raw_velocity_split"):
        """
        Create display traces from raw velocity.

        mode:
            "raw_velocity" = abs x, abs y
            "raw_velocity_split" = +x, -x, +y, -y
        """
        if mode == "raw_velocity":
            return {
                "abs x raw": self.minmax_for_display(np.abs(velocity_dict["x_raw"])),
                "abs y raw": self.minmax_for_display(np.abs(velocity_dict["y_raw"])),
            }
        
        if mode == "raw_velocity_no_processing":
            return {
                "abs x raw": np.abs(velocity_dict["x_raw"]),
                "abs y raw": np.abs(velocity_dict["y_raw"]),
            }


        if mode == "raw_velocity_split":
            return {
                "+x raw": self.minmax_for_display(velocity_dict["pos_x"]),
                "-x raw": self.minmax_for_display(velocity_dict["neg_x"]),
                "+y raw": self.minmax_for_display(velocity_dict["pos_y"]),
                "-y raw": self.minmax_for_display(velocity_dict["neg_y"]),
            }

        raise ValueError("mode must be 'raw_velocity', 'raw_velocity_no_processing' or 'raw_velocity_split'")


    def validate_response_prediction_shapes(self, response_matrix, y_pred):
        """
        Check that response and model prediction have matching neuron dimension.

        Expected:
            response_matrix: neurons x frames
            y_pred: frames x neurons
        """
        response_matrix = np.asarray(response_matrix)
        y_pred = np.asarray(y_pred)

        if response_matrix.ndim != 2:
            raise ValueError(f"response_matrix must be 2D, got {response_matrix.shape}")

        if y_pred.ndim != 2:
            raise ValueError(f"y_pred must be 2D, got {y_pred.shape}")

        if response_matrix.shape[0] != y_pred.shape[1]:
            raise ValueError(
                f"Neuron mismatch: response has {response_matrix.shape[0]} neurons, "
                f"y_pred has {y_pred.shape[1]} neurons"
            )

        if response_matrix.shape[1] != y_pred.shape[0]:
            print(
                f"Warning: frame mismatch. response has {response_matrix.shape[1]} frames, "
                f"y_pred has {y_pred.shape[0]} frames"
            )

        return True


    def get_top_and_bottom_neurons(self, frac_dev_expl, n_examples=4):
        """
        Return top positive and most negative neurons by fraction deviance explained.
        """
        frac_dev_expl = np.asarray(frac_dev_expl).squeeze()

        top_positive_neurons = np.argsort(frac_dev_expl)[::-1][:n_examples]
        most_negative_neurons = np.argsort(frac_dev_expl)[:n_examples]

        return top_positive_neurons, most_negative_neurons


    

# """
# Utilities to align *frame-wise* predictors to trial events and plot trial-averaged traces.

# Primary use-case:
# - You have a concatenated behavior matrix (predictors x frames) and want to make
#   average traces aligned to the same events used by `find_align_info`
#   (sound repeats, turn, reward).
# """

# from __future__ import annotations

# import math
# from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# import numpy as np


# # Behavior-matrix column layout (0-indexed) from user description
# BEHAV_COLS_DEFAULT: Dict[str, int] = {
#     "vel_y": 0,
#     "vel_x": 1,
#     "view_angle": 2,
#     "left_turn": 3,
#     "right_turn": 4,
#     "reward": 5,
#     "left_sound_rep1": 6,
#     "right_sound_rep1": 7,
#     "left_sound_rep2": 8,
#     "right_sound_rep2": 9,
#     "left_sound_rep3": 10,
#     "right_sound_rep3": 11,
#     "photostim": 12,
# }

# BEHAV_NAMES_DEFAULT: List[str] = [
#     "vel_y",
#     "vel_x",
#     "view_angle",
#     "left_turn",
#     "right_turn",
#     "reward",
#     "left_sound_rep1",
#     "right_sound_rep1",
#     "left_sound_rep2",
#     "right_sound_rep2",
#     "left_sound_rep3",
#     "right_sound_rep3",
#     "photostim",
# ]


# # Windows copied from `find_align_info` (left_pad, right_pad)
# EVENT_WINDOWS_DEFAULT: Dict[str, Tuple[int, int]] = {
#     "sound1": (6, 30),
#     "sound2": (1, 30),
#     "sound3": (1, 30),
#     "turn": (30, 12),
#     "reward": (1, 23),
# }


# def _first_onset(x_bool_1d: np.ndarray) -> Optional[int]:
#     idx = np.where(x_bool_1d)[0]
#     return int(idx[0]) if idx.size else None


# def trial_segments_from_condition_array(
#     condition_array_trials: np.ndarray,
#     n_total_frames: int,
#     trial_start_col: int = 4,
# ) -> List[Tuple[int, int]]:
#     """
#     Build (start,end) trial segments in global-frame coordinates using the
#     `condition_array_trials[:, trial_start_col]` column (MATLAB 5th column => Python index 4).
#     """
#     starts = np.asarray(condition_array_trials[:, trial_start_col]).ravel()
#     starts = starts[~np.isnan(starts)].astype(int)
#     starts = np.unique(starts)
#     starts = starts[(starts >= 0) & (starts < n_total_frames)]
#     starts.sort()

#     segs: List[Tuple[int, int]] = []
#     for i, s in enumerate(starts):
#         e = (starts[i + 1] - 1) if (i < len(starts) - 1) else (n_total_frames - 1)
#         if e >= s:
#             segs.append((int(s), int(e)))
#     return segs


# def compute_event_onsets_from_behav_matrix(
#     behav_matrix: np.ndarray,
#     trial_segments: Sequence[Tuple[int, int]],
#     behav_cols: Mapping[str, int] = BEHAV_COLS_DEFAULT,
# ) -> Dict[str, np.ndarray]:
#     """
#     Compute per-trial event onsets (0-based indices, relative within-trial).

#     Events:
#     - sound1/sound2/sound3: onset of each repeat, combining left+right sound columns
#     - turn: first onset of left_turn OR right_turn
#     - reward: first onset of reward

#     Returns:
#     - dict[event_name] = float array shape (n_trials,), where missing onsets are NaN.
#     """
#     onsets: Dict[str, List[Optional[int]]] = {k: [] for k in ["S1", "S2", "S3", "turn", "reward"]}

#     for (s, e) in trial_segments:
#         seg = behav_matrix[:, s : e + 1]

#         for rep in (1, 2, 3):
#             l = behav_cols[f"left_sound_rep{rep}"]
#             r = behav_cols[f"right_sound_rep{rep}"]
#             onset = _first_onset((seg[l, :] > 0) | (seg[r, :] > 0))
#             onsets[f"sound{rep}"].append(onset)

#         onset_turn = _first_onset((seg[behav_cols["left_turn"], :] > 0) | (seg[behav_cols["right_turn"], :] > 0))
#         onsets["turn"].append(onset_turn)

#         onset_reward = _first_onset(seg[behav_cols["reward"], :] > 0)
#         onsets["reward"].append(onset_reward)

#     out: Dict[str, np.ndarray] = {}
#     for k, v in onsets.items():
#         out[k] = np.array([np.nan if x is None else x for x in v], dtype=float)
#     return out


# def align_matrix_to_trial_events(
#     X: np.ndarray,
#     trial_segments: Sequence[Tuple[int, int]],
#     event_onsets: np.ndarray,
#     left_pad: int,
#     right_pad: int,
# ) -> np.ndarray:
#     """
#     Align a frame-wise matrix to per-trial event onsets.

#     Parameters:
#     - X: (features x frames_total)
#     - trial_segments: list of (start,end) in global frames
#     - event_onsets: float array (n_trials,), 0-based within-trial; NaN for missing trials
#     - left_pad/right_pad: alignment window sizes

#     Returns:
#     - aligned: (trials x features x window_len) with NaNs where trials are missing/out-of-bounds
#     """
#     X = np.asarray(X)
#     if X.ndim != 2:
#         raise ValueError(f"X must be 2D (features x frames). Got {X.shape}")

#     n_trials = len(trial_segments)
#     win_len = left_pad + right_pad + 1
#     aligned = np.full((n_trials, X.shape[0], win_len), np.nan, dtype=float)

#     for t, (s, e) in enumerate(trial_segments):
#         onset = event_onsets[t]
#         if np.isnan(onset):
#             continue

#         onset_i = int(onset)
#         rel_start = onset_i - left_pad
#         rel_end = onset_i + right_pad
#         trial_len = e - s + 1
#         if rel_start < 0 or rel_end >= trial_len:
#             continue

#         aligned[t, :, :] = X[:, (s + rel_start) : (s + rel_end + 1)]

#     return aligned


# def nansem(x: np.ndarray, axis: int = 0) -> np.ndarray:
#     x = np.asarray(x)
#     n = np.sum(~np.isnan(x), axis=axis)
#     sd = np.nanstd(x, axis=axis)
#     return sd / np.sqrt(np.maximum(n, 1))


# def plot_aligned_means_grid(
#     aligned: np.ndarray,
#     names: Sequence[str],
#     title: str,
#     left_pad: int,
#     ncols: int = 4,
#     ylim: Optional[Tuple[float, float]] = None,
# ):
#     """
#     Plot mean±SEM per feature in a grid.
#     Requires matplotlib (import inside to avoid hard dependency for pure compute usage).
#     """
#     import matplotlib.pyplot as plt

#     mean = np.nanmean(aligned, axis=0)
#     sem = nansem(aligned, axis=0)
#     t = np.arange(mean.shape[1]) - left_pad

#     n_feats = mean.shape[0]
#     nrows = math.ceil(n_feats / ncols)
#     fig, axs = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.2 * nrows), sharex=True)
#     axs = np.array(axs).reshape(-1)

#     for i in range(n_feats):
#         ax = axs[i]
#         ax.plot(t, mean[i], color="black", linewidth=1)
#         ax.fill_between(t, mean[i] - sem[i], mean[i] + sem[i], color="black", alpha=0.2, linewidth=0)
#         ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
#         ax.set_title(str(names[i]), fontsize=9)
#         ax.spines["top"].set_visible(False)
#         ax.spines["right"].set_visible(False)
#         if ylim is not None:
#             ax.set_ylim(ylim)

#     for j in range(n_feats, len(axs)):
#         axs[j].axis("off")

#     fig.suptitle(title)
#     fig.tight_layout()
#     return fig


# def plot_average_predictors_through_trial(
#     behav_matrix: np.ndarray,
#     condition_array_trials: np.ndarray,
#     X_to_plot: Optional[np.ndarray] = None,
#     X_names: Optional[Sequence[str]] = None,
#     event_windows: Mapping[str, Tuple[int, int]] = EVENT_WINDOWS_DEFAULT,
#     behav_cols: Mapping[str, int] = BEHAV_COLS_DEFAULT,
# ):
#     """
#     Convenience wrapper:
#     - builds trial segments from condition_array_trials
#     - extracts event onsets from behav_matrix
#     - aligns X_to_plot to each event in event_windows
#     - plots trial-averaged mean±SEM grids
#     """
#     if X_to_plot is None:
#         X_to_plot = behav_matrix
#     if X_names is None:
#         X_names = BEHAV_NAMES_DEFAULT if X_to_plot is behav_matrix else [f"feat{i}" for i in range(X_to_plot.shape[0])]

#     trial_segments = trial_segments_from_condition_array(condition_array_trials, n_total_frames=behav_matrix.shape[1])
#     event_onsets = compute_event_onsets_from_behav_matrix(behav_matrix, trial_segments, behav_cols=behav_cols)

#     figs = {}
#     for event, (lp, rp) in event_windows.items():
#         aligned = align_matrix_to_trial_events(X_to_plot, trial_segments, event_onsets[event], lp, rp)
#         n_valid = int(np.sum(~np.isnan(event_onsets[event])))
#         figs[event] = plot_aligned_means_grid(
#             aligned,
#             names=X_names,
#             title=f"{event} aligned (n={n_valid} trials)",
#             left_pad=lp,
#         )
#     return figs


# def plot_trial_locked_average_with_event_markers(
#     behav_matrix: np.ndarray,
#     condition_array_trials: np.ndarray,
#     X_to_plot: Optional[np.ndarray] = None,
#     X_names: Optional[Sequence[str]] = None,
#     behav_cols: Mapping[str, int] = BEHAV_COLS_DEFAULT,
#     events: Sequence[str] = ("sound1", "sound2", "sound3", "turn", "reward"),
#     frame_rate_hz: Optional[float] = 30.0,
#     trial_start_col: int = 4,
#     max_trial_len: Optional[int] = None,
#     show_sem: bool = True,
# ):
#     """
#     Make a *single* plot aligned to trial start (frame 0), and overlay markers for
#     multiple event onsets (sound1/2/3, turn, reward) simultaneously.

#     This is useful when you want to see predictors "throughout the trial" with all
#     events shown on one time axis (instead of separate event-aligned windows).

#     Notes:
#     - Trials are segmented using `condition_array_trials[:, trial_start_col]`.
#     - Each trial is truncated to a common length for averaging:
#         - If `max_trial_len` is provided, uses that (capped by each trial's length).
#         - Otherwise uses the minimum trial length across detected trials.
#     - Event markers are placed at the *median* event onset across trials.
#     """
#     import matplotlib.pyplot as plt

#     if X_to_plot is None:
#         X_to_plot = behav_matrix
#     if X_names is None:
#         X_names = BEHAV_NAMES_DEFAULT if X_to_plot is behav_matrix else [f"feat{i}" for i in range(X_to_plot.shape[0])]

#     trial_segments = trial_segments_from_condition_array(
#         condition_array_trials,
#         n_total_frames=behav_matrix.shape[1],
#         trial_start_col=trial_start_col,
#     )
#     if len(trial_segments) == 0:
#         raise ValueError("No trial segments found from condition_array_trials.")

#     # Determine common length for averaging
#     trial_lens = np.array([e - s + 1 for (s, e) in trial_segments], dtype=int)
#     common_len = int(np.min(trial_lens)) if max_trial_len is None else int(min(np.min(trial_lens), max_trial_len))
#     if common_len <= 1:
#         raise ValueError("Common trial length too small to plot.")

#     # Stack trial-locked data: (trials x features x time)
#     n_trials = len(trial_segments)
#     X = np.asarray(X_to_plot)
#     if X.ndim != 2:
#         raise ValueError(f"X_to_plot must be 2D (features x frames). Got {X.shape}")
#     trial_locked = np.full((n_trials, X.shape[0], common_len), np.nan, dtype=float)
#     for t, (s, e) in enumerate(trial_segments):
#         seg_len = e - s + 1
#         use_len = min(seg_len, common_len)
#         trial_locked[t, :, :use_len] = X[:, s : s + use_len]

#     # Compute event onsets from behav_matrix (relative within trial)
#     event_onsets = compute_event_onsets_from_behav_matrix(behav_matrix, trial_segments, behav_cols=behav_cols)

#     # Time axis
#     x = np.arange(common_len)
#     xlabel = "Frame (trial start = 0)"
#     if frame_rate_hz is not None and frame_rate_hz > 0:
#         x = x / float(frame_rate_hz)
#         xlabel = "Time (s, trial start = 0)"

#     mean = np.nanmean(trial_locked, axis=0)  # (features x time)
#     sem = nansem(trial_locked, axis=0) if show_sem else None

#     # Plot as stacked small multiples (one axis per feature) to keep it readable
#     n_feats = mean.shape[0]
#     ncols = 4
#     nrows = math.ceil(n_feats / ncols)
#     fig, axs = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.2 * nrows), sharex=True)
#     axs = np.array(axs).reshape(-1)

#     # Compute median onset time per event (only within common_len)
#     event_medians = {}
#     for ev in events:
#         if ev not in event_onsets:
#             continue
#         vals = event_onsets[ev]
#         vals = vals[~np.isnan(vals)]
#         vals = vals[vals < common_len]
#         if vals.size:
#             med = float(np.median(vals))
#             med_x = med / float(frame_rate_hz) if frame_rate_hz is not None and frame_rate_hz > 0 else med
#             event_medians[ev] = med_x

#     for i in range(n_feats):
#         ax = axs[i]
#         ax.plot(x, mean[i], color="black", linewidth=1)
#         if sem is not None:
#             ax.fill_between(x, mean[i] - sem[i], mean[i] + sem[i], color="black", alpha=0.2, linewidth=0)

#         # event markers
#         for ev, ev_x in event_medians.items():
#             ax.axvline(ev_x, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

#         ax.set_title(str(X_names[i]), fontsize=9)
#         ax.spines["top"].set_visible(False)
#         ax.spines["right"].set_visible(False)

#     for j in range(n_feats, len(axs)):
#         axs[j].axis("off")

#     fig.suptitle("Trial-locked average with event markers (S1/S2/S3/Turn/Reward)")
#     for ax in axs[: min(n_feats, len(axs))]:
#         ax.set_xlabel(xlabel)
#     fig.tight_layout()
#     return fig, event_medians
