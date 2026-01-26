
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
    def load_and_align_predictors_datasets(self,datasets, model_type,alignment):
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
            predictor_var = self.load_glm_variables(animalID, date, server, model_type) #load varaibles
            
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
    
    def load_glm_variables(self,animalID, date, server, model_type):
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
        save_directory = f'{server}/Connie/ProcessedData/{animalID}/{date}/GLM_3nmf_{model_type}/'
        for fold_number in range(10):
            path = os.path.join(save_directory, f"prepost trial cv 73 #{fold_number+1}") 

            # Load behavioral matrices
            behav = scipy.io.loadmat(os.path.join(path, 'behav_the_matrix.mat'))
            behav_matrix = behav['behav_the_matrix']
            behav_ids = scipy.io.loadmat(os.path.join(path, 'behav_the_matrix_ids.mat'))
            behav_matrix_ids_raw = behav_ids['behav_the_matrix_ids'][0]

            behav = scipy.io.loadmat(os.path.join(path, 'behav_big_matrix.mat'))
            behav_big_matrix = behav['behav_big_matrix']
            behav_big_matrix = self.safe_zscore(behav_big_matrix.T)
            behav_big_matrix = behav_big_matrix.T
            behav_big_ids = scipy.io.loadmat(os.path.join(path, 'behav_big_matrix_ids.mat'))
            behav_big_matrix_ids = behav_big_ids['behav_big_matrix_ids'][0]

            # Load condition array trials and frames
            condition_array = scipy.io.loadmat(os.path.join(path, 'condition_array_trials.mat'))
            condition_array_trials = condition_array['condition_array_trials']
            combined_frames = scipy.io.loadmat(os.path.join(path, 'combined_frames_included.mat'))
            combined_frames_included = combined_frames['combined_frames_included'].squeeze()

        
            # Load coupling matrix
            # coupling_matrix = scipy.io.loadmat(os.path.join(path, 'cells_big_matrix.mat'))
            # coupling_predictors = coupling_matrix['cells_big_matrix'] #really large matrix so it would be nice to keep it small
            with h5py.File(os.path.join(path,'cells_big_matrix.mat'), 'r') as file:
                # Get the data
                cel_matrix = file['cells_big_matrix'][()] 
                cel_matrix = cel_matrix.T
                cel_matrix = self.safe_zscore(cel_matrix)

            # Now stack into a 2D array: shape = (n_trials, n_predictors)
            coupling_predictors = cel_matrix
            coupling_predictors, _ = self.load_general_coupling_predictors(coupling_predictors)
            predictor_vars[fold_number] = {
                'behav_matrix': behav_matrix,
                'behav_matrix_ids_raw': behav_matrix_ids_raw,
                'behav_big_matrix': behav_big_matrix,
                'behav_big_matrix_ids': behav_big_matrix_ids,
                'condition_array_trials': condition_array_trials,
                'combined_frames_included': combined_frames_included,
                'coupling_predictors': coupling_predictors
            }

        return predictor_vars
    
    def load_general_coupling_predictors(self, coupling_predictors):
        """
        Load general coupling predictors for each cell type from first neurons of each type.
        
        Parameters
        ----------
        coupling_predictors : np.ndarray
            Array of shape (n_frames, n_total_predictors) or (n_total_predictors, n_frames),
            where each neuron has 9 predictors: 3 pyr, 3 som, 3 pv.
        
        Returns
        -------
        final_predictors : np.ndarray
            Array of shape (n_total_predictors_to_plot, n_frames)
        first_indices : dict
            Dictionary with first neuron index for each cell type
        """

        # Ensure shape is (frames, predictors)
        if coupling_predictors.shape[0] < coupling_predictors.shape[1]:
            coupling_predictors = coupling_predictors.T

        predictors_per_cell = 9  # 3 pyr, 3 som, 3 pv
        # Slices within a neuron
        pyr_indices = slice(0, 3)
        som_indices = slice(3, 6)
        pv_indices  = slice(6, 9)

        celltype_slices = {'pyr': pyr_indices, 'som': som_indices, 'pv': pv_indices}

        # Find first neuron index for each cell type
        first_indices = {}
        for cell_type in ['pyr','som','pv']:
            if cell_type in self.neuron_groups and len(self.neuron_groups[cell_type]) > 0:
                first_indices[cell_type] = self.neuron_groups[cell_type][0][0]
            else:
                first_indices[cell_type] = None

        # Collect predictors
        general_predictors = []

        for cell_type, first_idx in first_indices.items():
            if first_idx is None:
                continue  # skip if no neurons of this type

            # Extract this neuron's full predictor block
            start = first_idx * predictors_per_cell
            end   = (first_idx + 1) * predictors_per_cell
            neuron_block = coupling_predictors[:, start:end]  # shape: (frames, 9)

            # Add predictors for other cell types only
            for other_type, sl in celltype_slices.items():
                if other_type != cell_type:
                    general_predictors.append(neuron_block[:, sl])  # shape: (frames, 3)

        # Stack horizontally: shape (frames, n_factors)
        final_predictors = np.hstack(general_predictors)

        # Return in original format (predictors x frames)
        return final_predictors.T, first_indices
    
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
        print('frames shape:',frames.shape , 'n trials x n frames')
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
    
    

    def _match_factors(self,reference, target, is_data=False):
        """
        Matches factors in the target array to the reference using absolute correlation.
        reference, target: (n_factors, n_frames)
        returns reordered target matched to reference
        """
        if not is_data:
            # Both inputs are (n_factors, n_frames)
            corr = np.corrcoef(reference, target)[:reference.shape[0], reference.shape[0]:]
            row_ind, col_ind = linear_sum_assignment(-np.abs(corr))
            return target[col_ind, :]

        else:
            # target shape: (n_trials, n_factors, n_frames)
            n_trials, n_factors, n_frames = target.shape

            # Average across trials to get matching template
            target_avg = np.nanmean(target, axis=0)  # (n_factors, n_frames)

            corr = np.corrcoef(reference, target_avg)[:reference.shape[0], reference.shape[0]:]
            row_ind, col_ind = linear_sum_assignment(-np.abs(corr))

            # Reorder factors on axis=1
            matched = target[:, col_ind, :]
            return matched

    def match_and_aggregate_factors(self,
                                    aligned_predictors_dict,
                                    condition_array_dict,
                                    fields_to_separate,
                                    event_frames=None):
        """
        Match and aggregate coupling factors across datasets, folds, and conditions.

        Returns:
        --------
        results : dict
            results[dataset_key]['labels']
            results[dataset_key]['mean']   # list of (n_factors × n_frames)
            results[dataset_key]['sem']
            results[dataset_key]['data']   # list of (n_trials, n_factors, n_frames)
            
            results['all_datasets'] same structure
            If event_frames provided:
                results[key]['interval_mean']
                results[key]['interval_sem']
        """

        results = {}
        results_interval = {}
        pooled_by_label = {}

        # ---------- First pass: per-dataset, per-condition, fold-averaged ----------
        for dataset_key in aligned_predictors_dict:
            fold_data = aligned_predictors_dict[dataset_key]
            fold_conditions = condition_array_dict[dataset_key]

            condition_trials_by_label = {}

            for fold in fold_data:
                predictors = fold_data[fold]  # (trials, factors, frames)
                condition_array = fold_conditions[fold]['condition_array_trials']


                all_conditions, _ = self.get_trial_conditions_from_array(
                    condition_array, fields_to_separate
                )

                for trial_inds, _, label in all_conditions:
                    if len(trial_inds) == 0:
                        continue

                    trials = predictors[trial_inds, :, :]  # (trials, factors, frames)

                    condition_trials_by_label.setdefault(label, []).append(trials)
                    pooled_by_label.setdefault(label, []).append(trials)

            labels, means, sems, raw_data = [], [], [], []

            for label, trial_blocks in condition_trials_by_label.items():
                all_trials = np.concatenate(trial_blocks, axis=0)
                mean_val = np.nanmean(all_trials, axis=0)  # (factors × frames)
                sem_val  = sem(all_trials, axis=0, nan_policy='omit')

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

        # ---------- Factor matching across datasets ----------
        ref_key = next(iter(results.keys()))
        ref_means = results[ref_key]['mean']

        for dataset_key in results:
            if dataset_key == ref_key:
                continue

            matched_means, matched_sems, matched_data = [], [], []
            for ref_mat, tgt_mat, tgt_sem, tgt_data in zip(
                ref_means,
                results[dataset_key]['mean'],
                results[dataset_key]['sem'],
                results[dataset_key]['data']
            ):
                matched_means.append(self._match_factors(ref_mat, tgt_mat))
                matched_sems.append(self._match_factors(ref_mat, tgt_sem))
                matched_data.append(self._match_factors(ref_mat, tgt_data, is_data=True))

            results[dataset_key]['mean'] = matched_means
            results[dataset_key]['sem'] = matched_sems
            results[dataset_key]['data'] = matched_data
            # for ref_mat, tgt_mat in zip(ref_means, results[dataset_key]['mean']):
            #     matched_means.append(self._match_factors(ref_mat, tgt_mat))

            # results[dataset_key]['mean'] = matched_means

        # ---------- All-datasets aggregation ----------
        all_labels, all_means, all_sems, all_data = [], [], [], []

        for label, trial_blocks in pooled_by_label.items():
            all_trials = np.concatenate(trial_blocks, axis=0)
            mean_val = np.nanmean(all_trials, axis=0)
            sem_val  = sem(all_trials, axis=0, nan_policy='omit')

            all_labels.append(label)
            all_means.append(mean_val)
            all_sems.append(sem_val)
            all_data.append(all_trials)

        results['all_datasets'] = {
            'labels': all_labels,
            'mean': all_means,
            'sem': all_sems,
            'data': all_data
        }

        # ---------- Optional: interval averaging ----------
        if event_frames is not None:
            example = all_means[0]
            n_frames = example.shape[1]
            intervals = self.build_event_intervals(event_frames, n_frames, 101)
            n_events = len(intervals)

            for key in results:
                interval_means, interval_sems = [], []

                for mean_mat, sem_mat in zip(results[key]['mean'], results[key]['sem']):
                    im = np.full((mean_mat.shape[0], n_events), np.nan)
                    isem = np.full_like(im, np.nan)

                    for ev, frames in enumerate(intervals):
                        if len(frames) == 0:
                            continue

                        frames = np.asarray(frames, dtype=int)   # <-- THIS LINE

                        im[:, ev] = np.nanmean(mean_mat[:, frames], axis=1)
                        isem[:, ev] = np.nanmean(sem_mat[:, frames], axis=1)

                    interval_means.append(im)
                    interval_sems.append(isem)

                results[key]['interval_mean'] = interval_means
                results[key]['interval_sem'] = interval_sems

                results_interval[key] = {
                    'labels': results[key]['labels'],
                    'mean': interval_means,
                    'sem': interval_sems
                }

                interval_data = []

                for data_mat in results[key]['data']:  # (n_trials, n_factors, n_frames)
                    n_trials, n_factors, _ = data_mat.shape
                    idata = np.full((n_trials, n_factors, n_events), np.nan)

                    for ev, frames in enumerate(intervals):
                        if len(frames) == 0:
                            continue
                        frames = np.asarray(frames, dtype=int)
                        idata[:, :, ev] = np.nanmean(data_mat[:, :, frames], axis=2)

                    interval_data.append(idata)

                results[key]['interval_data'] = interval_data

        return results,results_interval




    

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
