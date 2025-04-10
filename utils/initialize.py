from typing import Dict, Any
from .path_utils import setup_paths
from helper_functions.data_aligner import DataAligner

def init_modules() -> Dict[str, Any]:
    """
    Initialize all required modules after path setup.
    
    Returns:
        Dict containing module functions
    """
     # First set up paths
    setup_paths()
    
    # Now we can safely import after paths are set
    from helper_functions.data_aligner import DataAligner
    from helper_functions.data_pipeline import DataPipeline
    
    def load_experimental_data(datasets):
        """Load experimental data using DataPipeline."""
        # from helper_functions.data_pipeline import DataPipeline
        pipeline = DataPipeline()
        return pipeline.load_data(datasets=datasets, load_celltypes=True)
    
    def setup_and_align_data(data_loader, alignment='all'):
        """Setup and align data for a single dataset."""
        # Load data
        neural_data, good_trials, trial_info, movement_in_imaging, frame_id_events, file_num, velocity_data, imaging = (
            data_loader.load_neural_data(neural_data_type='dff')
        )
        
        # Align frames
        imaging_frame_lengths = data_loader.load_alignment_data()
        global_frame_ids = data_loader.align_frames_to_session(file_num, frame_id_events, imaging_frame_lengths)
        
        # Define movement frames
        movement_frames = {}
        for trial in range(len(neural_data)):
            if trial in good_trials:
                movement_frames[trial] = {
                    'maze_frames': movement_in_imaging[trial]['maze_frames'],
                    'reward_frames': movement_in_imaging[trial]['reward_frames'],
                    'iti_frames': movement_in_imaging[trial]['iti_frames']
                }
            else:
                movement_frames[trial] = None
        
        # Initialize aligner and align data
        aligner = DataAligner(neural_data, movement_frames, velocity_data, global_frame_ids, good_trials)
        align_info, alignment_frames, left_padding, right_padding = aligner.find_align_info(imaging, 30, alternative_alignment=False)
        
        aligned_imaging, imaging_array, align_info, frames = aligner.align_behavior_data(
            imaging, align_info, alignment_frames, left_padding, right_padding, 
            alignment=alignment, cell_ids=None
        )
        
        return aligned_imaging, trial_info, good_trials
    
    return {
        'load_experimental_data': load_experimental_data,
        'setup_and_align_data': setup_and_align_data
    }