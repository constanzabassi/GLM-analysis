from typing import Dict, List, Tuple, Optional
import numpy as np
import pickle
from .alignment_utils import setup_and_align_data

class DatasetProcessor:
    def __init__(self, alignment: Dict = None):
        """
        Initialize processor with alignment parameters.
        
        Args:
            alignment: Dict with 'type' and 'data_type' keys
        """
        self.alignment = alignment or {
            'type': 'pre',
            'data_type': 'z_dff'
        }
        
    def process_datasets(self, 
                        data_loaders: List, 
                        celltype_info: Dict,
                        save_path: Optional[str] = None) -> Dict:
        """
        Process multiple datasets and store aligned data.
        
        Args:
            data_loaders: List of DataLoader instances
            celltype_info: Dictionary containing cell type information
            save_path: Optional path to save aligned data
            
        Returns:
            Dictionary containing aligned data for each dataset
        """
        aligned_data = {}
        
        for data_loader, (key, info) in zip(data_loaders, celltype_info.items()):
            animalID, date = key
            celltypes = celltype_info[animalID, date]['neuron_groups']
            
            # Align data
            aligned_imaging, trial_info, good_trials = setup_and_align_data(
                data_loader, 
                alignment=self.alignment
            )
            
            # Store data
            aligned_data[key] = {
                'aligned_imaging': aligned_imaging,
                'trial_info': trial_info,
                'good_trials': good_trials,
                'celltypes': celltypes
            }
        
        if save_path:
            with open(save_path, 'wb') as f:
                pickle.dump(aligned_data, f)
        
        return aligned_data
    
    @staticmethod
    def load_aligned_data(file_path: str) -> Dict:
        """Load previously saved aligned data."""
        with open(file_path, 'rb') as f:
            return pickle.load(f)