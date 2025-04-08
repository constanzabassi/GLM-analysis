import os
import sys
import numpy as np

# def setup_paths(base_dir=None):
#     """Setup paths for importing from different directories."""
#     if base_dir is None:
#         base_dir = 'C:\\Code\\Github\\States_pupil_analysis'
    
#     # Add States_pupil_analysis paths
#     sys.path.append(base_dir)
#     sys.path.append(os.path.join(base_dir, 'helper_functions'))
    
#     # Add GLM-analysis root to path
#     glm_dir = 'C:\\Code\\Github\\GLM-analysis'
#     sys.path.append(glm_dir)
    
#     return base_dir

def setup_paths(base_dir=None):
    """Setup paths for importing from different directories."""
    # Set up States_pupil_analysis paths
    if base_dir is None:
        base_dir = 'C:\\Code\\Github\\States_pupil_analysis'
    
    # Set up GLM-analysis paths
    glm_dir = 'C:\\Code\\Github\\GLM-analysis'
    
    paths = [
        base_dir,  # States_pupil_analysis base
        os.path.join(base_dir, 'helper_functions'),
        glm_dir,  # GLM-analysis base
        os.path.join(glm_dir, 'handlers'),
        os.path.join(glm_dir, 'analysis'),
        os.path.join(glm_dir, 'config'),
        os.path.join(glm_dir, 'utils')
    ]
    
    for path in paths:
        if path not in sys.path:
            sys.path.append(path)
    
    return base_dir