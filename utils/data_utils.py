from helper_functions.data_pipeline import DataPipeline

def load_experimental_data(datasets, load_celltypes=True):
    """
    Load experimental data using DataPipeline.
    
    Parameters:
    -----------
    datasets : list
        List of tuples containing (animalID, date, drive)
    load_celltypes : bool, optional
        Whether to load cell type information
        
    Returns:
    --------
    tuple
        (data_loaders, celltype_info)
    """
    pipeline = DataPipeline()
    data_loaders, celltype_info = pipeline.load_data(
        datasets=datasets,
        load_celltypes=load_celltypes
    )
    return data_loaders, celltype_info