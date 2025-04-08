from handlers.DataHandlerDecoding import DataHandlerDecoding  # Fixed path

class DatasetConfig:
    def __init__(self):
        self.all_datasets = {}
        self.mouse_dates_keys = []
        # Define which datasets lack specific decoded variables
        self.variable_indices = {
            'sound_category': {
                'missing': [8, 22],
                'present': []
            },
            'choice': {
                'missing': [8, 22],  # same as sound
                'present': []
            },
            'photostim': {
                'missing': [24],
                'present': []
            },
            'outcome': {
                'missing': [2, 7, 8, 20, 21, 22],
                'present': []
            }
        }

    def load_from_info(self, info_dir, data_handler):
        """Load datasets and populate available variables"""
        datasets, mouse_dates_keys = data_handler.load_info(info_dir)
        
        # Populate present indices for each variable
        all_indices = set(range(len(datasets)))
        for var in self.variable_indices:
            missing = set(self.variable_indices[var]['missing'])
            self.variable_indices[var]['present'] = list(all_indices - missing)
        
        # Store datasets with their variable availability
        for i, (dataset, key) in enumerate(zip(datasets, mouse_dates_keys)):
            animalID, date, server = dataset
            self.all_datasets[key] = {
                'animalID': animalID,
                'date': date,
                'server': server,
                'available_variables': {
                    var: i not in self.variable_indices[var]['missing']
                    for var in self.variable_indices
                }
            }
        
        self.mouse_dates_keys = mouse_dates_keys
        return datasets, mouse_dates_keys
    
    def get_datasets(self):
        """Get all datasets as list of tuples"""
        return [(self.all_datasets[key]['animalID'], 
                self.all_datasets[key]['date'], 
                self.all_datasets[key]['server']) 
                for key in self.mouse_dates_keys]

    def get_datasets_with_variables(self, variables=None, require_all=True, include_datasets=None):
        """
        Get datasets based on variable availability and optional dataset inclusion

        Parameters:
        -----------
        variables : list or None
            List of required variables. If None, returns all datasets
        require_all : bool
            If True, returns datasets with all specified variables
        include_datasets : list or None
            List of dataset indices to include regardless of variable availability
        """
        if not variables:
            filtered_datasets = self.get_datasets()
            print("\nUsing all datasets (no variables specified)")
            for animalID, date, server in filtered_datasets:
                print(f"Animal: {animalID}, Date: {date}")
            return filtered_datasets, self.mouse_dates_keys

        # Get unique base variable names
        unique_vars = set()
        for var in variables:
            base_var = var.replace('shuffled/', '')
            unique_vars.add(base_var)
        
        print(f"\nUnique base variables: {unique_vars}")
        
        # Initialize filtered lists
        filtered_datasets = []
        filtered_keys = []

        if len(unique_vars) == 1 or (len(unique_vars) == 2 and 
            {'sound_category', 'choice'}.issubset(unique_vars)):
            
            indices_to_remove = set()
            for var in unique_vars:
                base_var = var.replace('shuffled/', '')
                if base_var in self.variable_indices:
                    indices_to_remove.update(self.variable_indices[base_var]['missing'])
            
            # Include specified datasets
            if include_datasets:
                indices_to_remove = indices_to_remove - set(include_datasets)
            
            print(f"Indices to remove: {indices_to_remove}")
            
            # Filter datasets
            all_datasets = self.get_datasets()
            for i, (dataset, key) in enumerate(zip(all_datasets, self.mouse_dates_keys)):
                if i not in indices_to_remove:
                    filtered_datasets.append(dataset)
                    filtered_keys.append(key)
        else:
            filtered_datasets = self.get_datasets()
            filtered_keys = self.mouse_dates_keys.copy()

        print(f"\nFiltered datasets ({len(filtered_datasets)}):")
        for animalID, date, server in filtered_datasets:
            print(f"Animal: {animalID}, Date: {date}")

        return filtered_datasets, filtered_keys
    
    def get_specific_datasets(self, dataset_indices):
        """
        Get specific datasets by their indices
        
        Parameters:
        -----------
        dataset_indices : list
            List of dataset indices to load
            
        Returns:
        --------
        list, list
            Filtered datasets and corresponding keys
        """
        all_datasets = self.get_datasets()
        filtered_datasets = [all_datasets[i] for i in dataset_indices]
        filtered_keys = [self.mouse_dates_keys[i] for i in dataset_indices]
        
        print(f"\nLoading specific datasets:")
        for animalID, date, server in filtered_datasets:
            print(f"Animal: {animalID}, Date: {date}")
        
        return filtered_datasets, filtered_keys
        
    def print_datasets(self, filtered_datasets=None, filtered_keys=None):
        """
        Print datasets with their available variables
        
        Parameters:
        -----------
        filtered_datasets : list, optional
            List of filtered datasets to print
        filtered_keys : list, optional
            Corresponding keys for filtered datasets
        """
        if filtered_datasets is None:
            datasets_to_print = self.get_datasets()
            keys_to_print = self.mouse_dates_keys
        else:
            datasets_to_print = filtered_datasets
            keys_to_print = filtered_keys
            
        print("\nAvailable Datasets:")
        print("-" * 60)
        for (animalID, date, server), key in zip(datasets_to_print, keys_to_print):
            print(f"\nDataset: {key}")
            print(f"Animal: {animalID}, Date: {date}")
            if key in self.all_datasets:
                print("Available variables:")
                for var, available in self.all_datasets[key]['available_variables'].items():
                    status = "✓" if available else "✗"
                    print(f"  {var}: {status}")
        print("-" * 60)