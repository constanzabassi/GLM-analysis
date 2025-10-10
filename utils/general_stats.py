# import numpy as np
# import pickle
# from scipy import stats
# from scipy.stats import permutation_test
# from scipy.stats import bootstrap
# from scipy.stats import wilcoxon
# import itertools
# import pandas as pd
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import permutation_test, bootstrap
import os

class GeneralStats:
    def __init__(self):
        pass

    def calculate_bonferroni_significance(self,all_p_values, alpha=0.05):
        """
        Calculate Bonferroni corrected significance stars based on p-values.
        """
        num_tests = len(all_p_values)
        bonferroni_threshold = alpha / num_tests
        print(f"Bonferroni corrected alpha threshold: {bonferroni_threshold:.5f}")

        corrected_p_values = [p * num_tests for p in all_p_values]
        significance_stars = []

        for corrected_p in corrected_p_values:
            if corrected_p < 0.0001:
                significance_stars.append('****')
            elif corrected_p < 0.001:
                significance_stars.append('***')
            elif corrected_p < 0.01:
                significance_stars.append('**')
            elif corrected_p < 0.05:
                significance_stars.append('*')
            else:
                significance_stars.append('ns')  # Not significant
        
        return corrected_p_values, significance_stars
    
    def paired_permutation_test(self,data1, data2, num_permutations=10000):
        """Perform a paired permutation test between two datasets."""
        observed_diff = np.nanmean(data1) - np.nanmean(data2)
        combined = np.concatenate((data1, data2))
        
        more_extreme = 0
        for _ in range(num_permutations):
            np.random.shuffle(combined)
            perm_diff = np.nanmean(combined[:len(data1)]) - np.nanmean(combined[len(data1):])
            if abs(perm_diff) >= abs(observed_diff):
                more_extreme += 1
        
        p_value = (more_extreme + 1) / (num_permutations + 1)  # Adding 1 to avoid p=0
        return observed_diff, p_value


    def perform_permutation_test(self, data1, data2, paired=True, n_permutations=10000):
        """
        Perform permutation test between two groups using scipy.stats.
        
        Parameters:
        -----------
        data1, data2 : array-like
            Data arrays to compare
        paired : bool
            Whether to perform paired (True) or unpaired (False) test
        n_permutations : int
            Number of permutations to perform
        
        Returns:
        --------
        float
            p-value from permutation test
        float
            observed difference in means
        """
        data1 = np.array(data1)
        data2 = np.array(data2)
        
        # Calculate observed difference in means
        observed_diff = np.mean(data1) - np.mean(data2)
        
        if paired:
            def stat_func(x, y):
                return np.mean(x - y)
            permutation_type = "samples"
        else:
            def stat_func(x, y):
                return np.mean(x) - np.mean(y)
            permutation_type = "independent"
        
        # Perform permutation test
        res = permutation_test(
            (data1, data2), 
            stat_func,
            n_resamples=n_permutations,
            permutation_type=permutation_type
        )
        
        return res.pvalue, observed_diff
    
    def mannwhitney(self, data1, data2):
        u, p = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        return {'statistic': u, 'p_value': p}

    def ks_test(self, data1, data2):
        ks, p = stats.ks_2samp(data1, data2)
        return {'statistic': ks, 'p_value': p}


    def get_basic_stats(self, data1, n_bootstrap=1000, ci_level=0.95, random_state=None):
        """
        Compute basic statistics for a 1D array, including bootstrapped CIs for mean and std.
        Returns a dictionary 
        """
        data1 = np.array(data1)
        mean = np.nanmean(data1)
        sd = np.nanstd(data1)
        n = len(data1)
        n_nonan = np.sum(~np.isnan(data1))
        stats_dict = {
            'mean': mean,
            'sd': sd,
            'n': n,
            'n_nonan': n_nonan,
            'ci': np.nan,
            'bootstat': np.nan
        }
        # Only compute CIs if more than 1 non-NaN value
        if n_nonan > 1:
            # Remove NaNs for bootstrapping
            clean_data = data1[~np.isnan(data1)]
            # Bootstrapped CI for mean
            res_mean = bootstrap((clean_data,), np.mean, confidence_level=ci_level, n_resamples=n_bootstrap, random_state=random_state, method='basic')
            # Bootstrapped CI for std
            res_std = bootstrap((clean_data,), np.std, confidence_level=ci_level, n_resamples=n_bootstrap, random_state=random_state, method='basic')
            # ci: shape (2, 2): first col mean, second col std
            ci = np.column_stack([res_mean.confidence_interval, res_std.confidence_interval])
            stats_dict['ci'] = ci
            # bootstat: shape (n_bootstrap, 2): mean and std for each bootstrap sample
            boot_means = []
            boot_stds = []
            rng = np.random.default_rng(random_state)
            for _ in range(n_bootstrap):
                sample = rng.choice(clean_data, size=len(clean_data), replace=True)
                boot_means.append(np.mean(sample))
                boot_stds.append(np.std(sample))
            stats_dict['bootstat'] = np.column_stack([boot_means, boot_stds])
        return stats_dict
    
    def to_table(self, comparisons, stats, p_values, save_path=None, type = None):
        df = pd.DataFrame({
            'Group1': [c[0] for c in comparisons],
            'Group2': [c[1] for c in comparisons],
            'statistic': stats,
            'p_value': p_values
            if type is None else [p if p is not None else 'N/A' for p in p_values]
        })
        
        if save_path:
            df.to_csv(save_path, index=False)
        return df
    
    def basic_stats_to_table(self, stats_dict, save_path=None):
        """
        Convert a dictionary of basic stats (from get_basic_stats) to a DataFrame and optionally save it.

        Parameters
        ----------
        stats_dict : dict
            Keys are labels (e.g., group names), values are dicts from get_basic_stats.
        save_path : str, optional
            If provided, saves the DataFrame to this path (CSV or Excel).

        Returns
        -------
        df : pandas.DataFrame
            DataFrame with the basic stats.
        """
        # Flatten the stats_dict into a list of rows
        rows = []
        for label, stats in stats_dict.items():
            row = {'Label': label}
            row.update({k: v for k, v in stats.items() if k not in ['bootstat']})  # Exclude bootstat for table
            rows.append(row)
        df = pd.DataFrame(rows)
        
        if save_path:
            if save_path.endswith('.csv'):
                df.to_csv(save_path, index=False)
            elif save_path.endswith('.xlsx'):
                df.to_excel(save_path, index=False)
            else:
                raise ValueError("save_path must end with .csv or .xlsx")
        return df