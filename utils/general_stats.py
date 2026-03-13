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
from scipy.stats import kruskal

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
    
    
    def kruskal_wallis_to_pd(self,key, data1, data2,data3=None):
        if data3 is not None:
            kw_stat, kw_p_value = kruskal(data1, data2, data3)
        else:
            kw_stat, kw_p_value = kruskal(data1, data2)
        kw_row = pd.DataFrame({
        'Group1': [key],
        'Group2': [key],
        'statistic': [kw_stat],
        'p_value': [kw_p_value]
        # 'test_type': ['kruskal-wallis']
        })
        return kw_row

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

        if n_nonan > 1:
            sem = sd / np.sqrt(n_nonan)
        else:
            sem = np.nan

        stats_dict = {
            'mean': mean,
            'sd': sd,
            'sem': sem,
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
    
    def permutation_test_quadrants(self,
        counts_group1: np.ndarray,
        counts_group2: np.ndarray,
        group1_name='sound',
        group2_name='opto',
        n_permutations=10000,
        seed=None,
        save_path=None
    ):
        """
        Performs a permutation test comparing quadrant fractions between two groups.

        Parameters:
            counts_group1 (np.ndarray): shape (n_datasets_1, 4), raw counts for group 1
            counts_group2 (np.ndarray): shape (n_datasets_2, 4), raw counts for group 2
            group1_name (str): Label for group 1
            group2_name (str): Label for group 2
            n_permutations (int): Number of permutations
            seed (int): Random seed for reproducibility
            save_path (str): Optional path to save permutation test and summary stats

        Returns:
            p_value (float): Permutation test p-value
            observed_stat (float): L1 norm of difference in means
            permuted_stats (np.ndarray): Distribution of permuted test statistics
        """
        if seed is not None:
            np.random.seed(seed)

        # Normalize to get fractions per dataset
        def normalize_counts(counts):
            return np.array([row / np.sum(row) if np.sum(row) > 0 else np.zeros_like(row) for row in counts])

        norm1 = normalize_counts(counts_group1)
        norm2 = normalize_counts(counts_group2)

        # Compute observed difference (L1 norm between group means)
        mean1 = np.mean(norm1, axis=0)
        mean2 = np.mean(norm2, axis=0)
        observed_stat = np.sum(np.abs(mean1 - mean2))

        # Stack data and generate labels
        all_data = np.vstack([norm1, norm2])
        group_labels = np.array([0] * len(norm1) + [1] * len(norm2))

        # Permutation loop
        permuted_stats = []
        for _ in range(n_permutations):
            shuffled = np.random.permutation(group_labels)
            grp_a = all_data[shuffled == 0]
            grp_b = all_data[shuffled == 1]
            stat = np.sum(np.abs(np.mean(grp_a, axis=0) - np.mean(grp_b, axis=0)))
            permuted_stats.append(stat)

        permuted_stats = np.array(permuted_stats)
        p_value = np.mean(permuted_stats >= observed_stat)

        # Save summary + CSVs
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # Save mean/std summary
            summary_df = pd.DataFrame({
                'Quadrant': ['+/+', '+/–', '–/+', '–/–'],
                f'{group1_name}_mean': mean1,
                f'{group1_name}_std': np.std(norm1, axis=0),
                f'{group2_name}_mean': mean2,
                f'{group2_name}_std': np.std(norm2, axis=0),
            })
            summary_path = save_path.replace('.csv', '_summary_stats.csv')
            summary_df.to_csv(summary_path, index=False)

            # Save permutation result
            perm_df = pd.DataFrame({
                'observed_stat': [observed_stat],
                'p_value': [p_value]
            })
            perm_df.to_csv(save_path, index=False)

            print(f"Saved permutation test to: {save_path}")
            print(f"Saved summary stats to: {summary_path}")

        return p_value, observed_stat, permuted_stats
    

    def coupling_stats_by_celltype(self,all_df, celltypecolors, n_perm=10000):

        # compute difference metric
        all_df = all_df.copy()
        all_df['within_minus_between'] = all_df['coupling_within'] - all_df['coupling_between']

        # organize data by celltype
        values_by_celltype = {}

        for celltype in celltypecolors.keys():
            vals = all_df.loc[all_df['group'] == celltype, 'within_minus_between'].values
            vals = vals[~np.isnan(vals)]
            values_by_celltype[celltype] = vals

        # ----------------------------
        # Kruskal-Wallis across groups
        # ----------------------------

        groups = [values_by_celltype[k] for k in celltypecolors.keys()]

        kw_table = self.kruskal_wallis_to_pd(
            'within_minus_between',
            *groups
        )

        kw_significant = (kw_table["p_value"] < 0.05).any()

        # ----------------------------
        # Pairwise permutation tests
        # ----------------------------

        celltype_keys = list(celltypecolors.keys())

        all_p_values = []
        comparisons = []
        comparisons_names = []
        test_stats = []
        all_stats_dict = {}

        for i, celltype in enumerate(celltype_keys):

            for j in range(i + 1, len(celltype_keys)):

                other_celltype = celltype_keys[j]

                comparisons.append((i, j))

                data_i = values_by_celltype[celltype]
                data_j = values_by_celltype[other_celltype]

                if len(data_i) == 0 or len(data_j) == 0:
                    print(f"No data for {celltype} or {other_celltype}. Skipping.")
                    continue

                p_value, stat = self.perform_permutation_test(
                    data_i,
                    data_j,
                    paired=False,
                    n_permutations=n_perm
                )

                all_p_values.append(p_value)
                test_stats.append(stat)

                comparisons_names.append(
                    (f"{celltype}_within_minus_between",
                    f"{other_celltype}_within_minus_between")
                )

                print(f"Permutation test {celltype} vs {other_celltype}: p={p_value:.4f}")

                # store basic stats
                label1 = f"{celltype}_within_minus_between"
                label2 = f"{other_celltype}_within_minus_between"

                all_stats_dict[label1] = self.get_basic_stats(data_i)
                all_stats_dict[label2] = self.get_basic_stats(data_j)

        # Bonferroni correction
        _, significance_stars = self.calculate_bonferroni_significance(
            all_p_values,
            alpha=0.05
        )

        # create tables
        df_tests = self.to_table(
            comparisons_names,
            test_stats,
            all_p_values,
            type='permutation unpaired'
        )

        df_tests = pd.concat([df_tests, kw_table], ignore_index=True)

        df_stats = self.basic_stats_to_table(all_stats_dict)

        return df_tests, df_stats, significance_stars, comparisons, kw_significant