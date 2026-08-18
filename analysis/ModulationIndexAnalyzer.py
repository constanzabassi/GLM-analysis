"""Pre/post sound modulation index on aligned neural activity."""

from __future__ import annotations

from typing import Dict, Hashable, Optional

import numpy as np


class ModulationIndexAnalyzer:
    """Simple pooled-trial pre/post modulation index (Bassirunyan-compatible).

    MI = (mean_post - mean_pre) / (mean_post + mean_pre)

    Expected aligned format:
        aligned[key][fold] -> (n_trials, n_neurons, n_frames)
    """

    def __init__(
        self,
        pre_frames,
        post_frames,
        n_shuffles: int = 10000,
        alpha: float = 0.05,
        eps: float = 1e-12,
        random_state: Optional[int] = 0,
    ):
        self.pre_frames = np.asarray(pre_frames, dtype=int)
        self.post_frames = np.asarray(post_frames, dtype=int)
        self.n_shuffles = int(n_shuffles)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.random_state = random_state
        self._rng = np.random.default_rng(random_state)

    def compute_mod_index_prepost(
        self,
        post_avg: np.ndarray,
        pre_avg: np.ndarray,
    ) -> np.ndarray:
        """Compute neuron-wise MI from trial x neuron window averages.

        Parameters
        ----------
        post_avg, pre_avg : array, shape (n_trials, n_neurons)

        Returns
        -------
        mi : array, shape (n_neurons,)
        """
        post_avg = np.asarray(post_avg, dtype=float)
        pre_avg = np.asarray(pre_avg, dtype=float)
        if post_avg.ndim != 2 or pre_avg.ndim != 2:
            raise ValueError("post_avg and pre_avg must be 2D (n_trials, n_neurons)")
        if post_avg.shape[1] != pre_avg.shape[1]:
            raise ValueError("post_avg and pre_avg must share n_neurons")

        post_val = np.nanmean(post_avg, axis=0)
        pre_val = np.nanmean(pre_avg, axis=0)
        denom = post_val + pre_val

        mi = np.zeros(post_val.shape[0], dtype=float)
        valid = np.isfinite(denom) & (np.abs(denom) >= self.eps)
        mi[valid] = (post_val[valid] - pre_val[valid]) / denom[valid]
        return mi

    def window_trial_averages(
        self,
        activity_3d: np.ndarray,
        pre_frames=None,
        post_frames=None,
    ):
        """Average frames within pre/post windows.

        Parameters
        ----------
        activity_3d : array, shape (n_trials, n_neurons, n_frames)

        Returns
        -------
        pre_avg, post_avg : arrays, shape (n_trials, n_neurons)
        """
        activity_3d = np.asarray(activity_3d, dtype=float)
        if activity_3d.ndim != 3:
            raise ValueError("activity_3d must be (n_trials, n_neurons, n_frames)")

        pre_idx = self.pre_frames if pre_frames is None else np.asarray(pre_frames, dtype=int)
        post_idx = self.post_frames if post_frames is None else np.asarray(post_frames, dtype=int)

        n_frames = activity_3d.shape[2]
        if np.any(pre_idx < 0) or np.any(pre_idx >= n_frames):
            raise ValueError(f"pre_frames out of range for n_frames={n_frames}")
        if np.any(post_idx < 0) or np.any(post_idx >= n_frames):
            raise ValueError(f"post_frames out of range for n_frames={n_frames}")

        pre_avg = np.nanmean(activity_3d[:, :, pre_idx], axis=2)
        post_avg = np.nanmean(activity_3d[:, :, post_idx], axis=2)
        return pre_avg, post_avg

    def bootstrap_mod_index_prepost(
        self,
        post_avg: np.ndarray,
        pre_avg: np.ndarray,
        n_shuffles: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        """Two-sided p-values via unpaired label shuffle of trial averages.

        Matches Bassirunyan ``bootstrap_mod_index_cv`` for prepost* types:
        stack pre/post trial means, permute labels preserving group sizes,
        p = (# |boot| >= |obs| + 1) / (n_shuffles + 1).
        """
        post_avg = np.asarray(post_avg, dtype=float)
        pre_avg = np.asarray(pre_avg, dtype=float)
        n_shuffles = self.n_shuffles if n_shuffles is None else int(n_shuffles)

        observed = self.compute_mod_index_prepost(post_avg, pre_avg)
        n_post = post_avg.shape[0]
        n_pre = pre_avg.shape[0]
        n_neurons = post_avg.shape[1]
        combined = np.vstack([post_avg, pre_avg])
        n_total = combined.shape[0]

        rng = self._rng if random_state is None else np.random.default_rng(random_state)
        boot = np.empty((n_shuffles, n_neurons), dtype=float)

        for i in range(n_shuffles):
            perm = rng.permutation(n_total)
            sim_post = combined[perm[:n_post], :]
            sim_pre = combined[perm[n_post:n_post + n_pre], :]
            boot[i, :] = self.compute_mod_index_prepost(sim_post, sim_pre)

        abs_obs = np.abs(observed)
        abs_boot = np.abs(boot)
        counts = np.sum(abs_boot >= abs_obs[None, :], axis=0)
        p_values = (counts + 1.0) / (n_shuffles + 1.0)
        return p_values

    def compute_mi_for_aligned_tensor(
        self,
        activity_3d: np.ndarray,
        compute_p: bool = True,
        n_shuffles: Optional[int] = None,
    ) -> dict:
        """MI (and optional bootstrap p) for one aligned tensor."""
        pre_avg, post_avg = self.window_trial_averages(activity_3d)
        mi = self.compute_mod_index_prepost(post_avg, pre_avg)

        out = {
            "mi": mi,
            "pre_avg": pre_avg,
            "post_avg": post_avg,
            "p": None,
            "sig": None,
        }
        if compute_p:
            p = self.bootstrap_mod_index_prepost(
                post_avg, pre_avg, n_shuffles=n_shuffles
            )
            out["p"] = p
            out["sig"] = p <= self.alpha
        return out

    def compute_mi_across_folds(
        self,
        aligned_dict: Dict[Hashable, Dict[Hashable, np.ndarray]],
        compute_p: bool = False,
        n_shuffles: Optional[int] = None,
    ) -> dict:
        """Compute MI per fold, then average MI across folds.

        Per-fold p-values are optional and are *not* averaged. Prefer
        ``compute_pooled_bootstrap_across_folds`` for dataset-level significance.
        """
        mi_by_key_fold = {}
        p_by_key_fold = {}
        mi_mean_by_key = {}
        mi_sem_by_key = {}

        for key, folds in aligned_dict.items():
            mi_by_key_fold[key] = {}
            p_by_key_fold[key] = {}
            fold_mis = []

            for fold in sorted(folds.keys(), key=lambda x: (str(type(x)), x)):
                result = self.compute_mi_for_aligned_tensor(
                    folds[fold],
                    compute_p=compute_p,
                    n_shuffles=n_shuffles,
                )
                mi_by_key_fold[key][fold] = result["mi"]
                p_by_key_fold[key][fold] = result["p"]
                fold_mis.append(result["mi"])

            stacked = np.stack(fold_mis, axis=0)
            mi_mean_by_key[key] = np.nanmean(stacked, axis=0)
            if stacked.shape[0] > 1:
                mi_sem_by_key[key] = np.nanstd(stacked, axis=0, ddof=1) / np.sqrt(
                    stacked.shape[0]
                )
            else:
                mi_sem_by_key[key] = np.full(stacked.shape[1], np.nan)

        return {
            "mi_by_key_fold": mi_by_key_fold,
            "p_by_key_fold": p_by_key_fold,
            "mi_mean_by_key": mi_mean_by_key,
            "mi_sem_by_key": mi_sem_by_key,
        }

    def compute_pooled_bootstrap_across_folds(
        self,
        aligned_dict: Dict[Hashable, Dict[Hashable, np.ndarray]],
        n_shuffles: Optional[int] = None,
    ) -> dict:
        """Concatenate pre/post trial averages across folds; bootstrap once per key.

        Does not average p-values. Observed MI used for p is from pooled trials.
        """
        mi_pooled_by_key = {}
        p_by_key = {}
        sig_by_key = {}
        n_trials_by_key = {}

        for key, folds in aligned_dict.items():
            pre_list = []
            post_list = []
            for fold in sorted(folds.keys(), key=lambda x: (str(type(x)), x)):
                pre_avg, post_avg = self.window_trial_averages(folds[fold])
                pre_list.append(pre_avg)
                post_list.append(post_avg)

            pre_pooled = np.concatenate(pre_list, axis=0)
            post_pooled = np.concatenate(post_list, axis=0)
            mi = self.compute_mod_index_prepost(post_pooled, pre_pooled)
            p = self.bootstrap_mod_index_prepost(
                post_pooled, pre_pooled, n_shuffles=n_shuffles
            )

            mi_pooled_by_key[key] = mi
            p_by_key[key] = p
            sig_by_key[key] = p <= self.alpha
            n_trials_by_key[key] = {
                "n_pre": pre_pooled.shape[0],
                "n_post": post_pooled.shape[0],
            }

        return {
            "mi_pooled_by_key": mi_pooled_by_key,
            "p_by_key": p_by_key,
            "sig_by_key": sig_by_key,
            "n_trials_by_key": n_trials_by_key,
        }

    def compute_mi_true_vs_pred(
        self,
        aligned_true: Dict[Hashable, Dict[Hashable, np.ndarray]],
        aligned_pred: Dict[Hashable, Dict[Hashable, np.ndarray]],
        compute_p: bool = False,
        pooled_bootstrap: bool = False,
        n_shuffles: Optional[int] = None,
    ) -> dict:
        """Compute fold-averaged MI for true and predicted aligned activity."""
        true_res = self.compute_mi_across_folds(
            aligned_true, compute_p=compute_p, n_shuffles=n_shuffles
        )
        pred_res = self.compute_mi_across_folds(
            aligned_pred, compute_p=compute_p, n_shuffles=n_shuffles
        )

        comparison = {}
        for key in true_res["mi_mean_by_key"]:
            if key not in pred_res["mi_mean_by_key"]:
                continue
            mi_t = true_res["mi_mean_by_key"][key]
            mi_p = pred_res["mi_mean_by_key"][key]
            comparison[key] = {
                "mi_true_mean": mi_t,
                "mi_pred_mean": mi_p,
                "mi_residual": mi_t - mi_p,
            }

        out = {
            "true": true_res,
            "pred": pred_res,
            "comparison": comparison,
            "true_pooled_bootstrap": None,
            "pred_pooled_bootstrap": None,
        }

        if pooled_bootstrap:
            out["true_pooled_bootstrap"] = self.compute_pooled_bootstrap_across_folds(
                aligned_true, n_shuffles=n_shuffles
            )
            out["pred_pooled_bootstrap"] = self.compute_pooled_bootstrap_across_folds(
                aligned_pred, n_shuffles=n_shuffles
            )

        return out

    def compute_prepost_delta(
        self,
        post_avg: np.ndarray,
        pre_avg: np.ndarray,
    ) -> np.ndarray:
        """Neuron-wise post−pre difference from trial x neuron window averages.

        Parameters
        ----------
        post_avg, pre_avg : array, shape (n_trials, n_neurons)

        Returns
        -------
        delta : array, shape (n_neurons,)
            ``nanmean(post) - nanmean(pre)``. Same windows/trials as MI, without
            the MI denominator.
        """
        post_avg = np.asarray(post_avg, dtype=float)
        pre_avg = np.asarray(pre_avg, dtype=float)
        if post_avg.ndim != 2 or pre_avg.ndim != 2:
            raise ValueError("post_avg and pre_avg must be 2D (n_trials, n_neurons)")
        if post_avg.shape[1] != pre_avg.shape[1]:
            raise ValueError("post_avg and pre_avg must share n_neurons")

        post_val = np.nanmean(post_avg, axis=0)
        pre_val = np.nanmean(pre_avg, axis=0)
        return post_val - pre_val

    def bootstrap_prepost_delta(
        self,
        post_avg: np.ndarray,
        pre_avg: np.ndarray,
        n_shuffles: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> np.ndarray:
        """Optional two-sided p-values for post−pre delta via label shuffle.

        Same unpaired shuffle as ``bootstrap_mod_index_prepost``, with delta as
        the statistic. P-values are not averaged across folds.
        """
        post_avg = np.asarray(post_avg, dtype=float)
        pre_avg = np.asarray(pre_avg, dtype=float)
        n_shuffles = self.n_shuffles if n_shuffles is None else int(n_shuffles)

        observed = self.compute_prepost_delta(post_avg, pre_avg)
        n_post = post_avg.shape[0]
        n_pre = pre_avg.shape[0]
        n_neurons = post_avg.shape[1]
        combined = np.vstack([post_avg, pre_avg])
        n_total = combined.shape[0]

        rng = self._rng if random_state is None else np.random.default_rng(random_state)
        boot = np.empty((n_shuffles, n_neurons), dtype=float)
        for i in range(n_shuffles):
            perm = rng.permutation(n_total)
            sim_post = combined[perm[:n_post], :]
            sim_pre = combined[perm[n_post:n_post + n_pre], :]
            boot[i, :] = self.compute_prepost_delta(sim_post, sim_pre)

        abs_obs = np.abs(observed)
        counts = np.sum(np.abs(boot) >= abs_obs[None, :], axis=0)
        return (counts + 1.0) / (n_shuffles + 1.0)

    def compute_delta_for_aligned_tensor(
        self,
        activity_3d: np.ndarray,
        compute_p: bool = False,
        n_shuffles: Optional[int] = None,
    ) -> dict:
        """Post−pre delta for one aligned tensor (parallel to MI tensor method)."""
        pre_avg, post_avg = self.window_trial_averages(activity_3d)
        delta = self.compute_prepost_delta(post_avg, pre_avg)
        pre_mean = np.nanmean(pre_avg, axis=0)
        post_mean = np.nanmean(post_avg, axis=0)

        out = {
            "delta": delta,
            "pre_mean": pre_mean,
            "post_mean": post_mean,
            "pre_avg": pre_avg,
            "post_avg": post_avg,
            "p": None,
            "sig": None,
        }
        if compute_p:
            p = self.bootstrap_prepost_delta(
                post_avg, pre_avg, n_shuffles=n_shuffles
            )
            out["p"] = p
            out["sig"] = p <= self.alpha
        return out

    def _delta_across_folds_full(
        self,
        aligned_dict: Dict[Hashable, Dict[Hashable, np.ndarray]],
        compute_p: bool = False,
        n_shuffles: Optional[int] = None,
    ) -> dict:
        """Per-fold delta, then fold-mean delta. Does not average p-values."""
        delta_by_key_fold = {}
        p_by_key_fold = {}
        delta_mean_by_key = {}
        delta_sem_by_key = {}

        for key, folds in aligned_dict.items():
            delta_by_key_fold[key] = {}
            p_by_key_fold[key] = {}
            fold_deltas = []

            for fold in sorted(folds.keys(), key=lambda x: (str(type(x)), x)):
                result = self.compute_delta_for_aligned_tensor(
                    folds[fold],
                    compute_p=compute_p,
                    n_shuffles=n_shuffles,
                )
                delta_by_key_fold[key][fold] = result["delta"]
                p_by_key_fold[key][fold] = result["p"]
                fold_deltas.append(result["delta"])

            stacked = np.stack(fold_deltas, axis=0)
            delta_mean_by_key[key] = np.nanmean(stacked, axis=0)
            if stacked.shape[0] > 1:
                delta_sem_by_key[key] = np.nanstd(stacked, axis=0, ddof=1) / np.sqrt(
                    stacked.shape[0]
                )
            else:
                delta_sem_by_key[key] = np.full(stacked.shape[1], np.nan)

        return {
            "delta_by_key_fold": delta_by_key_fold,
            "p_by_key_fold": p_by_key_fold,
            "delta_mean_by_key": delta_mean_by_key,
            "delta_sem_by_key": delta_sem_by_key,
        }

    def compute_delta_across_folds(
        self,
        aligned_dict: Dict[Hashable, Dict[Hashable, np.ndarray]],
        average_folds: bool = True,
        compute_p: bool = False,
        n_shuffles: Optional[int] = None,
    ):
        """Compute delta per fold, then optionally average delta across folds.

        Parameters
        ----------
        aligned_dict : dict
            ``aligned_dict[key][fold] = (n_trials, n_neurons, n_frames)``
        average_folds : bool
            If True, return ``delta_dict[key] = (n_neurons,)``.
            If False, return ``delta_dict[key][fold] = (n_neurons,)``.
        """
        full = self._delta_across_folds_full(
            aligned_dict, compute_p=compute_p, n_shuffles=n_shuffles
        )
        if average_folds:
            return full["delta_mean_by_key"]
        return full["delta_by_key_fold"]

    def compute_delta_true_vs_pred(
        self,
        aligned_true: Dict[Hashable, Dict[Hashable, np.ndarray]],
        aligned_pred: Dict[Hashable, Dict[Hashable, np.ndarray]],
        average_folds: bool = True,
        compute_p: bool = False,
        n_shuffles: Optional[int] = None,
    ) -> dict:
        """True vs predicted post−pre delta, parallel to ``compute_mi_true_vs_pred``.

        Returns
        -------
        dict
            ``out["true"]`` / ``out["pred"]`` are neuron-wise dicts:
            fold-averaged if ``average_folds=True``, else per-fold.
        """
        true_full = self._delta_across_folds_full(
            aligned_true, compute_p=compute_p, n_shuffles=n_shuffles
        )
        pred_full = self._delta_across_folds_full(
            aligned_pred, compute_p=compute_p, n_shuffles=n_shuffles
        )

        if average_folds:
            delta_true = true_full["delta_mean_by_key"]
            delta_pred = pred_full["delta_mean_by_key"]
        else:
            delta_true = true_full["delta_by_key_fold"]
            delta_pred = pred_full["delta_by_key_fold"]

        comparison = {}
        residual = {}
        abs_true = {}
        abs_pred = {}
        abs_true_minus_abs_pred = {}
        if average_folds:
            for key in delta_true:
                if key not in delta_pred:
                    continue
                d_t = np.asarray(delta_true[key], dtype=float)
                d_p = np.asarray(delta_pred[key], dtype=float)
                comparison[key] = {
                    "delta_true_mean": d_t,
                    "delta_pred_mean": d_p,
                    "delta_residual": d_t - d_p,
                }
                residual[key] = d_t - d_p
                abs_true[key] = np.abs(d_t)
                abs_pred[key] = np.abs(d_p)
                abs_true_minus_abs_pred[key] = abs_true[key] - abs_pred[key]

        return {
            "true": delta_true,
            "pred": delta_pred,
            "true_minus_pred": residual,
            "abs_true": abs_true,
            "abs_pred": abs_pred,
            "abs_true_minus_abs_pred": abs_true_minus_abs_pred,
            "true_by_fold": true_full["delta_by_key_fold"],
            "pred_by_fold": pred_full["delta_by_key_fold"],
            "true_sem_by_key": true_full["delta_sem_by_key"],
            "pred_sem_by_key": pred_full["delta_sem_by_key"],
            "comparison": comparison,
        }
