"""Pre/post sound modulation index on aligned neural activity."""

from __future__ import annotations

from typing import Dict, Hashable, Optional, Sequence, Tuple, Union

import numpy as np


class ModulationIndexAnalyzer:
    """Simple pooled-trial pre/post modulation index (Bassirunyan-compatible).

    MI = (mean_post - mean_pre) / (mean_post + mean_pre)

    Expected aligned format:
        aligned[key][fold] -> (n_trials, n_neurons, n_frames)

    Optional condition-aware analysis uses ``conditions_updated``-style arrays:
        row/col 0: sound side (1=left, 2=right)
        row/col 1: photostim (0/1)
        row/col 2: correct (0/1; unused unless requested)
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

    def _apply_trial_mask(
        self,
        activity_3d: np.ndarray,
        trial_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Subset trials by boolean/index mask; preserve behavior when mask is None."""
        activity_3d = np.asarray(activity_3d, dtype=float)
        if activity_3d.ndim != 3:
            raise ValueError("activity_3d must be (n_trials, n_neurons, n_frames)")
        if trial_mask is None:
            return activity_3d

        trial_mask = np.asarray(trial_mask)
        n_trials = activity_3d.shape[0]
        if trial_mask.ndim != 1:
            raise ValueError("trial_mask must be 1D")
        if trial_mask.dtype == bool:
            if trial_mask.shape[0] != n_trials:
                raise ValueError(
                    f"trial_mask length {trial_mask.shape[0]} does not match "
                    f"n_trials={n_trials}"
                )
            selected = activity_3d[trial_mask, :, :]
        else:
            idx = trial_mask.astype(int)
            if np.any(idx < 0) or np.any(idx >= n_trials):
                raise ValueError("trial_mask indices out of range for activity_3d")
            selected = activity_3d[idx, :, :]

        if selected.shape[0] == 0:
            raise ValueError("No trials remain after applying trial_mask")
        return selected

    def _normalize_conditions(
        self,
        conditions: np.ndarray,
        n_trials: Optional[int] = None,
        sound_row: int = 0,
        stim_row: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (sound_side, photostim, conditions_as_3xN).

        Accepts either (3, n_trials) or (n_trials, 3).
        """
        conditions = np.asarray(conditions)
        if conditions.ndim != 2:
            raise ValueError(
                f"conditions must be 2D with shape (3, n_trials) or (n_trials, 3); "
                f"got {conditions.shape}"
            )

        if conditions.shape[0] == 3:
            cond_3xN = conditions
        elif conditions.shape[1] == 3:
            cond_3xN = conditions.T
        else:
            raise ValueError(
                f"conditions must have a size-3 axis; got shape {conditions.shape}"
            )

        if n_trials is not None and cond_3xN.shape[1] != n_trials:
            raise ValueError(
                f"conditions trial count {cond_3xN.shape[1]} does not match "
                f"n_trials={n_trials}"
            )
        if sound_row not in (0, 1, 2) or stim_row not in (0, 1, 2):
            raise ValueError("sound_row and stim_row must be 0, 1, or 2")

        sound_side = np.asarray(cond_3xN[sound_row, :], dtype=float).ravel()
        photostim = np.asarray(cond_3xN[stim_row, :], dtype=float).ravel()
        return sound_side, photostim, cond_3xN

    def build_trial_mask_from_conditions(
        self,
        conditions: np.ndarray,
        trial_selector: str = "sound_no_stim",
        sound_values: Sequence[Union[int, float]] = (1, 2),
        sound_row: int = 0,
        stim_row: int = 1,
        stim_value: Union[int, float] = 1,
        no_stim_value: Union[int, float] = 0,
        n_trials: Optional[int] = None,
    ) -> np.ndarray:
        """Boolean trial mask from conditions_updated rows.

        Selectors
        ---------
        sound_no_stim :
            sound side in ``sound_values`` and photostim == ``no_stim_value``
        sound_stim :
            sound side in ``sound_values`` and photostim == ``stim_value``
        """
        sound_side, photostim, _ = self._normalize_conditions(
            conditions, n_trials=n_trials, sound_row=sound_row, stim_row=stim_row
        )
        sound_ok = np.isin(sound_side, np.asarray(list(sound_values), dtype=float))
        if trial_selector == "sound_no_stim":
            return sound_ok & (photostim == float(no_stim_value))
        if trial_selector == "sound_stim":
            return sound_ok & (photostim == float(stim_value))
        raise ValueError(
            f"Unknown trial_selector={trial_selector!r}; "
            "expected 'sound_no_stim' or 'sound_stim'"
        )

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
        trial_mask: Optional[np.ndarray] = None,
    ) -> dict:
        """MI (and optional bootstrap p) for one aligned tensor.

        Parameters
        ----------
        trial_mask : optional
            If provided, subset trials first. If None, existing behavior is preserved.
        """
        activity_3d = self._apply_trial_mask(activity_3d, trial_mask=trial_mask)
        pre_avg, post_avg = self.window_trial_averages(activity_3d)
        mi = self.compute_mod_index_prepost(post_avg, pre_avg)

        out = {
            "mi": mi,
            "pre_avg": pre_avg,
            "post_avg": post_avg,
            "p": None,
            "sig": None,
            "n_trials": int(activity_3d.shape[0]),
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
        conditions_dict: Optional[Dict[Hashable, Dict[Hashable, np.ndarray]]] = None,
        trial_selector: Optional[str] = None,
        trial_mask_dict: Optional[Dict[Hashable, Dict[Hashable, np.ndarray]]] = None,
        sound_values: Sequence[Union[int, float]] = (1, 2),
        sound_row: int = 0,
        stim_row: int = 1,
        stim_value: Union[int, float] = 1,
        no_stim_value: Union[int, float] = 0,
    ) -> dict:
        """Compute MI per fold, then average MI across folds.

        Per-fold p-values are optional and are *not* averaged. Prefer
        ``compute_pooled_bootstrap_across_folds`` for dataset-level significance.

        Optional condition filtering
        ---------------------------
        ``trial_selector`` (e.g. ``\"sound_no_stim\"``) builds a per-fold trial mask
        from ``conditions_dict``. Explicit ``trial_mask_dict[key][fold]`` overrides
        the selector when provided. With neither, existing all-trial behavior is
        unchanged.
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
                trial_mask = None
                if trial_mask_dict is not None and key in trial_mask_dict:
                    trial_mask = trial_mask_dict[key].get(fold)
                elif trial_selector is not None:
                    if conditions_dict is None or key not in conditions_dict:
                        raise ValueError(
                            f"trial_selector={trial_selector!r} requires "
                            f"conditions_dict with key {key!r}"
                        )
                    if fold not in conditions_dict[key]:
                        raise ValueError(
                            f"conditions_dict[{key!r}] missing fold {fold!r}"
                        )
                    trial_mask = self.build_trial_mask_from_conditions(
                        conditions_dict[key][fold],
                        trial_selector=trial_selector,
                        sound_values=sound_values,
                        sound_row=sound_row,
                        stim_row=stim_row,
                        stim_value=stim_value,
                        no_stim_value=no_stim_value,
                        n_trials=np.asarray(folds[fold]).shape[0],
                    )

                result = self.compute_mi_for_aligned_tensor(
                    folds[fold],
                    compute_p=compute_p,
                    n_shuffles=n_shuffles,
                    trial_mask=trial_mask,
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
        trial_mask: Optional[np.ndarray] = None,
    ) -> dict:
        """Post−pre delta for one aligned tensor (parallel to MI tensor method)."""
        activity_3d = self._apply_trial_mask(activity_3d, trial_mask=trial_mask)
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
            "n_trials": int(activity_3d.shape[0]),
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
        conditions_dict: Optional[Dict[Hashable, Dict[Hashable, np.ndarray]]] = None,
        trial_selector: Optional[str] = None,
        trial_mask_dict: Optional[Dict[Hashable, Dict[Hashable, np.ndarray]]] = None,
        sound_values: Sequence[Union[int, float]] = (1, 2),
        sound_row: int = 0,
        stim_row: int = 1,
        stim_value: Union[int, float] = 1,
        no_stim_value: Union[int, float] = 0,
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
                trial_mask = None
                if trial_mask_dict is not None and key in trial_mask_dict:
                    trial_mask = trial_mask_dict[key].get(fold)
                elif trial_selector is not None:
                    if conditions_dict is None or key not in conditions_dict:
                        raise ValueError(
                            f"trial_selector={trial_selector!r} requires "
                            f"conditions_dict with key {key!r}"
                        )
                    if fold not in conditions_dict[key]:
                        raise ValueError(
                            f"conditions_dict[{key!r}] missing fold {fold!r}"
                        )
                    trial_mask = self.build_trial_mask_from_conditions(
                        conditions_dict[key][fold],
                        trial_selector=trial_selector,
                        sound_values=sound_values,
                        sound_row=sound_row,
                        stim_row=stim_row,
                        stim_value=stim_value,
                        no_stim_value=no_stim_value,
                        n_trials=np.asarray(folds[fold]).shape[0],
                    )

                result = self.compute_delta_for_aligned_tensor(
                    folds[fold],
                    compute_p=compute_p,
                    n_shuffles=n_shuffles,
                    trial_mask=trial_mask,
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
        conditions_dict: Optional[Dict[Hashable, Dict[Hashable, np.ndarray]]] = None,
        trial_selector: Optional[str] = None,
        trial_mask_dict: Optional[Dict[Hashable, Dict[Hashable, np.ndarray]]] = None,
        sound_values: Sequence[Union[int, float]] = (1, 2),
        sound_row: int = 0,
        stim_row: int = 1,
        stim_value: Union[int, float] = 1,
        no_stim_value: Union[int, float] = 0,
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
            aligned_dict,
            compute_p=compute_p,
            n_shuffles=n_shuffles,
            conditions_dict=conditions_dict,
            trial_selector=trial_selector,
            trial_mask_dict=trial_mask_dict,
            sound_values=sound_values,
            sound_row=sound_row,
            stim_row=stim_row,
            stim_value=stim_value,
            no_stim_value=no_stim_value,
        )
        if average_folds:
            return full["delta_mean_by_key"]
        return full["delta_by_key_fold"]

    def compute_control_mod_index(
        self,
        post_sound_stim_avg: np.ndarray,
        post_sound_only_avg: np.ndarray,
        eps: Optional[float] = None,
    ) -> np.ndarray:
        """Neuron-wise control/opto MI from post-period trial averages.

        Matches Bassirunyan ``compute_mod_index_ctrl.m``:

            ctrl_MI = (mean_stim - mean_ctrl) / (mean_stim + mean_ctrl)

        where both means are over the post window only (not pre vs post).
        Near-zero denominators are zeroed using ``eps`` (default ``self.eps``),
        matching the existing MI convention. MATLAB also uses epsilon=1e-5 and
        zeros values with |MI|>1; we apply the |MI|>1 cleanup for parity.
        """
        post_sound_stim_avg = np.asarray(post_sound_stim_avg, dtype=float)
        post_sound_only_avg = np.asarray(post_sound_only_avg, dtype=float)
        if post_sound_stim_avg.ndim != 2 or post_sound_only_avg.ndim != 2:
            raise ValueError(
                "post_sound_stim_avg and post_sound_only_avg must be "
                "2D (n_trials, n_neurons)"
            )
        if post_sound_stim_avg.shape[1] != post_sound_only_avg.shape[1]:
            raise ValueError("stim and control arrays must share n_neurons")

        eps_val = self.eps if eps is None else float(eps)
        stim_val = np.nanmean(post_sound_stim_avg, axis=0)
        ctrl_val = np.nanmean(post_sound_only_avg, axis=0)
        denom = stim_val + ctrl_val

        ctrl_mi = np.zeros(stim_val.shape[0], dtype=float)
        valid = np.isfinite(denom) & (np.abs(denom) >= eps_val)
        ctrl_mi[valid] = (stim_val[valid] - ctrl_val[valid]) / denom[valid]
        # Bassirunyan cleanup: extreme values with a zero mean -> 0; |MI|>1 -> 0
        extreme = (
            ((np.isclose(ctrl_mi, 1.0) | np.isclose(ctrl_mi, -1.0))
             & ((stim_val == 0) | (ctrl_val == 0)))
            | (np.abs(ctrl_mi) > 1.0)
        )
        ctrl_mi[extreme] = 0.0
        return ctrl_mi

    def compute_control_mod_index_for_aligned_tensor(
        self,
        activity_3d: np.ndarray,
        conditions: np.ndarray,
        sound_values: Sequence[Union[int, float]] = (1, 2),
        stim_row: int = 1,
        sound_row: int = 0,
        stim_value: Union[int, float] = 1,
        no_stim_value: Union[int, float] = 0,
        eps: Optional[float] = None,
        post_frames=None,
    ) -> dict:
        """Control MI: sound+stim vs sound-alone in the post period only.

        Trial definitions
        -----------------
        sound_only : sound side in ``sound_values`` and photostim == ``no_stim_value``
        sound_stim : sound side in ``sound_values`` and photostim == ``stim_value``

        Formula (Bassirunyan ``compute_mod_index_ctrl``)::

            ctrl_mod_index = (post_sound_stim - post_sound_only)
                             / (post_sound_stim + post_sound_only)
        """
        activity_3d = np.asarray(activity_3d, dtype=float)
        if activity_3d.ndim != 3:
            raise ValueError("activity_3d must be (n_trials, n_neurons, n_frames)")

        n_trials = activity_3d.shape[0]
        sound_side, photostim, _ = self._normalize_conditions(
            conditions, n_trials=n_trials, sound_row=sound_row, stim_row=stim_row
        )
        sound_ok = np.isin(sound_side, np.asarray(list(sound_values), dtype=float))
        sound_only_mask = sound_ok & (photostim == float(no_stim_value))
        sound_stim_mask = sound_ok & (photostim == float(stim_value))

        n_sound_only = int(np.sum(sound_only_mask))
        n_sound_stim = int(np.sum(sound_stim_mask))
        if n_sound_only == 0:
            raise ValueError("No sound-only trials (sound + photostim==0) found")
        if n_sound_stim == 0:
            raise ValueError("No sound+stim trials (sound + photostim==1) found")

        post_idx = self.post_frames if post_frames is None else np.asarray(post_frames, dtype=int)
        n_frames = activity_3d.shape[2]
        if np.any(post_idx < 0) or np.any(post_idx >= n_frames):
            raise ValueError(f"post_frames out of range for n_frames={n_frames}")

        # Trial x neuron post-window averages for each condition
        post_sound_only_avg = np.nanmean(
            activity_3d[sound_only_mask][:, :, post_idx], axis=2
        )
        post_sound_stim_avg = np.nanmean(
            activity_3d[sound_stim_mask][:, :, post_idx], axis=2
        )
        post_sound_only = np.nanmean(post_sound_only_avg, axis=0)
        post_sound_stim = np.nanmean(post_sound_stim_avg, axis=0)
        ctrl_mod_index = self.compute_control_mod_index(
            post_sound_stim_avg, post_sound_only_avg, eps=eps
        )

        return {
            "ctrl_mod_index": ctrl_mod_index,
            "post_sound_only_mean": post_sound_only,
            "post_sound_stim_mean": post_sound_stim,
            "n_sound_only_trials": n_sound_only,
            "n_sound_stim_trials": n_sound_stim,
        }

    def compute_control_mod_index_across_folds(
        self,
        aligned_dict: Dict[Hashable, Dict[Hashable, np.ndarray]],
        conditions_dict: Dict[Hashable, Dict[Hashable, np.ndarray]],
        average_folds: bool = True,
        sound_values: Sequence[Union[int, float]] = (1, 2),
        stim_row: int = 1,
        sound_row: int = 0,
        stim_value: Union[int, float] = 1,
        no_stim_value: Union[int, float] = 0,
        eps: Optional[float] = None,
    ):
        """Compute control MI per fold; optionally average across folds.

        Does not average p-values (none are computed here).
        """
        ctrl_by_key_fold = {}
        ctrl_mean_by_key = {}

        for key, folds in aligned_dict.items():
            if key not in conditions_dict:
                raise ValueError(f"conditions_dict missing key {key!r}")
            ctrl_by_key_fold[key] = {}
            fold_vals = []

            for fold in sorted(folds.keys(), key=lambda x: (str(type(x)), x)):
                if fold not in conditions_dict[key]:
                    raise ValueError(
                        f"conditions_dict[{key!r}] missing fold {fold!r}"
                    )
                result = self.compute_control_mod_index_for_aligned_tensor(
                    folds[fold],
                    conditions_dict[key][fold],
                    sound_values=sound_values,
                    stim_row=stim_row,
                    sound_row=sound_row,
                    stim_value=stim_value,
                    no_stim_value=no_stim_value,
                    eps=eps,
                )
                ctrl_by_key_fold[key][fold] = result["ctrl_mod_index"]
                fold_vals.append(result["ctrl_mod_index"])

            stacked = np.stack(fold_vals, axis=0)
            ctrl_mean_by_key[key] = np.nanmean(stacked, axis=0)

        if average_folds:
            return ctrl_mean_by_key
        return ctrl_by_key_fold

    def compute_control_mod_index_true_vs_pred(
        self,
        aligned_true: Dict[Hashable, Dict[Hashable, np.ndarray]],
        aligned_pred: Dict[Hashable, Dict[Hashable, np.ndarray]],
        conditions_dict: Dict[Hashable, Dict[Hashable, np.ndarray]],
        average_folds: bool = True,
        sound_values: Sequence[Union[int, float]] = (1, 2),
        stim_row: int = 1,
        sound_row: int = 0,
        stim_value: Union[int, float] = 1,
        no_stim_value: Union[int, float] = 0,
        eps: Optional[float] = None,
    ) -> dict:
        """True vs predicted control MI wrappers."""
        ctrl_mi_true = self.compute_control_mod_index_across_folds(
            aligned_true,
            conditions_dict,
            average_folds=average_folds,
            sound_values=sound_values,
            stim_row=stim_row,
            sound_row=sound_row,
            stim_value=stim_value,
            no_stim_value=no_stim_value,
            eps=eps,
        )
        ctrl_mi_pred = self.compute_control_mod_index_across_folds(
            aligned_pred,
            conditions_dict,
            average_folds=average_folds,
            sound_values=sound_values,
            stim_row=stim_row,
            sound_row=sound_row,
            stim_value=stim_value,
            no_stim_value=no_stim_value,
            eps=eps,
        )
        return {
            "true": ctrl_mi_true,
            "pred": ctrl_mi_pred,
        }

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
