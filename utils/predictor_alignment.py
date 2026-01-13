"""
Utilities to align *frame-wise* predictors to trial events and plot trial-averaged traces.

Primary use-case:
- You have a concatenated behavior matrix (predictors x frames) and want to make
  average traces aligned to the same events used by `find_align_info`
  (sound repeats, turn, reward).
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


# Behavior-matrix column layout (0-indexed) from user description
BEHAV_COLS_DEFAULT: Dict[str, int] = {
    "vel_y": 0,
    "vel_x": 1,
    "view_angle": 2,
    "left_turn": 3,
    "right_turn": 4,
    "reward": 5,
    "left_sound_rep1": 6,
    "right_sound_rep1": 7,
    "left_sound_rep2": 8,
    "right_sound_rep2": 9,
    "left_sound_rep3": 10,
    "right_sound_rep3": 11,
    "photostim": 12,
}

BEHAV_NAMES_DEFAULT: List[str] = [
    "vel_y",
    "vel_x",
    "view_angle",
    "left_turn",
    "right_turn",
    "reward",
    "left_sound_rep1",
    "right_sound_rep1",
    "left_sound_rep2",
    "right_sound_rep2",
    "left_sound_rep3",
    "right_sound_rep3",
    "photostim",
]


# Windows copied from `find_align_info` (left_pad, right_pad)
EVENT_WINDOWS_DEFAULT: Dict[str, Tuple[int, int]] = {
    "sound1": (6, 30),
    "sound2": (1, 30),
    "sound3": (1, 30),
    "turn": (30, 12),
    "reward": (1, 23),
}


def _first_onset(x_bool_1d: np.ndarray) -> Optional[int]:
    idx = np.where(x_bool_1d)[0]
    return int(idx[0]) if idx.size else None


def trial_segments_from_condition_array(
    condition_array_trials: np.ndarray,
    n_total_frames: int,
    trial_start_col: int = 4,
) -> List[Tuple[int, int]]:
    """
    Build (start,end) trial segments in global-frame coordinates using the
    `condition_array_trials[:, trial_start_col]` column (MATLAB 5th column => Python index 4).
    """
    starts = np.asarray(condition_array_trials[:, trial_start_col]).ravel()
    starts = starts[~np.isnan(starts)].astype(int)
    starts = np.unique(starts)
    starts = starts[(starts >= 0) & (starts < n_total_frames)]
    starts.sort()

    segs: List[Tuple[int, int]] = []
    for i, s in enumerate(starts):
        e = (starts[i + 1] - 1) if (i < len(starts) - 1) else (n_total_frames - 1)
        if e >= s:
            segs.append((int(s), int(e)))
    return segs


def compute_event_onsets_from_behav_matrix(
    behav_matrix: np.ndarray,
    trial_segments: Sequence[Tuple[int, int]],
    behav_cols: Mapping[str, int] = BEHAV_COLS_DEFAULT,
) -> Dict[str, np.ndarray]:
    """
    Compute per-trial event onsets (0-based indices, relative within-trial).

    Events:
    - sound1/sound2/sound3: onset of each repeat, combining left+right sound columns
    - turn: first onset of left_turn OR right_turn
    - reward: first onset of reward

    Returns:
    - dict[event_name] = float array shape (n_trials,), where missing onsets are NaN.
    """
    onsets: Dict[str, List[Optional[int]]] = {k: [] for k in ["sound1", "sound2", "sound3", "turn", "reward"]}

    for (s, e) in trial_segments:
        seg = behav_matrix[:, s : e + 1]

        for rep in (1, 2, 3):
            l = behav_cols[f"left_sound_rep{rep}"]
            r = behav_cols[f"right_sound_rep{rep}"]
            onset = _first_onset((seg[l, :] > 0) | (seg[r, :] > 0))
            onsets[f"sound{rep}"].append(onset)

        onset_turn = _first_onset((seg[behav_cols["left_turn"], :] > 0) | (seg[behav_cols["right_turn"], :] > 0))
        onsets["turn"].append(onset_turn)

        onset_reward = _first_onset(seg[behav_cols["reward"], :] > 0)
        onsets["reward"].append(onset_reward)

    out: Dict[str, np.ndarray] = {}
    for k, v in onsets.items():
        out[k] = np.array([np.nan if x is None else x for x in v], dtype=float)
    return out


def align_matrix_to_trial_events(
    X: np.ndarray,
    trial_segments: Sequence[Tuple[int, int]],
    event_onsets: np.ndarray,
    left_pad: int,
    right_pad: int,
) -> np.ndarray:
    """
    Align a frame-wise matrix to per-trial event onsets.

    Parameters:
    - X: (features x frames_total)
    - trial_segments: list of (start,end) in global frames
    - event_onsets: float array (n_trials,), 0-based within-trial; NaN for missing trials
    - left_pad/right_pad: alignment window sizes

    Returns:
    - aligned: (trials x features x window_len) with NaNs where trials are missing/out-of-bounds
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (features x frames). Got {X.shape}")

    n_trials = len(trial_segments)
    win_len = left_pad + right_pad + 1
    aligned = np.full((n_trials, X.shape[0], win_len), np.nan, dtype=float)

    for t, (s, e) in enumerate(trial_segments):
        onset = event_onsets[t]
        if np.isnan(onset):
            continue

        onset_i = int(onset)
        rel_start = onset_i - left_pad
        rel_end = onset_i + right_pad
        trial_len = e - s + 1
        if rel_start < 0 or rel_end >= trial_len:
            continue

        aligned[t, :, :] = X[:, (s + rel_start) : (s + rel_end + 1)]

    return aligned


def nansem(x: np.ndarray, axis: int = 0) -> np.ndarray:
    x = np.asarray(x)
    n = np.sum(~np.isnan(x), axis=axis)
    sd = np.nanstd(x, axis=axis)
    return sd / np.sqrt(np.maximum(n, 1))


def plot_aligned_means_grid(
    aligned: np.ndarray,
    names: Sequence[str],
    title: str,
    left_pad: int,
    ncols: int = 4,
    ylim: Optional[Tuple[float, float]] = None,
):
    """
    Plot mean±SEM per feature in a grid.
    Requires matplotlib (import inside to avoid hard dependency for pure compute usage).
    """
    import matplotlib.pyplot as plt

    mean = np.nanmean(aligned, axis=0)
    sem = nansem(aligned, axis=0)
    t = np.arange(mean.shape[1]) - left_pad

    n_feats = mean.shape[0]
    nrows = math.ceil(n_feats / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.2 * nrows), sharex=True)
    axs = np.array(axs).reshape(-1)

    for i in range(n_feats):
        ax = axs[i]
        ax.plot(t, mean[i], color="black", linewidth=1)
        ax.fill_between(t, mean[i] - sem[i], mean[i] + sem[i], color="black", alpha=0.2, linewidth=0)
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(str(names[i]), fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ylim is not None:
            ax.set_ylim(ylim)

    for j in range(n_feats, len(axs)):
        axs[j].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_average_predictors_through_trial(
    behav_matrix: np.ndarray,
    condition_array_trials: np.ndarray,
    X_to_plot: Optional[np.ndarray] = None,
    X_names: Optional[Sequence[str]] = None,
    event_windows: Mapping[str, Tuple[int, int]] = EVENT_WINDOWS_DEFAULT,
    behav_cols: Mapping[str, int] = BEHAV_COLS_DEFAULT,
):
    """
    Convenience wrapper:
    - builds trial segments from condition_array_trials
    - extracts event onsets from behav_matrix
    - aligns X_to_plot to each event in event_windows
    - plots trial-averaged mean±SEM grids
    """
    if X_to_plot is None:
        X_to_plot = behav_matrix
    if X_names is None:
        X_names = BEHAV_NAMES_DEFAULT if X_to_plot is behav_matrix else [f"feat{i}" for i in range(X_to_plot.shape[0])]

    trial_segments = trial_segments_from_condition_array(condition_array_trials, n_total_frames=behav_matrix.shape[1])
    event_onsets = compute_event_onsets_from_behav_matrix(behav_matrix, trial_segments, behav_cols=behav_cols)

    figs = {}
    for event, (lp, rp) in event_windows.items():
        aligned = align_matrix_to_trial_events(X_to_plot, trial_segments, event_onsets[event], lp, rp)
        n_valid = int(np.sum(~np.isnan(event_onsets[event])))
        figs[event] = plot_aligned_means_grid(
            aligned,
            names=X_names,
            title=f"{event} aligned (n={n_valid} trials)",
            left_pad=lp,
        )
    return figs

