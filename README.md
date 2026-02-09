# Neural Population GLM Analysis

Python tools and notebooks for analyzing generalized linear model (GLM) encoding/decoding results from calcium imaging datasets across multiple cell types and behavioral contexts.

## Project Summary

This codebase is focused on interpreting fitted model outputs, quantifying what neurons encode, and testing how coupling/decoding structure changes across conditions.

## What This Repository Covers

### 1) GLM encoding analysis

- Loads per-neuron GLM fit outputs (pickled fold-wise results).
- Compares model families (for example: behavior-only, full coupling, and cell-type ablations).
- Computes deviance explained summaries and coupling indices.
- Aggregates results across datasets and cell types (Pyr, SOM, PV).

Main code:
- `handlers/DataHandlerEncoding.py`
- `analysis/AnalysisManagerEncoding.py`

### 2) Decoding analysis

- Loads decoder `.mat` outputs for variables like `sound_category`, `choice`, `outcome`, and shuffled controls.
- Computes single-cell and population decoding summaries across folds/shuffles.
- Identifies significant neurons with threshold/shuffle-based methods.
- Supports formatting and exporting results for downstream analysis.

Main code:
- `handlers/DataHandlerDecoding.py`
- `analysis/DecoderAnalyzer.py`
- `config/DatasetConfig.py`

### 3) Predictor alignment and condition-level analyses

- Aligns behavioral and coupling predictors to task events.
- Splits by trial conditions (correctness, turn side, stim/control, inferred sound side).
- Matches coupling factors across datasets and aggregates condition-level traces.

Main code:
- `utils/GLMPredictorProcessor.py`

### 4) Plotting and statistics

- Publication-style plotting utilities for:
  - model comparison scatter/box/bar/CDF plots
  - decoding heatmaps and time courses
  - coupling quadrant and overlap summaries
  - significance overlays and summary panels
- Statistical helper methods for permutation tests, Wilcoxon/Mann-Whitney/KS/Kruskal tests, Bonferroni correction, and bootstrap summaries.

Main code:
- `utils/Plotter.py`
- `utils/general_stats.py`

## Data Expectations

This repo expects precomputed GLM/decoder outputs on an external data store (lab-style paths), not raw data inside this repository.

Typical expected layout:

```text
<server>/Connie/ProcessedData/<animalID>/<date>/<GLM_model_type>/
  |- results/
  |   |- poss_model_0_data_cluster_*.pkl
  |   |- poss_model_1_data_cluster_*.pkl
  |   `- ...
  `- decoding/
      |- <split>/decoder_results_regular_<variable>.mat
      `- <split>/decoder_results_shuffled_<variable>.mat
```

Several workflows also expect an `info.mat` file listing datasets and server IDs.

## Quick Start

### 1) Create environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy scipy pandas scikit-learn matplotlib seaborn h5py matplotlib-venn jupyter
```

### 2) Open notebooks

Start with:

- `notebooks/run_glm_encoding_analysis.ipynb`
- `notebooks/run_glm_decoding_analysis.ipynb`

Update path variables in the first cells (project root and data server paths) to match your machine.

