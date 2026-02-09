# Neural Population GLM Analysis

Python tools and notebooks for analyzing generalized linear model (GLM) encoding/decoding results from calcium imaging datasets across multiple cell types and behavioral contexts.

## Project Summary

This repository supports a full post-hoc neural population analysis workflow:

- Designed and implemented GLM comparison analyses to predict and interpret population neural activity from sensory, motor, and task-related variables.
- Built custom Python pipelines for encoding and decoding analysis across multiple cell types and behavioral states.
- Performed statistical analysis of functional coupling, cross-condition comparisons, and population decoding using cross-validation outputs, permutation tests, and nonparametric tests.

In short: this codebase is focused on interpreting fitted model outputs, quantifying what neurons encode, and testing how coupling/decoding structure changes across conditions.

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

## Repository Structure

```text
.
|-- analysis/
|   |-- AnalysisManagerEncoding.py
|   `-- DecoderAnalyzer.py
|-- handlers/
|   |-- DataHandlerEncoding.py
|   `-- DataHandlerDecoding.py
|-- utils/
|   |-- Plotter.py
|   |-- GLMPredictorProcessor.py
|   |-- general_stats.py
|   `-- ...
|-- config/
|   `-- DatasetConfig.py
`-- notebooks/
    |-- run_glm_encoding_analysis.ipynb
    |-- run_glm_decoding_analysis.ipynb
    |-- run_glm_encoding_analysis_all_neurons_coupling.ipynb
    `-- quick_glm_model_comparison.ipynb
```

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

### 3) (Optional) helper_functions dependency

Some notebook/util paths reference a separate `helper_functions` module (from a sibling project). If needed, add that project to `PYTHONPATH` or update `utils/path_utils.py` to your local layout.

## Example Workflow (Encoding)

```python
from handlers.DataHandlerEncoding import DataHandlerEncoding
from analysis.AnalysisManagerEncoding import AnalysisManagerEncoding
from utils.Plotter import Plotter

data_handler = DataHandlerEncoding(data=None)
datasets, keys = data_handler.load_info(info_dir=".../info_dir")
results = data_handler.process_multiple_datasets(
    datasets=datasets,
    model_type="GLM_3nmf_pre",
    results_type="results"
)

plotter = Plotter(data=results, save_results="./results")
analysis = AnalysisManagerEncoding(data=results, plotter=plotter)
```

## Current Scope and Limitations

- Notebook-first workflow (not yet packaged as an installable library).
- Path defaults are currently tailored to a specific lab filesystem and should be edited locally.
- This repository focuses on analysis of fitted outputs; model fitting/training is assumed to be done upstream.

## Potential Next Improvements

- Add a formal `requirements.txt` or `pyproject.toml`.
- Replace hardcoded filesystem paths with config files or CLI arguments.
- Add small synthetic-data examples and tests for key analysis functions.
- Add a reproducible end-to-end demo notebook with mock data.
