# GLM Analysis

Python tools for loading, validating, aggregating, and analyzing precomputed GLM encoding and decoding outputs from a multi-session calcium-imaging study.

This repository is research analysis and data-coordination code. It does not include the raw or processed study datasets because of their size and institutional storage requirements. Full end-to-end execution therefore requires access to the lab data store.

## What this repository demonstrates

- Multi-dataset cohort configuration and order validation
- Harmonization of MATLAB/HDF5 decoder outputs into Python structures
- Explicit tracking of permitted missing decoder files
- Structured recording of unexpected dataset-processing failures
- Aggregation across repeated splits and shuffled controls
- Separation of data loading, cohort rules, analysis, and plotting utilities

This is research analysis code with targeted ingestion and validation practices, not a production pipeline.

## Start here

The representative decoding workflow is:

`notebooks/run_glm_decoding_analysis.ipynb`

Supporting modules include:

- `config/DatasetConfig.py` — cohort inventory and variable availability
- `handlers/DataHandlerDecoding.py` — decoder loading, missing-file recording, and multi-dataset processing
- `analysis/DecoderAnalyzer.py` — decoding summaries and significance analyses
- `utils/Plotter.py` — visualization helpers

Encoding workflows are available in:

- `notebooks/run_glm_encoding_analysis.ipynb`
- `handlers/DataHandlerEncoding.py`

The decoding notebook is the clearest entrypoint for reviewing the repository's data-loading and validation logic.

## Decoding data flow

1. `DatasetConfig.load_from_info(...)` loads the fixed paper cohort from `info.mat`.
2. Loaded dataset keys are checked against the expected cohort order.
3. Sessions are filtered based on availability of decoded variables.
4. `process_multiple_datasets(...)` builds session-specific decoder paths.
5. Split-wise `.mat` files are loaded and harmonized.
6. Missing decoder outputs are recorded while the workflow continues for known gaps.
7. Unexpected dataset-processing exceptions are stored in a separate error ledger.
8. Results are aggregated across splits and shuffled controls for downstream analysis.

## Missing files and processing errors

After a multi-dataset run, inspect:

```python
data_handler.missing_decoder_files
data_handler.processing_errors
```

`missing_decoder_files` records decoder files that were not found and were handled using the repository's permitted-gap behavior.

`processing_errors` records unexpected exceptions that occurred while processing a dataset.

The fixed cohort order is validated before positional variable-availability rules are applied.

## Expected external data layout

The research data are not included. The decoding workflow expects a structure similar to:

```text
<server>/Connie/ProcessedData/<animalID>/<date>/<GLM_model_type>/
    decoding/
        <split>/decoder_results_regular_<variable>.mat
        <split>/decoder_results_shuffled_<variable>.mat
```

An `info.mat` file listing datasets and server identifiers is also required.

## Installation

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install requirements:

```bash
pip install -r requirements.txt
```

## Main dependencies

- NumPy
- SciPy
- pandas
- scikit-learn
- matplotlib
- seaborn
- h5py
- matplotlib-venn

Some archived notebooks may use additional packages.

## Configuring local paths

Before running:

1. Ensure the institutional data mounts referenced by `info.mat` are available.
2. Set the notebook working directory to the local repository.
3. Set `info_dir` to the folder containing `info.mat`.
4. Set output paths to local directories you control.
5. Update any machine-specific helper paths if using notebook plotting utilities.

Without the institutional data store, the code structure, loading logic, validation rules, and analysis utilities can still be reviewed, but the full workflow cannot be reproduced.

## Example usage

```python
from handlers.DataHandlerDecoding import DataHandlerDecoding
from config.DatasetConfig import DatasetConfig

decoded_variables = {
    "sound_category",
    "shuffled/sound_category",
    "choice",
    "shuffled/choice",
}

data_handler = DataHandlerDecoding(
    decoded_variables=decoded_variables
)

dataset_config = DatasetConfig()

info_dir = "PATH/TO/INFO_DIR"

datasets, mouse_dates_keys = (
    dataset_config.load_from_info(
        info_dir,
        data_handler,
    )
)

datasets, filtered_keys = (
    dataset_config.get_datasets_with_variables(
        decoded_variables,
        require_all=False,
    )
)

model_type = "GLM_3nmf_pre"

(
    mean_results,
    mean_results_all,
    cat_results,
    celltype_info,
) = data_handler.process_multiple_datasets(
    datasets,
    model_type,
    single_balanced=False,
)

print(data_handler.processing_errors)
print(data_handler.missing_decoder_files)
```

## Repository structure

```text
.
├── analysis/
├── config/
├── handlers/
├── notebooks/
│   └── old_notebooks/
├── original/
├── utils/
├── requirements.txt
└── README.md
```

- `handlers/` — data loading and multi-dataset orchestration
- `config/` — fixed cohort configuration and variable availability
- `analysis/` — downstream analysis helpers
- `utils/` — plotting, statistics, predictor handling, and path utilities
- `notebooks/` — representative analysis notebooks
- `notebooks/old_notebooks/` — archived exploratory notebooks
- `original/` — earlier archived implementations

## Reproducibility note

Results depend on external precomputed GLM and decoder files, fixed cohort definitions, and institutional path conventions.

This repository is best evaluated for how it handles cohort validation, file loading, permitted missingness, unexpected failures, aggregation, and scientific analysis rather than as a fully self-contained reproducible package.


