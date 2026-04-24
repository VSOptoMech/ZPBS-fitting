# XYZ Surface Fitting Toolkit

Standalone batch runner and GUI for notebook-style XYZ point clouds.

## What This Repository Does

This repository provides a curated standalone runtime snapshot for fitting spherical and Zernike models to XYZ point-cloud data.

Main entry points:

- `zpbs-batch-fit`: batch-process a folder of `.xyz` files and write summary outputs
- `zpbs-batch-gui`: launch the GUI for running batches, single-file inspection, and replaying saved summaries
- `src/zpbs/azp_csv_pipeline.py`: core numerical helper layer used by the batch runner and GUI

Runtime package layout:

- `src/zpbs/`: maintained package implementation used by the package entrypoints

## Setup

```bash
uv sync
```

Recommended usage:

- use `uv run` for the CLI and GUI
- keep the project environment consistent with `uv sync`
- prefer running from the repository root

## Input Expectations

The batch runner reads plain-text XYZ point clouds with three numeric columns.

Accepted formats:

- comma-delimited
- tab-delimited
- semicolon-delimited
- whitespace-delimited

An optional `X,Y,Z` header row is allowed.

Example:

```text
X,Y,Z
0.0,0.0,-12.3
10.0,0.0,-12.1
0.0,10.0,-12.0
```

The batch-discovery logic is designed around notebook-style filenames that encode surface identity and force level. Only these surface families are processed:

- `AA`
- `AP`
- `PA`
- `PP`

Files outside those families are ignored by the batch runner.

## Batch Runner

```bash
uv run zpbs-batch-fit /path/to/xyz_dir --glob '*_FVS_*.xyz'
```

Common useful options:

- `--recursive`: search subdirectories
- `--output-dir`: choose where run folders are written
- `--run-name`: force a stable run folder name
- `--qa-report`: generate an HTML QA gallery
- `--fail-fast`: stop on the first file error
- `--h5-path`: append results to a shared HDF5 file

Current maintained defaults:

- sphere-fit mode: `center_weighted`
- center weight: `0.5`
- normalization mode: `per-file`
- radius rounding: on
- Zernike coefficient rounding: `6` significant digits
- Zernike fitting method: maintained `lstsq` only
- GUI HDF5 export: off by default because HDF5 output includes raw point-cloud data and absolute source paths

Default opt-out flags:

```bash
--no-round-radii-um
--no-round-zernike-coeffs
```

## Batch Outputs

Each batch run writes a dedicated folder under `batch_outputs/`.

Typical contents:

- `batch_summary_YYYY-MM-DD.xlsx`
- `run_manifest.json`
- `batch_failures.csv` when failures occur
- `coefficients/` per-surface coefficient CSVs
- `coefficients/ZPs_batch_report.csv`
- optional `qa/` HTML output
- optional HDF5 output if requested; treat it as raw-data-bearing and not public-shareable by default

Per-surface coefficient CSVs include:

- `Base sphere radius (mm)`
- `Vertex (mm)` as the final near-center ZPBS fitted axial location
- `Vertex residual (mm)` as the near-center residual after the full ZPBS-to-data reconstruction
- `Norm. Radius (mm)`
- Zernike terms `Z1..Z45`

## GUI

```bash
uv run zpbs-batch-gui
```

The public GUI supports three workflows:

- `Run Batch`
- `Single File`
- `Inspect Summary`

Public-release constraint:

- private subset-inspection tooling is not included in this standalone public build

### Run Batch

The GUI builds and launches the same batch command you can run from the CLI.

The command preview reflects the actual CLI arguments that will be run.

### Single File

Use `Single File` to inspect one maintained `AA/AP/PA/PP` `.xyz` input live without running a full batch.

The tab supports:

- file browsing and adjacent-file navigation
- live sphere and Zernike fitting with maintained defaults
- overview plots, coefficient display, and diagnostics export

### Inspect Summary

Load a saved `batch_summary_YYYY-MM-DD.xlsx` workbook and replay one processed row.

The viewer provides:

- surface-family selection
- force-level selection
- initial vs deformed selection
- radial profile with measured, sphere, and Zernike curves
- measured surface map
- `Zernike Residual vs Radius`
- `Sphere Fit Residual vs Radius`

If a completed run folder is copied to another machine, coefficient paths under the original run folder are remapped automatically to the folder containing the loaded workbook. Raw `.xyz` files can be remapped by setting `Local XYZ Root`.

## Typical Workflow

1. Place your `.xyz` files in one folder.
2. Run a batch from the CLI or GUI.
3. Inspect `batch_summary_YYYY-MM-DD.xlsx`.
4. Review `coefficients/` and `ZPs_batch_report.csv`.
5. Reopen the saved summary workbook in the GUI to replay and inspect individual fits.
6. Use `Single File` for spot checks without running a full batch.

## Troubleshooting

- If `python3` is missing required packages, use `uv run` instead of system Python.
- If a copied run folder does not replay correctly in the GUI, verify `Local XYZ Root` points at the raw `.xyz` directory.
- If no files are processed, verify your filenames map into the supported `AA/AP/PA/PP` families.
- If some files fail during batch processing, inspect `batch_failures.csv`.
- If you changed dependencies, run `uv sync` again.
- If results differ between runs, compare `roc_mode`, `sphere_fit_mode`, `center_weight`, `normalization_mode`, `n_modes`, and rounding settings before comparing fit metrics.

## Supported Entry Points

- `src/zpbs/`
