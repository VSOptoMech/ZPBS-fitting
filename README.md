# XYZ Surface Fitting Toolkit

Standalone batch runner and GUI for notebook-style XYZ point clouds.

## What This Repository Does

This repository provides a practical workflow for fitting spherical and Zernike models to XYZ point-cloud data.

Main entry points:

- `batch_fit_xyz.py`: batch-process a folder of `.xyz` files and write summary outputs
- `batch_fit_xyz_gui.py`: launch the GUI for running batches and replaying saved summaries
- `azp_csv_pipeline.py`: core numerical helper layer used by the batch runner and GUI

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

The current batch-discovery logic is designed around notebook-style filenames that encode surface identity and force level. Only these surface families are processed:

- `AA`
- `AP`
- `PA`
- `PP`

Files outside those families are ignored by the batch runner.

## Batch Runner

```bash
uv run python batch_fit_xyz.py /path/to/xyz_dir --glob '*_FVS_*.xyz'
```

Common useful options:

- `--recursive`: search subdirectories
- `--output-dir`: choose where run folders are written
- `--run-name`: force a stable run folder name
- `--qa-report`: generate an HTML QA gallery
- `--fail-fast`: stop on the first file error
- `--h5-path`: append results to a shared HDF5 file

## Current Batch Behavior

- identical `*I` initial-state files are collapsed to one synthetic `F0.0mN` row per surface family
- collapsed initial rows retain their original `AAI/API/PAI/PPI` surface token
- `*D` deformed files remain one processed row per force level
- radius rounding is enabled by default
- Zernike coefficient rounding to `6` significant digits is enabled by default

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
- optional HDF5 output if requested

Per-surface coefficient CSVs include:

- `Base sphere radius (mm)`
- `Vertex (mm)`
- `Vertex residual (mm)`
- `Norm. Radius (mm)`
- Zernike terms `Z1..Z45`

## Summary Workbook Fields

The batch summary workbook records both identity fields and fit results.

Important identity/configuration fields:

- `run_name`: run folder / run identifier
- `roc_mode`: sphere-radius policy
- `normalization_mode`: Zernike normalization-radius policy
- `round_radii_um`: whether radius rounding was enabled
- `zernike_coeff_sigfigs`: coefficient rounding precision
- `source_file`: raw XYZ file that was actually processed
- `source_force_id`: original force token from the raw file
- `force_id`: processed force token used for reporting
- `surf_id`: surface family such as `AA`, `AP`, `PA`, or `PP`
- `surface_token`: full token such as `AAI`, `AAD`, `PPI`, or `PPD`

Important sphere/radius fields:

- `prefit_best_radius_um`: natural best-fit sphere before any imposed common or fixed radius
- `applied_reference_radius_um`: sphere radius actually enforced for the final run
- `observed_aperture_radius_um`: largest measured radial distance after recentering to the fitted sphere center
- `applied_normalization_radius_um`: radius used to normalize `rho` for Zernike fitting
- `common_normalization_radius_um`: shared normalization radius when `normalization_mode=common-per-surf-id`

Important quality fields:

- `sphere_rms`
- `surface_zernike_rms`
- `sphere_residual_zernike_rms`
- `surface_zernike_cond`
- `sphere_residual_zernike_cond`

Recommended quick comparison order:

1. `sphere_residual_zernike_rms`
2. `sphere_residual_zernike_cond`
3. `surface_zernike_rms`
4. `sphere_rms`
5. `applied_reference_radius_um`
6. `applied_normalization_radius_um`

## GUI

```bash
uv run python batch_fit_xyz_gui.py
```

The GUI supports two main workflows:

- `Run Batch`
- `Inspect Summary`

### Run Batch

The GUI builds and launches the same batch command you can run from the CLI.

Useful defaults:

- `Method`: `lstsq`
- `ROC mode`: `fit-per-file`
- `Normalization mode`: `per-file`
- round radii: on
- round Zernike coefficients: on

The command preview reflects the actual CLI arguments that will be run.

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

Residual-axis behavior:

- the two residual plots scale independently
- each plot snaps its own y-limits from its own actual extrema

### Path Remapping

If a completed run folder is copied to another machine, the saved workbook may still contain old absolute paths.

The viewer handles this in two layers:

- coefficient files under the original run folder are remapped automatically to the folder containing the loaded workbook
- raw `.xyz` files can be remapped by setting `Local XYZ Root`

`Original XYZ Root` is auto-filled when it can be inferred from the workbook or `run_manifest.json`.

## Typical Workflow

1. Place your `.xyz` files in one folder.
2. Run a batch from the CLI or GUI.
3. Inspect `batch_summary_YYYY-MM-DD.xlsx`.
4. Review `coefficients/` and `ZPs_batch_report.csv`.
5. Reopen the saved summary workbook in the GUI to replay and inspect individual fits.

## Troubleshooting

- If `python3` is missing required packages, use `uv run` instead of system Python.
- If a copied run folder does not replay correctly in the GUI, verify `Local XYZ Root` points at the raw `.xyz` directory.
- If no files are processed, verify your filenames map into the supported `AA/AP/PA/PP` families.
- If some files fail during batch processing, inspect `batch_failures.csv`.
- If you changed dependencies, run `uv sync` again.
- If results differ between runs, compare `method`, `roc_mode`, `normalization_mode`, `n_modes`, and rounding settings before comparing fit metrics.

## Supported Entry Points

- `batch_fit_xyz.py`
- `batch_fit_xyz_gui.py`
