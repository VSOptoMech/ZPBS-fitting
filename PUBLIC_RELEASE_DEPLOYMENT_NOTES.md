# Public Release Audit Notes

Date: 2026-04-24

## Findings Addressed

1. Private subset-inspection code was present in the public source tree.
   - Risk: public users could read, import, or re-enable private workbook/plotting logic.
   - Fix: removed subset UI, plotting canvas, labels, workbook-kind detection, imports, and exports from `src/zpbs`.

2. GUI HDF5 defaults exposed project-specific naming and encouraged raw-data output.
   - Risk: HDF5 output contains raw point-cloud arrays, absolute source paths, design IDs, FEA IDs, and fit metadata.
   - Fix: renamed the default HDF5 path to `batch_results.h5`, disabled HDF5 in the GUI by default, added a raw-data warning tooltip, documented the risk, and ignored `*.h5` / `*.hdf5`.

3. Initial-state collapse grouped files only by surface family.
   - Risk: multi-design batches could fail unnecessarily or collapse identical initial files across design/FEA boundaries and lose metadata.
   - Fix: grouped initial `*I` inputs by `(design_token, fea_id, surf_id, surface_token)`.

4. Coefficient export hard-coded `R01V`.
   - Risk: non-`R01V` designs were mislabeled in per-file and batch coefficient reports.
   - Fix: threaded the parsed `design_token` into coefficient export rows while preserving the old `R01V{design_id}` fallback for direct helper calls.

## Files Changed

- `.gitignore`
- `README.md`
- `src/zpbs/azp_csv_pipeline.py`
- `src/zpbs/gui/__init__.py`
- `src/zpbs/gui/canvases.py`
- `src/zpbs/gui/support.py`
- `src/zpbs/gui/window.py`
- `src/zpbs/io/workbook.py`
- `src/zpbs/io/xyz.py`
- `src/zpbs/pipeline/surface_fit.py`
- `src/zpbs/reporting/batch_reports.py`

## Deployment Script Checks

Run these checks before publishing a public release:

```bash
uv run python -m compileall -q src
uv run zpbs-batch-fit --help >/tmp/zpbs_batch_help.txt
uv run python -c "import zpbs.gui; from zpbs.gui.window import BatchFitWindow; print('imports ok')"
git diff --check
```

Block release if private subset-inspection identifiers reappear in public source:

```bash
if rg -n "Subset|subset|SUBSET|detect_subset|SubsetPlotCanvas|zernike_subset|drop_importance|global_consistent|mode_consistency" src/zpbs; then
  echo "Private subset-inspection code is present in public source." >&2
  exit 1
fi
```

Block release if project-specific HDF5 names reappear:

```bash
if rg -n -i "ardea|real_data_validation" src README.md pyproject.toml .gitignore; then
  echo "Project-specific public-release term found." >&2
  exit 1
fi
```

Block release if tracked secrets or sensitive markers are detected:

```bash
git grep -nI -E -i \
  "(api[_-]?key|secret|token|password|passwd|credential|proprietary|confidential|internal use|do not distribute|customer|client|AKIA|BEGIN (RSA|OPENSSH|PRIVATE) KEY|sk-[A-Za-z0-9]|xox[baprs]-|ghp_|github_pat_|mongodb\\+srv|postgres://|mysql://|https?://[^[:space:]]*:[^[:space:]@]+@)" \
  -- .
```

The secret scan may report false positives for ordinary metadata field names such as `design_token` or `surface_token`; deployment automation should either maintain an allowlist for known source identifiers or require manual review of matches.

## Behavioral Smoke Checks

Confirm multi-design initial files are not collapsed across design boundaries:

```bash
uv run python - <<'PY'
from pathlib import Path
from tempfile import TemporaryDirectory
from zpbs.io.xyz import collapse_identical_initial_inputs

with TemporaryDirectory() as tmp:
    root = Path(tmp)
    payload = "X,Y,Z\n0,0,0\n1,0,0\n"
    (root / "R01V1-A_F1.0mN_FVS_AAI.xyz").write_text(payload)
    (root / "R02V2-B_F1.0mN_FVS_AAI.xyz").write_text(payload)
    items = collapse_identical_initial_inputs(sorted(root.glob("*.xyz")))
    assert len(items) == 2, len(items)
    assert [item.metadata.design_token for item in items] == ["R01V1", "R02V2"]
print("initial-collapse boundary ok")
PY
```

Confirm coefficient export preserves non-`R01V` design tokens:

```bash
uv run python - <<'PY'
import numpy as np
from zpbs.azp_csv_pipeline import build_zernike_coefficients_rows

row = build_zernike_coefficients_rows(
    design_id="123",
    design_token="R02V123",
    fea_id="FEA",
    surf_id="AA",
    tension_mn="0",
    base_sphere_roc_um=1000,
    vertex_um=0,
    vertex_residual_um=0,
    norm_radius_um=1000,
    zernike_coefficients_mm=np.zeros(45),
)[0]
assert row == ("Design", "R02V123"), row
print("design-token export ok")
PY
```

Confirm GUI HDF5 remains opt-in:

```bash
QT_QPA_PLATFORM=offscreen uv run python - <<'PY'
import sys
from PyQt5.QtWidgets import QApplication
from zpbs.gui.window import BatchFitWindow

app = QApplication(sys.argv)
window = BatchFitWindow()
assert window.h5_path_edit.text() == "batch_results.h5"
assert window.h5_enabled_check.isChecked() is False
assert "--h5-path" not in window.command_preview.toPlainText()
window.close()
app.quit()
print("gui hdf5 default ok")
PY
```
