from __future__ import annotations

import hashlib
import csv
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from ..common import is_focus_surface_family, sanitize_h5_name
from ..io.workbook import parse_coefficients_csv, parse_name_value_csv_rows
from ..io.xyz import load_xyz_point_cloud, parse_surface_metadata
from ..models import FitArtifacts
from ..fit.sphere_reference import select_vertex_target
from ..pipeline.surface_fit import build_fit_artifacts
from ..reporting.batch_reports import artifacts_to_summary_row, write_json, write_overview_plot
from ..pipeline.tilt_correction import (
    VertexTiltCorrection,
    apply_vertex_tilt_correction_to_artifacts,
    export_coefficient_rows_for_artifacts,
    split_coefficient_rows,
)


@dataclass(frozen=True)
class SingleFileAnalysisRequest:
    """Immutable request for one single-file analysis run."""

    request_id: int
    source_file: Path
    sphere_fit_mode: str
    center_weight: float
    n_modes: int
    round_radii_um: bool
    zernike_coeff_sigfigs: int | None
    temp_root: Path

    def source_mtime_ns(self) -> int:
        """Return the file mtime used for result caching."""
        return self.source_file.stat().st_mtime_ns

    def cache_key(self) -> tuple[object, ...]:
        """Return the stable cache key for this analysis request."""
        return (
            str(self.source_file.resolve()),
            self.source_mtime_ns(),
            self.sphere_fit_mode,
            float(self.center_weight),
            int(self.n_modes),
            bool(self.round_radii_um),
            self.zernike_coeff_sigfigs,
        )

    def cache_digest(self) -> str:
        """Return a filesystem-safe digest of the cache key."""
        payload = json.dumps(self.cache_key(), sort_keys=False, default=str).encode("utf-8")
        return hashlib.sha1(payload).hexdigest()[:16]


@dataclass(frozen=True)
class SingleFileAnalysisResult:
    """Materialized single-file analysis result with temp artifacts."""

    request_id: int
    cache_key: tuple[object, ...]
    result_dir: Path
    artifacts: FitArtifacts
    coeff_rows: list[tuple[str, str]]
    coeff_meta: dict[str, str]
    coeffs: list[tuple[str, float]]
    summary_row: dict[str, object]
    csv_path: Path
    overview_plot_path: Path
    analysis_json_path: Path

    @property
    def export_dir_name(self) -> str:
        """Return the default directory name for bundle exports."""
        metadata = self.artifacts.metadata
        return sanitize_h5_name(f"{metadata.design_token}_{metadata.surface_token}_{metadata.force_id}")


def _write_name_value_rows(path: Path, rows: list[tuple[str, str]]) -> None:
    """Write two-column name/value rows to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, dialect="excel")
        writer.writerows(rows)


def tilt_correction_summary_rows(correction: VertexTiltCorrection) -> list[tuple[str, str]]:
    """Build diagnostic rows for the optional Single File vertex tilt correction."""
    return [
        ("Vertex tilt correction", "on"),
        ("Original center slope x (mrad)", f"{correction.original_x_mrad:.6f}"),
        ("Original center slope y (mrad)", f"{correction.original_y_mrad:.6f}"),
        ("Original center slope magnitude (mrad)", f"{correction.original_magnitude_mrad:.6f}"),
        ("Corrected center slope x (mrad)", f"{correction.corrected_x_mrad:.6f}"),
        ("Corrected center slope y (mrad)", f"{correction.corrected_y_mrad:.6f}"),
        ("Corrected center slope magnitude (mrad)", f"{correction.corrected_magnitude_mrad:.6f}"),
        ("Applied Z2 correction (um)", f"{correction.delta_z2_um:.9f}"),
        ("Applied Z3 correction (um)", f"{correction.delta_z3_um:.9f}"),
    ]


def build_tilt_corrected_single_file_result(result: SingleFileAnalysisResult) -> SingleFileAnalysisResult:
    """Build a Single File result view whose exported/displayed coefficients have zero vertex tilt."""
    corrected_artifacts, correction = apply_vertex_tilt_correction_to_artifacts(result.artifacts)
    corrected_csv_path = result.result_dir / f"{result.csv_path.stem}_zero_vertex_tilt.csv"
    corrected_overview_path = result.result_dir / "overview_zero_vertex_tilt.png"
    corrected_json_path = result.result_dir / "analysis_zero_vertex_tilt.json"
    corrected_artifacts = replace(
        corrected_artifacts,
        output_coefficients_csv=corrected_csv_path,
        run_name=f"{corrected_artifacts.run_name} | zero vertex tilt",
    )
    coeff_rows = export_coefficient_rows_for_artifacts(corrected_artifacts)
    coeff_meta, coeffs = split_coefficient_rows(coeff_rows)
    summary_row = artifacts_to_summary_row(corrected_artifacts)
    summary_row.update(
        {
            "vertex_tilt_correction": "on",
            "original_center_slope_x_mrad": correction.original_x_mrad,
            "original_center_slope_y_mrad": correction.original_y_mrad,
            "original_center_slope_magnitude_mrad": correction.original_magnitude_mrad,
            "corrected_center_slope_x_mrad": correction.corrected_x_mrad,
            "corrected_center_slope_y_mrad": correction.corrected_y_mrad,
            "corrected_center_slope_magnitude_mrad": correction.corrected_magnitude_mrad,
            "delta_z2_um": correction.delta_z2_um,
            "delta_z3_um": correction.delta_z3_um,
        }
    )

    _write_name_value_rows(corrected_csv_path, coeff_rows)
    write_overview_plot(corrected_overview_path, corrected_artifacts)
    write_json(
        corrected_json_path,
        {
            "request_id": result.request_id,
            "cache_key": [*list(result.cache_key), "zero_vertex_tilt"],
            "source_file": str(corrected_artifacts.source_file),
            "csv_path": str(corrected_csv_path),
            "overview_plot_path": str(corrected_overview_path),
            "summary_row": summary_row,
            "coeff_rows": [{"name": name, "value": value} for name, value in coeff_rows],
            "vertex_tilt_correction": dict(summary_row),
        },
    )

    return SingleFileAnalysisResult(
        request_id=result.request_id,
        cache_key=(*result.cache_key, "zero_vertex_tilt"),
        result_dir=result.result_dir,
        artifacts=corrected_artifacts,
        coeff_rows=coeff_rows,
        coeff_meta=coeff_meta,
        coeffs=coeffs,
        summary_row=summary_row,
        csv_path=corrected_csv_path,
        overview_plot_path=corrected_overview_path,
        analysis_json_path=corrected_json_path,
    )


def _is_single_file_supported_metadata(metadata: object) -> bool:
    return bool(
        getattr(metadata, "filename_kind", "standard") == "generic"
        or is_focus_surface_family(getattr(metadata, "surf_id", ""))
    )


def _infer_generic_surface_token(source_file: Path) -> str:
    """Infer a Single File-only generic surface convention from point-cloud shape."""
    x_vals, y_vals, z_vals = load_xyz_point_cloud(source_file)
    x_arr = np.asarray(x_vals, dtype=float)
    y_arr = np.asarray(y_vals, dtype=float)
    z_arr = np.asarray(z_vals, dtype=float)
    target = select_vertex_target(x_arr, y_arr, z_arr)
    rho = np.sqrt(np.square(x_arr) + np.square(y_arr))
    rho_max = float(np.max(rho))
    edge = rho >= rho_max * 0.999 if rho_max > 0.0 else np.ones_like(rho, dtype=bool)
    edge_z = z_arr[edge] if np.any(edge) else z_arr
    edge_median_z = float(np.median(edge_z))
    return "CX" if target.z_um >= edge_median_z else "CC"


def resolve_single_file_metadata(source_file: Path) -> object:
    """Resolve parser metadata, inferring generic Single File surface convention when needed."""
    metadata = parse_surface_metadata(source_file)
    if getattr(metadata, "filename_kind", "standard") != "generic":
        return metadata
    surface_token = _infer_generic_surface_token(source_file)
    return replace(metadata, surface_token=surface_token, surf_id=surface_token)


def validate_single_file_source(source_file: Path) -> None:
    """Reject unsupported single-file inputs with a clear error."""
    if not source_file.exists():
        raise ValueError(f"Selected source file does not exist: {source_file}")
    if source_file.suffix.lower() != ".xyz":
        raise ValueError(f"Single-file analysis supports only .xyz point-cloud files; got {source_file.name}.")
    metadata = parse_surface_metadata(source_file)
    if not _is_single_file_supported_metadata(metadata):
        raise ValueError(
            "Single-file analysis supports AA/AP/PA/PP maintained filenames or generic .xyz files; "
            f"got surf_id={metadata.surf_id!r} for {source_file.name}."
        )


def build_single_file_candidates(folder: Path, *, pattern: str = "*.xyz") -> list[Path]:
    """Return top-level XYZ candidates accepted by the Single File workflow."""
    if not folder.exists():
        return []
    return sorted(
        [
            path
            for path in folder.glob(pattern)
            if path.is_file() and _is_single_file_supported_metadata(parse_surface_metadata(path))
        ],
        key=lambda path: (parse_surface_metadata(path).filename_kind == "generic", path.name.lower()),
    )


def run_single_file_analysis(request: SingleFileAnalysisRequest) -> SingleFileAnalysisResult:
    """Run the maintained single-file analysis and write temp outputs."""
    source_file = request.source_file.resolve()
    validate_single_file_source(source_file)
    metadata = resolve_single_file_metadata(source_file)

    result_dir = request.temp_root / request.cache_digest()
    result_dir.mkdir(parents=True, exist_ok=True)

    artifacts = build_fit_artifacts(
        source_file,
        metadata=metadata,
        source_metadata=metadata,
        output_dir=result_dir,
        n_modes=request.n_modes,
        rcond=None,
        roc_mode="fit-per-file",
        reference_radius_um=None,
        sphere_fit_mode=request.sphere_fit_mode,
        center_weight=request.center_weight,
        normalization_mode="per-file",
        normalization_radius_um=None,
        run_name="single_file_analysis",
        common_reference_radius_um=None,
        common_normalization_radius_um=None,
        round_radii_um=request.round_radii_um,
        zernike_coeff_sigfigs=request.zernike_coeff_sigfigs,
    )

    csv_path = artifacts.output_coefficients_csv
    coeff_rows = parse_name_value_csv_rows(csv_path)
    coeff_meta, coeffs = parse_coefficients_csv(csv_path)
    overview_plot_path = result_dir / "overview.png"
    analysis_json_path = result_dir / "analysis.json"
    write_overview_plot(overview_plot_path, artifacts)

    summary_row = artifacts_to_summary_row(artifacts)
    summary_row.update(
        {
            "filename_kind": metadata.filename_kind,
            "filename_suffix": metadata.filename_suffix,
            "surface_convention": metadata.surf_id if metadata.filename_kind == "generic" else "",
        }
    )
    write_json(
        analysis_json_path,
        {
            "request_id": request.request_id,
            "cache_key": list(request.cache_key()),
            "source_file": str(source_file),
            "csv_path": str(csv_path),
            "overview_plot_path": str(overview_plot_path),
            "summary_row": summary_row,
            "coeff_rows": [{"name": name, "value": value} for name, value in coeff_rows],
        },
    )

    return SingleFileAnalysisResult(
        request_id=request.request_id,
        cache_key=request.cache_key(),
        result_dir=result_dir,
        artifacts=artifacts,
        coeff_rows=coeff_rows,
        coeff_meta=coeff_meta,
        coeffs=coeffs,
        summary_row=summary_row,
        csv_path=csv_path,
        overview_plot_path=overview_plot_path,
        analysis_json_path=analysis_json_path,
    )


class SingleFileAnalysisWorker(QObject):
    """Background Qt worker for single-file analysis."""

    finished = pyqtSignal(object)
    failed = pyqtSignal(int, str)

    def __init__(self, request: SingleFileAnalysisRequest) -> None:
        super().__init__()
        self.request = request

    @pyqtSlot()
    def run(self) -> None:
        """Execute one single-file analysis request."""
        try:
            result = run_single_file_analysis(self.request)
        except Exception as exc:
            self.failed.emit(self.request.request_id, str(exc))
            return
        self.finished.emit(result)


__all__ = [
    "SingleFileAnalysisRequest",
    "SingleFileAnalysisResult",
    "SingleFileAnalysisWorker",
    "VertexTiltCorrection",
    "build_tilt_corrected_single_file_result",
    "build_single_file_candidates",
    "resolve_single_file_metadata",
    "run_single_file_analysis",
    "tilt_correction_summary_rows",
    "validate_single_file_source",
]
