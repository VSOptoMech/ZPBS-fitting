from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from ..common import is_focus_surface_family, sanitize_h5_name
from ..io.workbook import parse_coefficients_csv, parse_name_value_csv_rows
from ..io.xyz import parse_surface_metadata
from ..models import FitArtifacts
from ..pipeline.surface_fit import build_fit_artifacts
from ..reporting.batch_reports import artifacts_to_summary_row, write_json, write_overview_plot


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


def validate_single_file_source(source_file: Path) -> None:
    """Reject non-maintained single-file inputs with a clear error."""
    if not source_file.exists():
        raise ValueError(f"Selected source file does not exist: {source_file}")
    metadata = parse_surface_metadata(source_file)
    if not is_focus_surface_family(metadata.surf_id):
        raise ValueError(
            "Single-file analysis supports only AA/AP/PA/PP inputs in the maintained GUI; "
            f"got surf_id={metadata.surf_id!r} for {source_file.name}."
        )


def build_single_file_candidates(folder: Path, *, pattern: str = "*_FVS_*.xyz") -> list[Path]:
    """Return the maintained top-level XYZ candidates for folder-backed review."""
    if not folder.exists():
        return []
    return sorted(
        [
            path
            for path in folder.glob(pattern)
            if path.is_file() and is_focus_surface_family(parse_surface_metadata(path).surf_id)
        ],
        key=lambda path: path.name.lower(),
    )


def run_single_file_analysis(request: SingleFileAnalysisRequest) -> SingleFileAnalysisResult:
    """Run the maintained single-file analysis and write temp outputs."""
    source_file = request.source_file.resolve()
    validate_single_file_source(source_file)
    metadata = parse_surface_metadata(source_file)

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
    "build_single_file_candidates",
    "run_single_file_analysis",
    "validate_single_file_source",
]
