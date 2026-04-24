from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str((Path(__file__).resolve().parents[3] / ".matplotlib").resolve()))

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QTableWidgetItem, QWidget

from ..common import (
    parse_boolish,
    parse_optional_int,
    uses_posterior_sign_convention,
    validate_sphere_reference_configuration,
    validate_zernike_method,
)
from ..io.xyz import load_xyz_point_cloud, parse_surface_metadata
from ..models import FitArtifacts
from ..pipeline.surface_fit import run_fit_pipeline, zpbs_residual_on_axis_m0_um
from ..reporting.batch_reports import render_overview_plot

from .plotting import render_detailed_analysis_figure
from .support import parse_optional_float
from ..io.workbook import parse_coefficients_csv


class NumericTableWidgetItem(QTableWidgetItem):
    """QTableWidget item with stable numeric sorting."""

    def __init__(self, text: str, sort_value: float | int | str) -> None:
        super().__init__(text)
        self.sort_value = sort_value

    def __lt__(self, other: object) -> bool:
        if isinstance(other, NumericTableWidgetItem):
            return self.sort_value < other.sort_value
        return super().__lt__(other)


class OverviewPlotCanvas(FigureCanvas):
    """High-level QA-style overview plot for a single fitted file."""

    def __init__(self, parent: QWidget | None = None) -> None:
        self.figure = Figure(figsize=(9.2, 5.8), constrained_layout=True)
        super().__init__(self.figure)
        self.setParent(parent)

    def clear_message(self, message: str) -> None:
        """Replace the canvas contents with a centered status message."""
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        axis.axis("off")
        axis.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
        self.draw_idle()

    def plot_artifacts(self, artifacts: FitArtifacts) -> None:
        """Render the overview plot directly from one fit result."""
        render_overview_plot(self.figure, artifacts, bins=128, compact=False)
        self.draw_idle()


class FitPreviewCanvas(FigureCanvas):
    """Matplotlib canvas for interactive inspection of one summary row."""

    def __init__(self, parent: QWidget | None = None) -> None:
        self.figure = Figure(figsize=(9.0, 7.0), constrained_layout=True)
        self._preview_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        super().__init__(self.figure)
        self.setParent(parent)

    @staticmethod
    def _mtime_ns(path: Path) -> int | None:
        """Return a file's nanosecond mtime when present, else None."""
        try:
            return path.stat().st_mtime_ns
        except FileNotFoundError:
            return None

    def _build_preview_cache_key(
        self,
        *,
        source_file: Path,
        coeff_file: Path,
        reference_radius: float | None,
        normalization_mode: str,
        normalization_radius: float | None,
        method: str,
        n_modes: int,
        rcond: float | None,
        sphere_fit_mode: str,
        center_weight: float,
        round_radii_um: bool,
        zernike_coeff_sigfigs: int | None,
    ) -> tuple[Any, ...]:
        """Build a stable cache key for expensive preview recomputation."""
        return (
            str(source_file),
            self._mtime_ns(source_file),
            str(coeff_file),
            self._mtime_ns(coeff_file),
            reference_radius,
            normalization_mode,
            normalization_radius,
            method,
            n_modes,
            rcond,
            sphere_fit_mode,
            center_weight,
            round_radii_um,
            zernike_coeff_sigfigs,
        )

    def _load_preview_payload(
        self,
        *,
        row: dict[str, str],
        manifest: dict[str, object],
        source_file: Path,
        coeff_file: Path,
        metadata: Any,
        reference_radius: float | None,
        normalization_mode: str,
        normalization_radius: float | None,
        method: str,
        n_modes: int,
        rcond: float | None,
        sphere_fit_mode: str,
        center_weight: float,
        round_radii_um: bool,
        zernike_coeff_sigfigs: int | None,
    ) -> dict[str, Any]:
        """Load and cache the expensive fit/coefficients payload for one preview selection."""
        cache_key = self._build_preview_cache_key(
            source_file=source_file,
            coeff_file=coeff_file,
            reference_radius=reference_radius,
            normalization_mode=normalization_mode,
            normalization_radius=normalization_radius,
            method=method,
            n_modes=n_modes,
            rcond=rcond,
            sphere_fit_mode=sphere_fit_mode,
            center_weight=center_weight,
            round_radii_um=round_radii_um,
            zernike_coeff_sigfigs=zernike_coeff_sigfigs,
        )
        payload = self._preview_cache.get(cache_key)
        if payload is not None:
            return payload

        x, y, z = load_xyz_point_cloud(source_file)
        fit_data = run_fit_pipeline(
            x,
            y,
            z,
            surf_id=metadata.surf_id,
            method=method,
            n_modes=n_modes,
            rcond=rcond,
            reference_radius_um=reference_radius,
            normalization_radius_um=normalization_radius,
            sphere_fit_mode=sphere_fit_mode,
            center_weight=center_weight,
            round_radii_um=round_radii_um,
            zernike_coeff_sigfigs=zernike_coeff_sigfigs,
        )
        coeff_meta, coeffs = parse_coefficients_csv(coeff_file)
        artifacts = self._fit_artifacts_from_preview(
            row=row,
            manifest=manifest,
            source_file=source_file,
            coeff_file=coeff_file,
            metadata=metadata,
            fit_data=fit_data,
        )
        payload = {
            "artifacts": artifacts,
            "coeff_meta": coeff_meta,
            "coeffs": coeffs,
            "row": dict(row),
            "manifest": dict(manifest),
        }
        self._preview_cache[cache_key] = payload
        return payload

    def _fit_artifacts_from_preview(
        self,
        *,
        row: dict[str, str],
        manifest: dict[str, object],
        source_file: Path,
        coeff_file: Path,
        metadata: Any,
        fit_data: dict[str, Any],
    ) -> FitArtifacts:
        """Build a FitArtifacts-compatible object from replayed summary data."""
        rho = np.asarray(fit_data["rho"], dtype=float)
        z = np.asarray(fit_data["z"], dtype=float)
        zeros = np.zeros_like(rho)
        sphere_residuals_um = np.asarray(
            fit_data.get("sphere_residuals_um", fit_data.get("sphere_residuals", zeros)),
            dtype=float,
        )
        rho_norm = np.asarray(fit_data.get("rho_norm", rho / max(float(np.max(rho)), 1.0)), dtype=float)
        phi = np.asarray(fit_data.get("phi", zeros), dtype=float)
        zpbs_residual_surface_um = np.asarray(
            fit_data.get(
                "zpbs_residual_surface_um",
                fit_data.get("zernike_residual_surface", zeros),
            ),
            dtype=float,
        )
        zpbs_residual_residuals_um = np.asarray(
            fit_data.get(
                "zpbs_residual_residuals_um",
                fit_data.get("zernike_residual_residuals", sphere_residuals_um - zpbs_residual_surface_um),
            ),
            dtype=float,
        )
        zpbs_residual_coefficients_um = np.asarray(
            fit_data.get("zpbs_residual_coefficients_um", fit_data.get("zpoly_fits2", np.zeros(45, dtype=float))),
            dtype=float,
        )
        if "zpbs_to_data_surface_um" in fit_data:
            zpbs_to_data_surface_um = np.asarray(fit_data["zpbs_to_data_surface_um"], dtype=float)
            zpbs_to_data_residuals_um = np.asarray(
                fit_data.get("zpbs_to_data_residuals_um", z - zpbs_to_data_surface_um),
                dtype=float,
            )
        else:
            branch_sign = 1.0
            if "reference_vertex_z_um" in fit_data and "z0_fit" in fit_data:
                branch_sign = 1.0 if float(fit_data["reference_vertex_z_um"]) >= float(fit_data["z0_fit"]) else -1.0
            elif uses_posterior_sign_convention(str(metadata.surf_id)):
                branch_sign = -1.0
            term = np.sqrt(np.clip(float(fit_data["applied_reference_radius_um"]) ** 2 - np.square(rho), 0.0, None))
            sphere_surface_um = float(fit_data["z0_fit"]) + (branch_sign * term)
            residual_sign = -1.0 if uses_posterior_sign_convention(str(metadata.surf_id)) else 1.0
            zpbs_to_data_surface_um = sphere_surface_um + (residual_sign * zpbs_residual_surface_um)
            zpbs_to_data_residuals_um = z - zpbs_to_data_surface_um
        sphere_sse_um2 = float(
            fit_data.get("sphere_sse_um2", fit_data.get("sphere_sse", np.sum(np.square(sphere_residuals_um))))
        )
        sphere_mae_um = float(
            fit_data.get("sphere_mae_um", fit_data.get("sphere_mae", np.mean(np.abs(sphere_residuals_um))))
        )
        sphere_rms_um = float(
            fit_data.get("sphere_rms_um", fit_data.get("sphere_rms", np.sqrt(np.mean(np.square(sphere_residuals_um)))))
        )
        zpbs_residual_sse_um2 = float(
            fit_data.get(
                "zpbs_residual_sse_um2",
                fit_data.get("sphere_residual_zernike_sse", np.sum(np.square(zpbs_residual_residuals_um))),
            )
        )
        zpbs_residual_mae_um = float(
            fit_data.get(
                "zpbs_residual_mae_um",
                fit_data.get("sphere_residual_zernike_mae", np.mean(np.abs(zpbs_residual_residuals_um))),
            )
        )
        zpbs_residual_rms_um = float(
            fit_data.get(
                "zpbs_residual_rms_um",
                fit_data.get("sphere_residual_zernike_rms", np.sqrt(np.mean(np.square(zpbs_residual_residuals_um)))),
            )
        )
        zpbs_residual_cond = float(
            fit_data.get("zpbs_residual_cond", fit_data.get("sphere_residual_zernike_cond", 0.0))
        )
        common_reference_radius_um = parse_optional_float(row.get("common_reference_radius_um", ""))
        common_normalization_radius_um = parse_optional_float(row.get("common_normalization_radius_um", ""))
        idx_center = int(np.argmin(rho)) if len(rho) else 0
        row_vertex_residual_um = parse_optional_float(row.get("vertex_residual_um", ""))
        roc_mode = str(row.get("roc_mode", manifest.get("roc_mode", "fit-per-file"))).strip() or "fit-per-file"
        normalization_mode = (
            str(row.get("normalization_mode", manifest.get("normalization_mode", "per-file"))).strip() or "per-file"
        )
        run_name = str(row.get("run_name", manifest.get("run_name", "summary_preview"))).strip() or "summary_preview"
        return FitArtifacts(
            metadata=metadata,
            source_metadata=metadata,
            source_file=source_file,
            output_coefficients_csv=coeff_file,
            points_used=len(fit_data["x"]),
            x=fit_data["x"],
            y=fit_data["y"],
            z=z,
            rho=rho,
            phi=phi,
            rho_norm=rho_norm,
            sphere_residuals_um=sphere_residuals_um,
            zpbs_to_data_surface_um=zpbs_to_data_surface_um,
            zpbs_to_data_residuals_um=zpbs_to_data_residuals_um,
            zpbs_residual_surface_um=zpbs_residual_surface_um,
            zpbs_residual_residuals_um=zpbs_residual_residuals_um,
            zpbs_residual_coefficients_um=zpbs_residual_coefficients_um,
            x0_fit=float(fit_data["x0_fit"]),
            y0_fit=float(fit_data["y0_fit"]),
            z0_fit=float(fit_data["z0_fit"]),
            fitted_sphere_radius_um=float(fit_data["fitted_sphere_radius_um"]),
            applied_reference_radius_um=float(fit_data["applied_reference_radius_um"]),
            prefit_best_radius_um=parse_optional_float(row.get("prefit_best_radius_um", "")),
            sphere_fit_mode=str(fit_data["sphere_fit_mode"]),
            center_weight=float(fit_data["center_weight"]),
            sphere_sse_um2=sphere_sse_um2,
            sphere_mae_um=sphere_mae_um,
            sphere_rms_um=sphere_rms_um,
            zpbs_residual_sse_um2=zpbs_residual_sse_um2,
            zpbs_residual_mae_um=zpbs_residual_mae_um,
            zpbs_residual_rms_um=zpbs_residual_rms_um,
            zpbs_residual_cond=zpbs_residual_cond,
            zpbs_residual_on_axis_m0_um=float(
                fit_data.get("zpbs_residual_on_axis_m0_um", zpbs_residual_on_axis_m0_um(zpbs_residual_coefficients_um))
            ),
            observed_aperture_radius_um=float(fit_data["observed_aperture_radius_um"]),
            norm_radius_um=float(fit_data["norm_radius_um"]),
            target_vertex_x_um=float(fit_data["target_vertex_x_um"]),
            target_vertex_y_um=float(fit_data["target_vertex_y_um"]),
            target_vertex_z_um=float(fit_data["target_vertex_z_um"]),
            reference_vertex_x_um=float(fit_data["reference_vertex_x_um"]),
            reference_vertex_y_um=float(fit_data["reference_vertex_y_um"]),
            reference_vertex_z_um=float(fit_data["reference_vertex_z_um"]),
            vertex_mismatch_z_um=float(fit_data["vertex_mismatch_z_um"]),
            vertex_um=float(
                fit_data.get(
                    "vertex_fit_um",
                    fit_data.get(
                        "zv",
                        z[0] if len(z) else 0.0,
                    )
                    - fit_data.get(
                        "vertex_residual_um",
                        row_vertex_residual_um
                        if row_vertex_residual_um is not None
                        else (zpbs_to_data_residuals_um[idx_center] if len(zpbs_to_data_residuals_um) else 0.0),
                    ),
                )
            ),
            sphere_vertex_residual_um=float(
                fit_data.get(
                    "sphere_vertex_residual_um",
                    fit_data.get("zv2", sphere_residuals_um[idx_center] if len(sphere_residuals_um) else 0.0),
                )
            ),
            vertex_residual_um=float(
                fit_data.get(
                    "vertex_residual_um",
                    row_vertex_residual_um
                    if row_vertex_residual_um is not None
                    else (zpbs_to_data_residuals_um[idx_center] if len(zpbs_to_data_residuals_um) else 0.0),
                )
            ),
            method=str(fit_data.get("method", "lstsq")),
            n_modes=int(fit_data.get("n_modes", 45)),
            roc_mode=roc_mode,
            normalization_mode=normalization_mode,
            run_name=run_name,
            common_reference_radius_um=common_reference_radius_um,
            common_normalization_radius_um=common_normalization_radius_um,
            round_radii_um=bool(fit_data.get("round_radii_um", False)),
            zernike_coeff_sigfigs=fit_data.get("zernike_coeff_sigfigs"),
        )

    @staticmethod
    def _is_compact_summary_row(row: dict[str, str], manifest: dict[str, object]) -> bool:
        schema_version = parse_optional_int(manifest.get("summary_schema_version"))
        return bool(schema_version and schema_version >= 2) or "run_name" not in row

    def _resolve_replay_setting(
        self,
        *,
        row: dict[str, str],
        manifest: dict[str, object],
        field_name: str,
        legacy_default: object,
        required_for_compact: bool = True,
    ) -> object:
        row_value = row.get(field_name)
        if row_value not in (None, ""):
            return row_value
        manifest_value = manifest.get(field_name)
        if manifest_value not in (None, "", "null"):
            return manifest_value
        if self._is_compact_summary_row(row, manifest) and required_for_compact:
            raise ValueError(
                f"Compact summary replay requires {field_name!r} in the workbook row or sibling run_manifest.json."
            )
        return legacy_default

    def clear_message(self, message: str) -> None:
        """Replace the canvas contents with a centered status message."""
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        axis.axis("off")
        axis.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
        self.draw_idle()

    def plot_selection(
        self,
        row: dict[str, str],
        manifest: dict[str, object],
        rho_axis_limit_um: float | None = None,
    ) -> tuple[str, dict[str, str]]:
        source_file = Path(row.get("_resolved_source_file", row["source_file"]))
        coeff_file = Path(row.get("_resolved_output_coefficients_csv", row["output_coefficients_csv"]))
        metadata = parse_surface_metadata(source_file)
        compact_summary = self._is_compact_summary_row(row, manifest)

        roc_mode = (
            str(
                self._resolve_replay_setting(
                    row=row,
                    manifest=manifest,
                    field_name="roc_mode",
                    legacy_default="fit-per-file",
                )
            ).strip()
            or "fit-per-file"
        )
        normalization_mode = (
            str(
                self._resolve_replay_setting(
                    row=row,
                    manifest=manifest,
                    field_name="normalization_mode",
                    legacy_default="per-file",
                )
            ).strip()
            or "per-file"
        )
        method = validate_zernike_method(
            str(
                self._resolve_replay_setting(
                    row=row,
                    manifest=manifest,
                    field_name="method",
                    legacy_default="lstsq",
                )
            )
        )
        sphere_fit_mode = (
            str(
                self._resolve_replay_setting(
                    row=row,
                    manifest=manifest,
                    field_name="sphere_fit_mode",
                    legacy_default="legacy_lsq",
                )
            ).strip()
            or "legacy_lsq"
        )
        center_weight_raw = self._resolve_replay_setting(
            row=row,
            manifest=manifest,
            field_name="center_weight",
            legacy_default="0.0",
        )
        center_weight = 0.0 if center_weight_raw in (None, "", "null") else float(str(center_weight_raw))
        n_modes = int(
            float(
                self._resolve_replay_setting(
                    row=row,
                    manifest=manifest,
                    field_name="n_modes",
                    legacy_default="45",
                )
            )
        )
        round_radii_um = parse_boolish(
            self._resolve_replay_setting(
                row=row,
                manifest=manifest,
                field_name="round_radii_um",
                legacy_default=False,
            ),
            default=False,
        )
        zernike_coeff_sigfigs = parse_optional_int(
            self._resolve_replay_setting(
                row=row,
                manifest=manifest,
                field_name="zernike_coeff_sigfigs",
                legacy_default=None,
                required_for_compact=False,
            )
        )

        reference_radius = None
        if roc_mode != "fit-per-file":
            reference_radius = parse_optional_float(row.get("applied_reference_radius_um", ""))
            if compact_summary and reference_radius is None:
                raise ValueError(
                    "Compact summary replay requires 'applied_reference_radius_um' for non per-file ROC runs."
                )

        normalization_radius = None
        if normalization_mode == "common-per-surf-id":
            normalization_radius = parse_optional_float(row.get("applied_normalization_radius_um", ""))
            if compact_summary and normalization_radius is None:
                raise ValueError(
                    "Compact summary replay requires 'applied_normalization_radius_um' for common normalization runs."
                )

        rcond_raw = manifest.get("rcond")
        rcond = None if rcond_raw in (None, "", "null") else float(str(rcond_raw))
        validate_sphere_reference_configuration(
            roc_mode=roc_mode,
            sphere_fit_mode=sphere_fit_mode,
        )
        payload = self._load_preview_payload(
            row=row,
            manifest=manifest,
            source_file=source_file,
            coeff_file=coeff_file,
            metadata=metadata,
            reference_radius=reference_radius,
            normalization_mode=normalization_mode,
            normalization_radius=normalization_radius,
            method=method,
            n_modes=n_modes,
            rcond=rcond,
            sphere_fit_mode=sphere_fit_mode,
            center_weight=center_weight,
            round_radii_um=round_radii_um,
            zernike_coeff_sigfigs=zernike_coeff_sigfigs,
        )
        text, details = self.plot_artifacts(
            payload["artifacts"],
            coeff_meta=payload["coeff_meta"],
            coeffs=payload["coeffs"],
            source_file=source_file,
            rho_axis_limit_um=rho_axis_limit_um,
        )
        self.draw_idle()
        return text, details

    def plot_artifacts(
        self,
        artifacts: FitArtifacts,
        *,
        coeff_meta: dict[str, str],
        coeffs: list[tuple[str, float]],
        source_file: Path | None = None,
        rho_axis_limit_um: float | None = None,
    ) -> tuple[str, dict[str, str]]:
        """Render the detailed plot directly from one fit result."""
        text, details = render_detailed_analysis_figure(
            self.figure,
            artifacts=artifacts,
            coeff_meta=coeff_meta,
            coeffs=coeffs,
            source_file=source_file,
            rho_axis_limit_um=rho_axis_limit_um,
        )
        return text, details
