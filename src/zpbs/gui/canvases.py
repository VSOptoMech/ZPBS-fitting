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
from .support import (
    format_metric,
    parse_int_list_text,
    parse_optional_float,
)
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


class SubsetPlotCanvas(FigureCanvas):
    """Dedicated canvas for subset omission workbooks."""

    def __init__(self, parent: QWidget | None = None) -> None:
        self.figure = Figure(figsize=(9.0, 7.0), constrained_layout=True)
        super().__init__(self.figure)
        self.setParent(parent)

    def clear_message(self, message: str) -> None:
        """Replace the canvas contents with a centered status message."""
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        axis.axis("off")
        axis.text(0.5, 0.5, message, ha="center", va="center", fontsize=12)
        self.draw_idle()

    def plot_drop_importance(self, rows: list[dict[str, str]], selected_row: dict[str, str] | None) -> str:
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        ordered_rows = sorted(rows, key=lambda row: int(float(row.get("impact_rank", "0"))))
        modes = [int(float(row.get("removed_mode_noll", "0"))) for row in ordered_rows]
        deltas = [float(row.get("delta_rms_vs_full", "0")) for row in ordered_rows]
        colors = ["#94a3b8"] * len(ordered_rows)
        selected_text = "No selected mode."
        if selected_row is not None:
            selected_mode = int(float(selected_row.get("removed_mode_noll", "0")))
            for index, mode in enumerate(modes):
                if mode == selected_mode:
                    colors[index] = "#dc2626"
                    break
            selected_text = (
                f"Removed mode Z{selected_mode} | "
                f"delta_rms_vs_full={format_metric(float(selected_row.get('delta_rms_vs_full', '0')))} | "
                f"subset_rms={format_metric(float(selected_row.get('subset_rms', '0')))}"
            )
        axis.bar(modes, deltas, color=colors, width=0.85)
        axis.set_title("Drop Importance: delta RMS vs full fit", fontsize=11)
        axis.set_xlabel("Removed Mode (Noll)")
        axis.set_ylabel("delta_rms_vs_full")
        axis.grid(True, axis="y", alpha=0.2)
        axis.text(0.01, 0.99, selected_text, transform=axis.transAxes, ha="left", va="top", fontsize=9)
        self.draw_idle()
        return selected_text

    def plot_subset_path(
        self,
        rows: list[dict[str, str]],
        selected_row: dict[str, str] | None,
        *,
        title: str,
    ) -> str:
        self.figure.clear()
        grid = self.figure.add_gridspec(2, 1, height_ratios=[1.2, 0.9])
        ax_line = self.figure.add_subplot(grid[0, 0])
        ax_text = self.figure.add_subplot(grid[1, 0])

        ordered_rows = sorted(rows, key=lambda row: int(float(row.get("step_index", "0"))))
        counts = [int(float(row.get("active_mode_count", "0"))) for row in ordered_rows]
        deltas = [float(row.get("delta_rms_vs_full", "0")) for row in ordered_rows]
        ax_line.plot(counts, deltas, color="#2563eb", linewidth=1.2, marker="o", markersize=3)
        ax_line.set_title(title, fontsize=11)
        ax_line.set_xlabel("Active Mode Count")
        ax_line.set_ylabel("delta_rms_vs_full")
        ax_line.grid(True, alpha=0.2)

        selected_text = "Select a path step."
        if selected_row is not None:
            selected_count = int(float(selected_row.get("active_mode_count", "0")))
            selected_delta = float(selected_row.get("delta_rms_vs_full", "0"))
            ax_line.scatter([selected_count], [selected_delta], color="#dc2626", s=42, zorder=5)
            active_modes = parse_int_list_text(selected_row.get("active_modes_noll", ""))
            removed_mode = selected_row.get("removed_mode_noll", "").strip() or "None"
            selected_text = "\n".join(
                [
                    f"Selected step {selected_row.get('step_index', '')} | active_mode_count={selected_count}",
                    f"subset_rms={format_metric(float(selected_row.get('rms', '0')))} | full_fit_rms={format_metric(float(ordered_rows[0].get('rms', '0')))}",
                    f"removed_mode={removed_mode} | delta_rms_vs_full={format_metric(selected_delta)}",
                    f"active_modes={active_modes}",
                ]
            )

        ax_text.axis("off")
        ax_text.text(0.0, 1.0, selected_text, va="top", ha="left", fontsize=9, family="monospace", wrap=True)
        self.draw_idle()
        return selected_text

    def plot_mode_consistency(
        self,
        rows: list[dict[str, str]],
        selected_row: dict[str, str] | None,
        *,
        title: str,
    ) -> str:
        self.figure.clear()
        grid = self.figure.add_gridspec(2, 1, height_ratios=[1.0, 0.8])
        ax_heatmap = self.figure.add_subplot(grid[0, 0])
        ax_line = self.figure.add_subplot(grid[1, 0])

        ordered_rows = sorted(rows, key=lambda row: int(float(row.get("mode_noll", "0"))))
        metrics = [
            ("median_rank", "Median Rank"),
            ("top1_count", "Top1"),
            ("top3_count", "Top3"),
            ("top5_count", "Top5"),
            ("p95_delta_rms", "p95 dRMS"),
        ]
        matrix: list[list[float]] = []
        for field, _label in metrics:
            values = np.array([float(row.get(field, "0")) for row in ordered_rows], dtype=float)
            if values.size == 0 or np.allclose(values.max(), values.min()):
                matrix.append(np.zeros(len(ordered_rows), dtype=float).tolist())
            else:
                matrix.append(((values - values.min()) / (values.max() - values.min())).tolist())
        image = ax_heatmap.imshow(np.array(matrix), aspect="auto", cmap="magma", interpolation="nearest")
        modes = [int(float(row.get("mode_noll", "0"))) for row in ordered_rows]
        ax_heatmap.set_title(title, fontsize=11)
        ax_heatmap.set_xticks(range(len(modes)))
        ax_heatmap.set_xticklabels([str(mode) for mode in modes], rotation=90, fontsize=7)
        ax_heatmap.set_yticks(range(len(metrics)))
        ax_heatmap.set_yticklabels([label for _field, label in metrics], fontsize=8)
        self.figure.colorbar(image, ax=ax_heatmap, fraction=0.025, pad=0.02, label="Normalized metric")

        p95_values = [float(row.get("p95_delta_rms", "0")) for row in ordered_rows]
        ax_line.plot(modes, p95_values, color="#2563eb", linewidth=1.1)
        ax_line.set_xlabel("Mode (Noll)")
        ax_line.set_ylabel("p95_delta_rms")
        ax_line.grid(True, alpha=0.2)

        selected_text = "Select a mode row."
        if selected_row is not None:
            selected_mode = int(float(selected_row.get("mode_noll", "0")))
            try:
                selected_index = modes.index(selected_mode)
            except ValueError:
                selected_index = -1
            if selected_index >= 0:
                ax_heatmap.axvline(selected_index, color="#22c55e", linewidth=1.2)
            ax_line.scatter(
                [selected_mode],
                [float(selected_row.get("p95_delta_rms", "0"))],
                color="#dc2626",
                s=42,
                zorder=5,
            )
            selected_text = (
                f"Mode Z{selected_mode} | median_rank={format_metric(float(selected_row.get('median_rank', '0')))} | "
                f"top1/top3/top5={selected_row.get('top1_count', '')}/{selected_row.get('top3_count', '')}/{selected_row.get('top5_count', '')} | "
                f"p95_delta_rms={format_metric(float(selected_row.get('p95_delta_rms', '0')))}"
            )

        ax_line.text(0.01, 0.99, selected_text, transform=ax_line.transAxes, ha="left", va="top", fontsize=9)
        self.draw_idle()
        return selected_text

    def plot_global_mode_order(
        self,
        rows: list[dict[str, str]],
        *,
        prefix_count: int,
    ) -> str:
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        ordered_rows = sorted(rows, key=lambda row: int(float(row.get("global_order", "0"))))
        orders = [int(float(row.get("global_order", "0"))) for row in ordered_rows]
        modes = [int(float(row.get("mode_noll", "0"))) for row in ordered_rows]
        scores = [float(row.get("p95_delta_rms", "0")) for row in ordered_rows]
        colors = ["#2563eb" if order <= prefix_count else "#cbd5e1" for order in orders]
        axis.bar(orders, scores, color=colors)
        axis.set_title("Global Mode Order", fontsize=11)
        axis.set_xlabel("Global Removal Order")
        axis.set_ylabel("p95_delta_rms")
        axis.grid(True, axis="y", alpha=0.2)
        for order, mode, score in zip(orders, modes, scores, strict=False):
            if order <= prefix_count:
                axis.text(order, score, f"Z{mode}", rotation=90, ha="center", va="bottom", fontsize=7)
        text = f"Highlighted removal prefix: {prefix_count} modes"
        axis.text(0.01, 0.99, text, transform=axis.transAxes, ha="left", va="top", fontsize=9)
        self.draw_idle()
        return text

    def plot_global_subset_aggregate(
        self,
        rows: list[dict[str, str]],
        *,
        selected_count: int | None,
    ) -> str:
        self.figure.clear()
        axis = self.figure.add_subplot(111)
        ordered_rows = sorted(rows, key=lambda row: int(float(row.get("removed_mode_count", "0"))))
        counts = [int(float(row.get("removed_mode_count", "0"))) for row in ordered_rows]
        median_values = [float(row.get("median_delta_rms", "0")) for row in ordered_rows]
        p95_values = [float(row.get("p95_delta_rms", "0")) for row in ordered_rows]
        max_values = [float(row.get("max_delta_rms", "0")) for row in ordered_rows]
        axis.plot(counts, median_values, color="#2563eb", linewidth=1.1, label="median_delta_rms")
        axis.plot(counts, p95_values, color="#dc2626", linewidth=1.1, label="p95_delta_rms")
        axis.plot(counts, max_values, color="#059669", linewidth=1.1, label="max_delta_rms")
        detail = "Select a removed-mode count."
        if selected_count is not None:
            match = next(
                (row for row in ordered_rows if int(float(row.get("removed_mode_count", "0"))) == selected_count), None
            )
            if match is not None:
                axis.axvline(selected_count, color="#7c3aed", linewidth=1.0, alpha=0.8)
                detail = (
                    f"removed_mode_count={selected_count} | median={format_metric(float(match.get('median_delta_rms', '0')))} | "
                    f"p95={format_metric(float(match.get('p95_delta_rms', '0')))} | max={format_metric(float(match.get('max_delta_rms', '0')))}"
                )
        axis.set_title("Global Consistent Subset Aggregate", fontsize=11)
        axis.set_xlabel("Removed Mode Count")
        axis.set_ylabel("delta_rms_vs_full")
        axis.grid(True, alpha=0.2)
        axis.legend(frameon=False, fontsize=8)
        axis.text(0.01, 0.99, detail, transform=axis.transAxes, ha="left", va="top", fontsize=9)
        self.draw_idle()
        return detail

    def plot_global_subset_source(self, rows: list[dict[str, str]], selected_row: dict[str, str] | None) -> str:
        self.figure.clear()
        grid = self.figure.add_gridspec(2, 1, height_ratios=[1.2, 0.9])
        ax_line = self.figure.add_subplot(grid[0, 0])
        ax_text = self.figure.add_subplot(grid[1, 0])

        ordered_rows = sorted(rows, key=lambda row: int(float(row.get("removed_mode_count", "0"))))
        removed_counts = [int(float(row.get("removed_mode_count", "0"))) for row in ordered_rows]
        deltas = [float(row.get("delta_rms_vs_full", "0")) for row in ordered_rows]
        ax_line.plot(removed_counts, deltas, color="#2563eb", linewidth=1.2, marker="o", markersize=3)
        ax_line.set_title("Global Consistent Subset: per-source path", fontsize=11)
        ax_line.set_xlabel("Removed Mode Count")
        ax_line.set_ylabel("delta_rms_vs_full")
        ax_line.grid(True, alpha=0.2)

        selected_text = "Select a removal step."
        if selected_row is not None:
            removed_count = int(float(selected_row.get("removed_mode_count", "0")))
            ax_line.scatter([removed_count], [float(selected_row.get("delta_rms_vs_full", "0"))], color="#dc2626", s=42)
            selected_text = "\n".join(
                [
                    f"removed_mode_count={removed_count} | active_mode_count={selected_row.get('active_mode_count', '')}",
                    f"subset_rms={format_metric(float(selected_row.get('rms', '0')))} | delta_rms_vs_full={format_metric(float(selected_row.get('delta_rms_vs_full', '0')))}",
                    f"removed_modes={parse_int_list_text(selected_row.get('removed_modes_noll', ''))}",
                    f"active_modes={parse_int_list_text(selected_row.get('active_modes_noll', ''))}",
                ]
            )

        ax_text.axis("off")
        ax_text.text(0.0, 1.0, selected_text, va="top", ha="left", fontsize=9, family="monospace", wrap=True)
        self.draw_idle()
        return selected_text


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
