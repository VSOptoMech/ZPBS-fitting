from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.figure import Figure

from ..common import uses_posterior_sign_convention
from ..models import FitArtifacts
from ..reporting.batch_reports import radial_bin_profile, sphere_profile_z
from .support import (
    METHOD_LABELS,
    NORMALIZATION_MODE_LABELS,
    ROC_MODE_LABELS,
    SPHERE_FIT_MODE_LABELS,
    display_label,
    format_metric,
    snapped_axis_limits,
)


def _render_detailed_analysis(
    figure: Figure,
    *,
    artifacts: FitArtifacts,
    coeff_meta: dict[str, str],
    coeffs: list[tuple[str, float]],
    source_file: Path,
    rho_axis_limit_um: float | None = None,
) -> tuple[str, dict[str, str]]:
    """Render the detailed four-panel analysis view for one fitted file."""
    top_coeffs = sorted(coeffs[1:], key=lambda item: abs(item[1]), reverse=True)[:8]
    method_label = display_label(METHOD_LABELS, artifacts.method)
    roc_mode_label = display_label(ROC_MODE_LABELS, artifacts.roc_mode)
    normalization_mode_label = display_label(NORMALIZATION_MODE_LABELS, artifacts.normalization_mode)
    sphere_fit_mode_label = display_label(SPHERE_FIT_MODE_LABELS, artifacts.sphere_fit_mode)

    sphere_z = sphere_profile_z(
        artifacts.rho,
        z0_fit=artifacts.z0_fit,
        radius_um=artifacts.applied_reference_radius_um,
        posterior_surface=uses_posterior_sign_convention(artifacts.metadata.surf_id),
        np=np,
    )
    rho_meas, z_meas = radial_bin_profile(artifacts.rho, artifacts.z, bins=64, np=np)
    rho_sphere, z_sphere = radial_bin_profile(artifacts.rho, sphere_z, bins=64, np=np)
    rho_fit, z_fit = radial_bin_profile(artifacts.rho, artifacts.zernike_surface, bins=64, np=np)
    rho_resid, z_resid = radial_bin_profile(artifacts.rho, artifacts.zernike_surface_residuals, bins=64, np=np)
    rho_sphere_resid, z_sphere_resid = radial_bin_profile(artifacts.rho, artifacts.sphere_residuals, bins=64, np=np)

    sample_stride = max(1, len(artifacts.x) // 3500)
    figure.clear()
    grid = figure.add_gridspec(2, 2, width_ratios=[1.2, 1.0], height_ratios=[1.1, 1.0])
    ax_profile = figure.add_subplot(grid[0, 0])
    ax_map = figure.add_subplot(grid[0, 1])
    ax_residual = figure.add_subplot(grid[1, 0])
    ax_sphere_residual = figure.add_subplot(grid[1, 1])

    ax_profile.plot(rho_meas, z_meas, color="#111827", linewidth=1.2, label="Measured")
    ax_profile.plot(rho_sphere, z_sphere, color="#2563eb", linewidth=1.0, label="Sphere")
    ax_profile.plot(rho_fit, z_fit, color="#dc2626", linewidth=1.0, label="Zernike")
    ax_profile.set_title("Radial Profile vs Fit", fontsize=10)
    ax_profile.set_xlabel("rho (um)")
    ax_profile.set_ylabel("z (um)")
    if rho_axis_limit_um is not None:
        ax_profile.set_xlim(0.0, rho_axis_limit_um)
    ax_profile.grid(True, alpha=0.2)
    ax_profile.legend(fontsize=8, frameon=False)

    scatter = ax_map.scatter(
        artifacts.x[::sample_stride],
        artifacts.y[::sample_stride],
        c=artifacts.z[::sample_stride],
        s=7,
        cmap="viridis",
        linewidths=0,
    )
    ax_map.set_title("Measured Surface Map", fontsize=10)
    ax_map.set_xlabel("x (um)")
    ax_map.set_ylabel("y (um)")
    ax_map.grid(True, alpha=0.2)
    figure.colorbar(scatter, ax=ax_map, fraction=0.046, pad=0.04, label="z (um)")

    ax_residual.plot(rho_resid, z_resid, color="#059669", linewidth=1.0)
    ax_residual.axhline(0.0, color="#6b7280", linewidth=0.8, alpha=0.8)
    ax_residual.set_title("Zernike Residual vs Radius", fontsize=10)
    ax_residual.set_xlabel("rho (um)")
    ax_residual.set_ylabel("Residual (um)")
    if rho_axis_limit_um is not None:
        ax_residual.set_xlim(0.0, rho_axis_limit_um)
    zernike_y_min, zernike_y_max, zernike_step = snapped_axis_limits(artifacts.zernike_surface_residuals)
    ax_residual.set_ylim(zernike_y_min, zernike_y_max)
    ax_residual.set_yticks(np.arange(zernike_y_min, zernike_y_max + (0.5 * zernike_step), zernike_step))
    ax_residual.grid(True, alpha=0.2)

    ax_sphere_residual.plot(rho_sphere_resid, z_sphere_resid, color="#7c3aed", linewidth=1.0)
    ax_sphere_residual.axhline(0.0, color="#6b7280", linewidth=0.8, alpha=0.8)
    ax_sphere_residual.set_title("Sphere Fit Residual vs Radius", fontsize=10)
    ax_sphere_residual.set_xlabel("rho (um)")
    ax_sphere_residual.set_ylabel("Residual (um)")
    if rho_axis_limit_um is not None:
        ax_sphere_residual.set_xlim(0.0, rho_axis_limit_um)
    sphere_y_min, sphere_y_max, sphere_step = snapped_axis_limits(artifacts.sphere_residuals)
    ax_sphere_residual.set_ylim(sphere_y_min, sphere_y_max)
    ax_sphere_residual.set_yticks(np.arange(sphere_y_min, sphere_y_max + (0.5 * sphere_step), sphere_step))
    ax_sphere_residual.grid(True, alpha=0.2)

    coeff_lines_list = [
        "    ".join(f"{name}={value:.2e}" for name, value in top_coeffs[index : index + 2])
        for index in range(0, len(top_coeffs), 2)
    ]
    coeff_lines = "\n".join(coeff_lines_list) or "No coefficients"
    text = "\n".join(
        [
            f"File: {source_file.name}",
            f"Surface: {artifacts.metadata.surf_id} ({artifacts.metadata.surface_token})",
            f"Force: {artifacts.metadata.force_id}",
            f"Method: {method_label}",
            f"ROC mode: {roc_mode_label}",
            f"Sphere fit mode: {sphere_fit_mode_label}",
            f"Center weight: {artifacts.center_weight:.2f}",
            f"Norm mode: {normalization_mode_label}",
            f"Round radii: {'on' if artifacts.round_radii_um else 'off'}",
            (
                f"Coeff precision: {artifacts.zernike_coeff_sigfigs} sig figs"
                if artifacts.zernike_coeff_sigfigs
                else "Coeff precision: full"
            ),
            f"Fitted sphere radius: {artifacts.fitted_sphere_radius_um:.2f} um",
            f"Sphere center: ({artifacts.x0_fit:.2f}, {artifacts.y0_fit:.2f}, {artifacts.z0_fit:.2f}) um",
            (
                "Target vertex: "
                f"({artifacts.target_vertex_x_um:.2f}, {artifacts.target_vertex_y_um:.2f}, {artifacts.target_vertex_z_um:.2f}) um"
            ),
            (
                "Reference vertex: "
                f"({artifacts.reference_vertex_x_um:.2f}, {artifacts.reference_vertex_y_um:.2f}, {artifacts.reference_vertex_z_um:.2f}) um"
            ),
            f"Vertex mismatch z: {artifacts.vertex_mismatch_z_um:.2e} um",
            f"Applied norm radius: {artifacts.norm_radius_um:.2f} um",
            f"Observed aperture radius: {artifacts.observed_aperture_radius_um:.2f} um",
            f"Sphere SSE: {artifacts.sphere_sse:.2e}",
            f"Surface RMS: {artifacts.surface_zernike_rms:.2e}",
            f"Residual RMS: {artifacts.sphere_residual_zernike_rms:.2e}",
            f"Coeff file: {artifacts.output_coefficients_csv.name}",
            f"Coeff metadata radius: {coeff_meta.get('Norm. Radius (mm)', 'n/a')} mm",
            "",
            "Top coeffs:",
            coeff_lines,
        ]
    )

    figure.suptitle(
        f"{artifacts.run_name} | {artifacts.metadata.force_id} | {artifacts.metadata.surface_token}",
        fontsize=11,
    )

    details = {
        "source_file": str(source_file),
        "coeff_file": str(artifacts.output_coefficients_csv),
        "sphere_sse": format_metric(artifacts.sphere_sse, precision=2),
        "surface_rms": format_metric(artifacts.surface_zernike_rms, precision=2),
        "residual_rms": format_metric(artifacts.sphere_residual_zernike_rms, precision=2),
        "sphere_fit_mode": artifacts.sphere_fit_mode,
        "center_weight": f"{artifacts.center_weight:.2f}",
        "applied_norm_radius_um": f"{artifacts.norm_radius_um:.2f}",
        "observed_aperture_radius_um": f"{artifacts.observed_aperture_radius_um:.2f}",
    }
    return text, details


def render_detailed_analysis_figure(
    figure: Figure,
    *,
    artifacts: FitArtifacts,
    coeff_meta: dict[str, str],
    coeffs: list[tuple[str, float]],
    source_file: Path | None = None,
    rho_axis_limit_um: float | None = None,
) -> tuple[str, dict[str, str]]:
    """Populate a Matplotlib figure with the detailed four-panel analysis view."""
    return _render_detailed_analysis(
        figure,
        artifacts=artifacts,
        coeff_meta=coeff_meta,
        coeffs=coeffs,
        source_file=source_file or artifacts.source_file,
        rho_axis_limit_um=rho_axis_limit_um,
    )


def write_detailed_analysis_plot(
    file_path: Path,
    *,
    artifacts: FitArtifacts,
    coeff_meta: dict[str, str],
    coeffs: list[tuple[str, float]],
    source_file: Path | None = None,
    rho_axis_limit_um: float | None = None,
) -> tuple[str, dict[str, str]]:
    """Write a saved detailed analysis image for one fitted file."""
    figure = Figure(figsize=(9.6, 7.4), constrained_layout=True)
    text, details = render_detailed_analysis_figure(
        figure,
        artifacts=artifacts,
        coeff_meta=coeff_meta,
        coeffs=coeffs,
        source_file=source_file,
        rho_axis_limit_um=rho_axis_limit_um,
    )
    file_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(file_path, bbox_inches="tight", dpi=150)
    return text, details
