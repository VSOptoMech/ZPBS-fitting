"""Shared helpers for optional Zemax-facing vertex tilt correction."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..azp_csv_pipeline import N_ZERNIKE_TERMS, build_zernike_coefficients_rows, zernike_polar_basis
from ..common import format_tension, round_sigfigs_array, signed_sphere_radius_um, uses_posterior_sign_convention
from ..models import FitArtifacts
from .surface_fit import zpbs_residual_on_axis_m0_um


@dataclass(frozen=True)
class VertexTiltCorrection:
    """Summary of the in-memory vertex tilt correction applied to a fit artifact."""

    original_x_mrad: float
    original_y_mrad: float
    corrected_x_mrad: float
    corrected_y_mrad: float
    delta_z2_um: float
    delta_z3_um: float

    @property
    def original_magnitude_mrad(self) -> float:
        return float(np.hypot(self.original_x_mrad, self.original_y_mrad))

    @property
    def corrected_magnitude_mrad(self) -> float:
        return float(np.hypot(self.corrected_x_mrad, self.corrected_y_mrad))


_CENTER_GRADIENT_TERMS = (
    # zero-based coefficient index, center derivative scale with respect to normalized x/y, axis
    (1, 2.0, "x"),  # Z2
    (2, 2.0, "y"),  # Z3
    (6, -2.0 * np.sqrt(8.0), "y"),  # Z7
    (7, -2.0 * np.sqrt(8.0), "x"),  # Z8
    (15, 3.0 * np.sqrt(12.0), "x"),  # Z16
    (16, 3.0 * np.sqrt(12.0), "y"),  # Z17
    (28, -16.0, "y"),  # Z29
    (29, -16.0, "x"),  # Z30
)


def center_gradient_mrad(coefficients_um: np.ndarray, norm_radius_um: float) -> tuple[float, float]:
    """Return the net center gradient encoded by maintained m=1 terms."""
    norm = float(norm_radius_um)
    if norm <= 0.0:
        raise ValueError("Normalization radius must be positive to compute vertex tilt correction.")
    coeffs = np.asarray(coefficients_um, dtype=float)
    gx = 0.0
    gy = 0.0
    for index, scale, axis in _CENTER_GRADIENT_TERMS:
        if index >= len(coeffs):
            continue
        contribution = float(coeffs[index]) * float(scale) / norm
        if axis == "x":
            gx += contribution
        else:
            gy += contribution
    return gx * 1000.0, gy * 1000.0


def zero_vertex_tilt_coefficients(
    coefficients_um: np.ndarray,
    norm_radius_um: float,
) -> tuple[np.ndarray, VertexTiltCorrection]:
    """Cancel net center slope by applying the correction to Z2/Z3 only."""
    coeffs = np.asarray(coefficients_um, dtype=float).copy()
    original_x_mrad, original_y_mrad = center_gradient_mrad(coeffs, norm_radius_um)
    original_x = original_x_mrad / 1000.0
    original_y = original_y_mrad / 1000.0
    delta_z2_um = -original_x * float(norm_radius_um) / 2.0
    delta_z3_um = -original_y * float(norm_radius_um) / 2.0
    if len(coeffs) >= 2:
        coeffs[1] += delta_z2_um
    if len(coeffs) >= 3:
        coeffs[2] += delta_z3_um
    corrected_x_mrad, corrected_y_mrad = center_gradient_mrad(coeffs, norm_radius_um)
    return coeffs, VertexTiltCorrection(
        original_x_mrad=original_x_mrad,
        original_y_mrad=original_y_mrad,
        corrected_x_mrad=corrected_x_mrad,
        corrected_y_mrad=corrected_y_mrad,
        delta_z2_um=delta_z2_um,
        delta_z3_um=delta_z3_um,
    )


def apply_vertex_tilt_correction_to_artifacts(artifacts: FitArtifacts) -> tuple[FitArtifacts, VertexTiltCorrection]:
    """Return a FitArtifacts copy with zero center slope in the residual Zernike model."""
    adjusted_coeffs, correction = zero_vertex_tilt_coefficients(
        np.asarray(artifacts.zpbs_residual_coefficients_um, dtype=float),
        artifacts.norm_radius_um,
    )
    rho_norm = np.asarray(artifacts.rho_norm, dtype=float)
    phi = np.asarray(artifacts.phi, dtype=float)
    basis = zernike_polar_basis(rho_norm, phi, n_modes=N_ZERNIKE_TERMS)
    zpbs_residual_surface_um = basis @ adjusted_coeffs
    sphere_residuals_um = np.asarray(artifacts.sphere_residuals_um, dtype=float)
    zpbs_residual_residuals_um = sphere_residuals_um - zpbs_residual_surface_um
    residual_sign = -1.0 if uses_posterior_sign_convention(artifacts.metadata.surf_id) else 1.0
    zpbs_to_data_residuals_um = residual_sign * zpbs_residual_residuals_um
    z_values = np.asarray(artifacts.z, dtype=float)
    zpbs_to_data_surface_um = z_values - zpbs_to_data_residuals_um
    zpbs_residual_sse_um2 = float(np.sum(np.square(zpbs_residual_residuals_um)))
    zpbs_residual_mae_um = float(np.mean(np.abs(zpbs_residual_residuals_um)))
    zpbs_residual_rms_um = float(np.sqrt(np.mean(np.square(zpbs_residual_residuals_um))))
    idx_center = int(np.argmin(np.asarray(artifacts.rho, dtype=float))) if len(np.asarray(artifacts.rho)) else 0

    corrected = replace(
        artifacts,
        zpbs_residual_coefficients_um=adjusted_coeffs,
        zpbs_residual_surface_um=zpbs_residual_surface_um,
        zpbs_residual_residuals_um=zpbs_residual_residuals_um,
        zpbs_residual_sse_um2=zpbs_residual_sse_um2,
        zpbs_residual_mae_um=zpbs_residual_mae_um,
        zpbs_residual_rms_um=zpbs_residual_rms_um,
        zpbs_residual_on_axis_m0_um=zpbs_residual_on_axis_m0_um(adjusted_coeffs),
        zpbs_to_data_surface_um=zpbs_to_data_surface_um,
        zpbs_to_data_residuals_um=zpbs_to_data_residuals_um,
        vertex_um=float(zpbs_to_data_surface_um[idx_center]) if len(zpbs_to_data_surface_um) else artifacts.vertex_um,
        vertex_residual_um=float(zpbs_to_data_residuals_um[idx_center])
        if len(zpbs_to_data_residuals_um)
        else artifacts.vertex_residual_um,
    )
    return corrected, correction


def export_coefficient_rows_for_artifacts(artifacts: FitArtifacts) -> list[tuple[str, str]]:
    """Build displayed/exportable coefficient rows from one FitArtifacts object."""
    export_coefficients_um = np.asarray(artifacts.zpbs_residual_coefficients_um, dtype=float).copy()
    export_coefficients_um[0] = export_coefficients_um[0] - artifacts.sphere_vertex_residual_um
    if artifacts.zernike_coeff_sigfigs is not None:
        export_coefficients_um = round_sigfigs_array(export_coefficients_um, artifacts.zernike_coeff_sigfigs, np=np)
    export_sign = 1.0 if uses_posterior_sign_convention(artifacts.metadata.surf_id) else -1.0
    export_coefficients_mm = export_sign * export_coefficients_um / 1000.0
    signed_roc_um = signed_sphere_radius_um(
        artifacts.applied_reference_radius_um,
        reference_vertex_z_um=artifacts.reference_vertex_z_um,
        z0_fit=artifacts.z0_fit,
    )
    rows = build_zernike_coefficients_rows(
        design_id=artifacts.metadata.design_id,
        design_token=artifacts.metadata.design_token,
        fea_id=artifacts.metadata.fea_id,
        surf_id=artifacts.metadata.surf_id,
        tension_mn=format_tension(artifacts.metadata.force_id),
        base_sphere_roc_um=signed_roc_um,
        vertex_um=artifacts.vertex_um,
        vertex_residual_um=artifacts.vertex_residual_um,
        norm_radius_um=artifacts.norm_radius_um,
        zernike_coefficients_mm=export_coefficients_mm,
    )
    return [(str(name), str(value)) for name, value in rows]


def split_coefficient_rows(rows: list[tuple[str, str]]) -> tuple[dict[str, str], list[tuple[str, float]]]:
    """Split two-column coefficient rows into metadata and numeric Z terms."""
    metadata: dict[str, str] = {}
    coeffs: list[tuple[str, float]] = []
    for name, value in rows:
        if name.startswith("Z") and name[1:].isdigit():
            coeffs.append((name, float(value)))
        else:
            metadata[name] = value
    return metadata, coeffs
