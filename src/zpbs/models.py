"""Shared maintained dataclasses for the ZPBS package."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SurfaceMetadata:
    """Metadata parsed from the notebook-style XYZ filename."""

    design_token: str
    design_id: str
    fea_id: str
    force_id: str
    surface_token: str
    surf_id: str


@dataclass
class FitArtifacts:
    """Full headless fit result for one source file."""

    metadata: SurfaceMetadata
    source_metadata: SurfaceMetadata
    source_file: Path
    output_coefficients_csv: Path
    points_used: int
    x: Any
    y: Any
    z: Any
    rho: Any
    phi: Any
    rho_norm: Any
    sphere_residuals_um: Any
    zpbs_to_data_surface_um: Any
    zpbs_to_data_residuals_um: Any
    zpbs_residual_surface_um: Any
    zpbs_residual_residuals_um: Any
    zpbs_residual_coefficients_um: Any
    x0_fit: float
    y0_fit: float
    z0_fit: float
    fitted_sphere_radius_um: float
    applied_reference_radius_um: float
    prefit_best_radius_um: float | None
    sphere_fit_mode: str
    center_weight: float
    sphere_sse_um2: float
    sphere_mae_um: float
    sphere_rms_um: float
    zpbs_residual_sse_um2: float
    zpbs_residual_mae_um: float
    zpbs_residual_rms_um: float
    zpbs_residual_cond: float
    zpbs_residual_on_axis_m0_um: float
    observed_aperture_radius_um: float
    norm_radius_um: float
    target_vertex_x_um: float
    target_vertex_y_um: float
    target_vertex_z_um: float
    reference_vertex_x_um: float
    reference_vertex_y_um: float
    reference_vertex_z_um: float
    vertex_mismatch_z_um: float
    vertex_um: float
    sphere_vertex_residual_um: float
    vertex_residual_um: float
    method: str
    n_modes: int
    roc_mode: str
    normalization_mode: str
    run_name: str
    common_reference_radius_um: float | None
    common_normalization_radius_um: float | None
    round_radii_um: bool
    zernike_coeff_sigfigs: int | None

    @property
    def sphere_residuals(self) -> Any:
        return self.sphere_residuals_um

    @property
    def zernike_surface(self) -> Any:
        return self.zpbs_to_data_surface_um

    @property
    def zernike_surface_residuals(self) -> Any:
        return self.zpbs_to_data_residuals_um

    @property
    def zernike_residual_surface(self) -> Any:
        return self.zpbs_residual_surface_um

    @property
    def zernike_residual_residuals(self) -> Any:
        return self.zpbs_residual_residuals_um

    @property
    def zpoly_fits(self) -> Any:
        return self.zpbs_residual_coefficients_um

    @property
    def zpoly_fits2(self) -> Any:
        return self.zpbs_residual_coefficients_um

    @property
    def sphere_sse(self) -> float:
        return self.sphere_sse_um2

    @property
    def sphere_mae(self) -> float:
        return self.sphere_mae_um

    @property
    def sphere_rms(self) -> float:
        return self.sphere_rms_um

    @property
    def sphere_residual_zernike_sse(self) -> float:
        return self.zpbs_residual_sse_um2

    @property
    def sphere_residual_zernike_mae(self) -> float:
        return self.zpbs_residual_mae_um

    @property
    def sphere_residual_zernike_rms(self) -> float:
        return self.zpbs_residual_rms_um

    @property
    def sphere_residual_zernike_cond(self) -> float:
        return self.zpbs_residual_cond


@dataclass(frozen=True)
class ProcessingInput:
    """One effective batch input after collapsing duplicate initial-state files."""

    source_file: Path
    source_metadata: SurfaceMetadata
    metadata: SurfaceMetadata


@dataclass(frozen=True)
class SpherePrefitEntry:
    """Point-cloud and robust-sphere fit data for one processed input."""

    x: Any
    y: Any
    z: Any
    raw_aperture_radius_um: float
    x0_fit: float
    y0_fit: float
    z0_fit: float
    best_radius_um: float
    best_sphere_residuals: Any
    observed_aperture_radius_um: float
    sphere_fit_mode: str
    center_weight: float
    target_vertex_x_um: float
    target_vertex_y_um: float
    target_vertex_z_um: float
    reference_vertex_x_um: float
    reference_vertex_y_um: float
    reference_vertex_z_um: float
    vertex_mismatch_z_um: float


@dataclass(frozen=True)
class VertexTarget:
    """Chosen target vertex from the raw measured point cloud."""

    index: int
    x_um: float
    y_um: float
    z_um: float
    rho_um: float


@dataclass(frozen=True)
class SphereReferenceFit:
    """Resolved sphere reference used for residual formation and diagnostics."""

    x0_fit: float
    y0_fit: float
    z0_fit: float
    radius_um: float
    residuals: Any
    sphere_fit_mode: str
    center_weight: float
    target_vertex_x_um: float
    target_vertex_y_um: float
    target_vertex_z_um: float
    reference_vertex_x_um: float
    reference_vertex_y_um: float
    reference_vertex_z_um: float
    vertex_mismatch_z_um: float
