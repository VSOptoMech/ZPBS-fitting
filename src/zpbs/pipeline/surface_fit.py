"""Per-surface and batch-level maintained fitting pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from zpbs.azp_csv_pipeline import (
    _pad_coeffs_to_45,
    cartesian_to_polar,
    export_zernike_coefficients_csv,
    fit_zernike_lstsq,
    zernike_polar,
    zernike_polar_basis,
)
from zpbs.common import (
    clamp_normalization_radius_um,
    clamp_reference_radius_um,
    format_tension,
    make_output_filename,
    round_nearest_micrometer,
    round_sigfigs_array,
    uses_posterior_sign_convention,
    validate_center_weight,
    validate_zernike_method,
)
from zpbs.fit.sphere_reference import (
    SphereReferenceFit,
    VertexTarget,
    fit_sphere_robust,
    fit_sphere_with_fixed_radius,
    get_sphere_prefit_entry,
)
from zpbs.models import FitArtifacts, ProcessingInput, SpherePrefitEntry


_ON_AXIS_M0_ZERO_BASED = (0, 3, 10, 21, 36)
_ON_AXIS_M0_WEIGHTS = (
    1.0,
    -float(np.sqrt(3.0)),
    float(np.sqrt(5.0)),
    -float(np.sqrt(7.0)),
    3.0,
)


def zpbs_residual_on_axis_m0_um(coefficients_um: np.ndarray) -> float:
    """Evaluate the residual-stage m=0 contribution at rho=0 in the maintained Noll basis."""
    coeffs = np.asarray(coefficients_um, dtype=float)
    total = 0.0
    for index, weight in zip(_ON_AXIS_M0_ZERO_BASED, _ON_AXIS_M0_WEIGHTS, strict=True):
        if index < len(coeffs):
            total += float(coeffs[index]) * weight
    return float(total)


def run_fit_pipeline(
    x: list[float],
    y: list[float],
    z: list[float],
    *,
    surf_id: str,
    method: str = "lstsq",
    n_modes: int,
    rcond: float | None,
    reference_radius_um: float | None,
    normalization_radius_um: float | None,
    sphere_fit_mode: str = "center_weighted",
    center_weight: float = 0.5,
    round_radii_um: bool = False,
    zernike_coeff_sigfigs: int | None = None,
    prefit_data: SpherePrefitEntry | None = None,
) -> dict[str, Any]:
    """Run the headless sphere-plus-Zernike workflow with optional common radius."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    z_arr = np.asarray(z, dtype=float)
    if len(x_arr) != len(y_arr) or len(y_arr) != len(z_arr):
        raise ValueError("x, y, z must have equal length.")
    if len(x_arr) == 0:
        raise ValueError("No data points provided.")
    center_weight = validate_center_weight(center_weight)
    method_key = validate_zernike_method(method)
    if zernike_coeff_sigfigs is not None and zernike_coeff_sigfigs < 1:
        raise ValueError("--zernike-coeff-sigfigs must be positive.")

    raw_aperture_radius_um = float(np.max(np.sqrt(x_arr**2 + y_arr**2)))
    if prefit_data is None:
        sphere_prefit = fit_sphere_robust(
            x_arr,
            y_arr,
            z_arr,
            sphere_fit_mode=sphere_fit_mode,
            center_weight=center_weight,
        )
        prefit_best_radius_um = float(sphere_prefit.radius_um)
    else:
        sphere_prefit = SphereReferenceFit(
            x0_fit=float(prefit_data.x0_fit),
            y0_fit=float(prefit_data.y0_fit),
            z0_fit=float(prefit_data.z0_fit),
            radius_um=float(prefit_data.best_radius_um),
            residuals=np.asarray(prefit_data.best_sphere_residuals, dtype=float),
            sphere_fit_mode=str(prefit_data.sphere_fit_mode),
            center_weight=float(prefit_data.center_weight),
            target_vertex_x_um=float(prefit_data.target_vertex_x_um),
            target_vertex_y_um=float(prefit_data.target_vertex_y_um),
            target_vertex_z_um=float(prefit_data.target_vertex_z_um),
            reference_vertex_x_um=float(prefit_data.reference_vertex_x_um),
            reference_vertex_y_um=float(prefit_data.reference_vertex_y_um),
            reference_vertex_z_um=float(prefit_data.reference_vertex_z_um),
            vertex_mismatch_z_um=float(prefit_data.vertex_mismatch_z_um),
        )
        prefit_best_radius_um = float(prefit_data.best_radius_um)
        sphere_fit_mode = sphere_prefit.sphere_fit_mode
        center_weight = sphere_prefit.center_weight

    if reference_radius_um is None:
        if round_radii_um:
            applied_radius_um = clamp_reference_radius_um(
                round_nearest_micrometer(prefit_best_radius_um),
                raw_aperture_radius_um,
            )
            target_vertex = VertexTarget(
                index=-1,
                x_um=sphere_prefit.target_vertex_x_um,
                y_um=sphere_prefit.target_vertex_y_um,
                z_um=sphere_prefit.target_vertex_z_um,
                rho_um=0.0,
            )
            surface_branch_sign = 1.0 if sphere_prefit.reference_vertex_z_um >= sphere_prefit.z0_fit else -1.0
            sphere_fit = fit_sphere_with_fixed_radius(
                x_arr,
                y_arr,
                z_arr,
                radius_um=applied_radius_um,
                initial_guess_center=(
                    float(sphere_prefit.x0_fit),
                    float(sphere_prefit.y0_fit),
                    float(sphere_prefit.z0_fit),
                ),
                sphere_fit_mode=sphere_fit_mode,
                center_weight=center_weight,
                target_vertex=target_vertex,
                surface_branch_sign=surface_branch_sign,
            )
        else:
            applied_radius_um = prefit_best_radius_um
            sphere_fit = SphereReferenceFit(
                x0_fit=float(sphere_prefit.x0_fit),
                y0_fit=float(sphere_prefit.y0_fit),
                z0_fit=float(sphere_prefit.z0_fit),
                radius_um=float(sphere_prefit.radius_um),
                residuals=np.asarray(sphere_prefit.residuals, dtype=float),
                sphere_fit_mode=sphere_prefit.sphere_fit_mode,
                center_weight=sphere_prefit.center_weight,
                target_vertex_x_um=sphere_prefit.target_vertex_x_um,
                target_vertex_y_um=sphere_prefit.target_vertex_y_um,
                target_vertex_z_um=sphere_prefit.target_vertex_z_um,
                reference_vertex_x_um=sphere_prefit.reference_vertex_x_um,
                reference_vertex_y_um=sphere_prefit.reference_vertex_y_um,
                reference_vertex_z_um=sphere_prefit.reference_vertex_z_um,
                vertex_mismatch_z_um=sphere_prefit.vertex_mismatch_z_um,
            )
    else:
        requested_radius_um = float(reference_radius_um)
        if round_radii_um:
            requested_radius_um = clamp_reference_radius_um(
                round_nearest_micrometer(requested_radius_um),
                raw_aperture_radius_um,
            )
        target_vertex = VertexTarget(
            index=-1,
            x_um=sphere_prefit.target_vertex_x_um,
            y_um=sphere_prefit.target_vertex_y_um,
            z_um=sphere_prefit.target_vertex_z_um,
            rho_um=0.0,
        )
        surface_branch_sign = 1.0 if sphere_prefit.reference_vertex_z_um >= sphere_prefit.z0_fit else -1.0
        sphere_fit = fit_sphere_with_fixed_radius(
            x_arr,
            y_arr,
            z_arr,
            radius_um=requested_radius_um,
            initial_guess_center=(
                float(sphere_prefit.x0_fit),
                float(sphere_prefit.y0_fit),
                float(sphere_prefit.z0_fit),
            ),
            sphere_fit_mode=sphere_fit_mode,
            center_weight=center_weight,
            target_vertex=target_vertex,
            surface_branch_sign=surface_branch_sign,
        )
        applied_radius_um = requested_radius_um

    x0_fit = float(sphere_fit.x0_fit)
    y0_fit = float(sphere_fit.y0_fit)
    z0_fit = float(sphere_fit.z0_fit)
    rho, phi = cartesian_to_polar(x_arr - x0_fit, y_arr - y0_fit)
    observed_rho_max = float(np.max(rho))
    if observed_rho_max <= 0:
        raise ValueError("Invalid polar radius max; all points may be coincident.")

    # Rebuild the actual fitted sphere branch in measured-z coordinates so stage metrics
    # and the combined ZPBS-to-data reconstruction do not depend on the surface family token.
    branch_sign = 1.0 if float(sphere_fit.reference_vertex_z_um) >= z0_fit else -1.0
    term = np.sqrt(np.clip(float(sphere_fit.radius_um) ** 2 - np.square(rho), 0.0, None))
    sphere_surface_um = z0_fit + (branch_sign * term)
    sphere_residuals_z = z_arr - sphere_surface_um
    residual_sign = -1.0 if uses_posterior_sign_convention(surf_id) else 1.0
    sphere_residuals = residual_sign * sphere_residuals_z

    sphere_sse_um2 = float(np.sum(np.square(sphere_residuals)))
    sphere_mae_um = float(np.mean(np.abs(sphere_residuals)))
    sphere_rms_um = float(np.sqrt(np.mean(np.square(sphere_residuals))))

    requested_norm_radius_um = observed_rho_max if normalization_radius_um is None else float(normalization_radius_um)
    if round_radii_um:
        norm_radius_um = clamp_normalization_radius_um(
            round_nearest_micrometer(requested_norm_radius_um),
            observed_rho_max,
        )
    else:
        norm_radius_um = requested_norm_radius_um
        if norm_radius_um < observed_rho_max:
            raise ValueError(
                f"Normalization radius {norm_radius_um} um is smaller than the observed aperture radius {observed_rho_max} um."
            )

    rho_norm = rho / norm_radius_um
    pol_loci = np.stack((rho_norm, phi), axis=1)
    idx_center = int(np.argmin(rho))
    zv = float(z_arr[idx_center])
    sphere_vertex_residual_um = float(sphere_residuals[idx_center])
    zpbs_residual_coefficients_raw, _, _, _, residual_cond = fit_zernike_lstsq(
        rho_norm, phi, sphere_residuals, n_modes=n_modes, rcond=rcond
    )

    zpbs_residual_coefficients_um = _pad_coeffs_to_45(np.asarray(zpbs_residual_coefficients_raw, dtype=float))
    if zernike_coeff_sigfigs is not None:
        zpbs_residual_coefficients_um = round_sigfigs_array(zpbs_residual_coefficients_um, zernike_coeff_sigfigs, np=np)

    _ = zernike_polar_basis
    zpbs_residual_surface_um = zernike_polar(pol_loci, *zpbs_residual_coefficients_um)
    zpbs_residual_residuals_um = sphere_residuals - zpbs_residual_surface_um
    zpbs_residual_sse_um2 = float(np.sum(np.square(zpbs_residual_residuals_um)))
    zpbs_residual_mae_um = float(np.mean(np.abs(zpbs_residual_residuals_um)))
    zpbs_residual_rms_um = float(np.sqrt(np.mean(np.square(zpbs_residual_residuals_um))))
    zpbs_residual_on_axis_m0_value_um = zpbs_residual_on_axis_m0_um(zpbs_residual_coefficients_um)

    zpbs_to_data_surface_um = sphere_surface_um + (residual_sign * zpbs_residual_surface_um)
    zpbs_to_data_residuals_um = z_arr - zpbs_to_data_surface_um
    vertex_fit_um = float(zpbs_to_data_surface_um[idx_center])
    vertex_residual_um = float(zpbs_to_data_residuals_um[idx_center])

    return {
        "x": x_arr,
        "y": y_arr,
        "z": z_arr,
        "rho": rho,
        "phi": phi,
        "rho_norm": rho_norm,
        "x0_fit": float(x0_fit),
        "y0_fit": float(y0_fit),
        "z0_fit": float(z0_fit),
        "fitted_sphere_radius_um": applied_radius_um,
        "applied_reference_radius_um": applied_radius_um,
        "prefit_best_radius_um": prefit_best_radius_um,
        "sphere_fit_mode": sphere_fit.sphere_fit_mode,
        "center_weight": sphere_fit.center_weight,
        "sphere_residuals_um": sphere_residuals,
        "sphere_sse_um2": sphere_sse_um2,
        "sphere_mae_um": sphere_mae_um,
        "sphere_rms_um": sphere_rms_um,
        "zv": zv,
        "zv2": sphere_vertex_residual_um,
        "vertex_fit_um": vertex_fit_um,
        "sphere_vertex_residual_um": sphere_vertex_residual_um,
        "vertex_residual_um": vertex_residual_um,
        "zpbs_residual_coefficients_um": zpbs_residual_coefficients_um,
        "zpbs_residual_surface_um": zpbs_residual_surface_um,
        "zpbs_residual_residuals_um": zpbs_residual_residuals_um,
        "zpbs_residual_sse_um2": zpbs_residual_sse_um2,
        "zpbs_residual_mae_um": zpbs_residual_mae_um,
        "zpbs_residual_rms_um": zpbs_residual_rms_um,
        "zpbs_residual_cond": float(residual_cond),
        "zpbs_residual_on_axis_m0_um": zpbs_residual_on_axis_m0_value_um,
        "zpbs_to_data_surface_um": zpbs_to_data_surface_um,
        "zpbs_to_data_residuals_um": zpbs_to_data_residuals_um,
        "observed_aperture_radius_um": observed_rho_max,
        "norm_radius_um": norm_radius_um,
        "target_vertex_x_um": sphere_fit.target_vertex_x_um,
        "target_vertex_y_um": sphere_fit.target_vertex_y_um,
        "target_vertex_z_um": sphere_fit.target_vertex_z_um,
        "reference_vertex_x_um": sphere_fit.reference_vertex_x_um,
        "reference_vertex_y_um": sphere_fit.reference_vertex_y_um,
        "reference_vertex_z_um": sphere_fit.reference_vertex_z_um,
        "vertex_mismatch_z_um": sphere_fit.vertex_mismatch_z_um,
        "method": method_key,
        "n_modes": n_modes,
        "round_radii_um": round_radii_um,
        "zernike_coeff_sigfigs": zernike_coeff_sigfigs,
    }


def build_fit_artifacts(
    file_path: Path,
    *,
    metadata: Any | None,
    source_metadata: Any | None,
    output_dir: Path,
    method: str = "lstsq",
    n_modes: int,
    rcond: float | None,
    roc_mode: str,
    reference_radius_um: float | None,
    sphere_fit_mode: str,
    center_weight: float,
    normalization_mode: str,
    normalization_radius_um: float | None,
    run_name: str,
    common_reference_radius_um: float | None,
    common_normalization_radius_um: float | None,
    round_radii_um: bool,
    zernike_coeff_sigfigs: int | None,
    prefit_data: SpherePrefitEntry | None = None,
) -> FitArtifacts:
    """Run one fit and package all data needed for CSV and HDF5 outputs."""
    from zpbs.io.xyz import parse_surface_metadata

    metadata = parse_surface_metadata(file_path) if metadata is None else metadata
    source_metadata = parse_surface_metadata(file_path) if source_metadata is None else source_metadata
    if prefit_data is None:
        prefit_data = get_sphere_prefit_entry(
            file_path,
            surf_id=metadata.surf_id,
            sphere_fit_mode=sphere_fit_mode,
            center_weight=center_weight,
        )

    fit_data = run_fit_pipeline(
        prefit_data.x,
        prefit_data.y,
        prefit_data.z,
        surf_id=metadata.surf_id,
        method=method,
        n_modes=n_modes,
        rcond=rcond,
        reference_radius_um=reference_radius_um,
        normalization_radius_um=normalization_radius_um,
        sphere_fit_mode=sphere_fit_mode,
        center_weight=center_weight,
        round_radii_um=round_radii_um,
        zernike_coeff_sigfigs=zernike_coeff_sigfigs,
        prefit_data=prefit_data,
    )

    export_coefficients_um = fit_data["zpbs_residual_coefficients_um"].copy()
    export_coefficients_um[0] = export_coefficients_um[0] - fit_data["sphere_vertex_residual_um"]
    if zernike_coeff_sigfigs is not None:
        export_coefficients_um = round_sigfigs_array(export_coefficients_um, zernike_coeff_sigfigs, np=np)
    # The existing CSV convention already negates the anterior export. Posterior surfaces
    # need one extra sign inversion here so the written coefficients are Zemax-ready.
    export_sign = 1.0 if uses_posterior_sign_convention(metadata.surf_id) else -1.0
    export_coefficients_mm = export_sign * export_coefficients_um / 1000.0
    signed_roc_um = (1.0 if float(fit_data["reference_vertex_z_um"]) >= float(fit_data["z0_fit"]) else -1.0) * float(
        fit_data["applied_reference_radius_um"]
    )

    coeff_dir = output_dir / "coefficients"
    coeff_path = coeff_dir / make_output_filename(metadata)
    export_zernike_coefficients_csv(
        coeff_path,
        design_id=metadata.design_id,
        fea_id=metadata.fea_id,
        surf_id=metadata.surf_id,
        tension_mn=format_tension(metadata.force_id),
        base_sphere_roc_um=signed_roc_um,
        vertex_um=fit_data["vertex_fit_um"],
        vertex_residual_um=fit_data["vertex_residual_um"],
        norm_radius_um=fit_data["norm_radius_um"],
        zernike_coefficients_mm=export_coefficients_mm,
    )

    return FitArtifacts(
        metadata=metadata,
        source_metadata=source_metadata,
        source_file=file_path,
        output_coefficients_csv=coeff_path,
        points_used=len(fit_data["x"]),
        x=fit_data["x"],
        y=fit_data["y"],
        z=fit_data["z"],
        rho=fit_data["rho"],
        phi=fit_data["phi"],
        rho_norm=fit_data["rho_norm"],
        sphere_residuals_um=fit_data["sphere_residuals_um"],
        zpbs_to_data_surface_um=fit_data["zpbs_to_data_surface_um"],
        zpbs_to_data_residuals_um=fit_data["zpbs_to_data_residuals_um"],
        zpbs_residual_surface_um=fit_data["zpbs_residual_surface_um"],
        zpbs_residual_residuals_um=fit_data["zpbs_residual_residuals_um"],
        zpbs_residual_coefficients_um=fit_data["zpbs_residual_coefficients_um"],
        x0_fit=fit_data["x0_fit"],
        y0_fit=fit_data["y0_fit"],
        z0_fit=fit_data["z0_fit"],
        fitted_sphere_radius_um=fit_data["fitted_sphere_radius_um"],
        applied_reference_radius_um=fit_data["applied_reference_radius_um"],
        prefit_best_radius_um=fit_data["prefit_best_radius_um"],
        sphere_fit_mode=fit_data["sphere_fit_mode"],
        center_weight=fit_data["center_weight"],
        sphere_sse_um2=fit_data["sphere_sse_um2"],
        sphere_mae_um=fit_data["sphere_mae_um"],
        sphere_rms_um=fit_data["sphere_rms_um"],
        zpbs_residual_sse_um2=fit_data["zpbs_residual_sse_um2"],
        zpbs_residual_mae_um=fit_data["zpbs_residual_mae_um"],
        zpbs_residual_rms_um=fit_data["zpbs_residual_rms_um"],
        zpbs_residual_cond=fit_data["zpbs_residual_cond"],
        zpbs_residual_on_axis_m0_um=fit_data["zpbs_residual_on_axis_m0_um"],
        observed_aperture_radius_um=fit_data["observed_aperture_radius_um"],
        norm_radius_um=fit_data["norm_radius_um"],
        target_vertex_x_um=fit_data["target_vertex_x_um"],
        target_vertex_y_um=fit_data["target_vertex_y_um"],
        target_vertex_z_um=fit_data["target_vertex_z_um"],
        reference_vertex_x_um=fit_data["reference_vertex_x_um"],
        reference_vertex_y_um=fit_data["reference_vertex_y_um"],
        reference_vertex_z_um=fit_data["reference_vertex_z_um"],
        vertex_mismatch_z_um=fit_data["vertex_mismatch_z_um"],
        vertex_um=fit_data["vertex_fit_um"],
        sphere_vertex_residual_um=fit_data["sphere_vertex_residual_um"],
        vertex_residual_um=fit_data["vertex_residual_um"],
        method=fit_data["method"],
        n_modes=fit_data["n_modes"],
        roc_mode=roc_mode,
        normalization_mode=normalization_mode,
        run_name=run_name,
        common_reference_radius_um=common_reference_radius_um,
        common_normalization_radius_um=common_normalization_radius_um,
        round_radii_um=fit_data["round_radii_um"],
        zernike_coeff_sigfigs=fit_data["zernike_coeff_sigfigs"],
    )


def precompute_best_radii(
    items: list[ProcessingInput],
    *,
    sphere_fit_mode: str,
    center_weight: float,
) -> tuple[dict[Path, float], list[dict[str, str]]]:
    """Fit a best sphere ROC for each file ahead of a common-radius batch run."""
    radii: dict[Path, float] = {}
    failures: list[dict[str, str]] = []

    for item in items:
        try:
            entry = get_sphere_prefit_entry(
                item.source_file,
                surf_id=item.metadata.surf_id,
                sphere_fit_mode=sphere_fit_mode,
                center_weight=center_weight,
            )
            radii[item.source_file] = entry.best_radius_um
        except Exception as exc:
            failures.append({"source_file": str(item.source_file), "error": f"prefit radius failed: {exc}"})

    return radii, failures


def precompute_common_normalization_radii_by_surf_id(
    items: list[ProcessingInput],
    *,
    sphere_fit_mode: str,
    center_weight: float,
    round_radii_um: bool,
) -> tuple[dict[str, float], dict[Path, float], list[dict[str, str]]]:
    """Derive one shared normalization radius per surf_id, chosen to cover all files in that family."""
    per_file_observed: dict[Path, float] = {}
    per_surf_id: dict[str, float] = {}
    failures: list[dict[str, str]] = []

    for item in items:
        try:
            entry = get_sphere_prefit_entry(
                item.source_file,
                surf_id=item.metadata.surf_id,
                sphere_fit_mode=sphere_fit_mode,
                center_weight=center_weight,
            )
            observed_radius = entry.observed_aperture_radius_um
            per_file_observed[item.source_file] = observed_radius
            current = per_surf_id.get(item.metadata.surf_id)
            per_surf_id[item.metadata.surf_id] = observed_radius if current is None else max(current, observed_radius)
        except Exception as exc:
            failures.append(
                {"source_file": str(item.source_file), "error": f"prefit normalization radius failed: {exc}"}
            )

    if round_radii_um:
        per_surf_id = {
            surf_id: clamp_normalization_radius_um(round_nearest_micrometer(radius_um), radius_um)
            for surf_id, radius_um in per_surf_id.items()
        }

    return per_surf_id, per_file_observed, failures
