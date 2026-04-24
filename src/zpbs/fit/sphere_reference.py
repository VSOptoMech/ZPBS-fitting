"""Sphere-reference selection and fitting helpers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from zpbs.azp_csv_pipeline import cartesian_to_polar, fit_sphere
from zpbs.common import validate_center_weight
from zpbs.io.xyz import load_xyz_point_cloud
from zpbs.models import SpherePrefitEntry, SphereReferenceFit, VertexTarget


def select_vertex_target(x: Any, y: Any, z: Any, *, near_center_rel_tol: float = 1e-3) -> VertexTarget:
    """Choose the data vertex target, leaving room for future local patch estimators."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    z_arr = np.asarray(z, dtype=float)
    rho = np.sqrt(x_arr**2 + y_arr**2)
    rho_max = float(np.max(rho)) if rho.size else 0.0
    tol_um = max(1.0, rho_max * near_center_rel_tol)
    near_center = np.where(rho <= tol_um)[0]
    if near_center.size:
        index = int(near_center[np.argmin(rho[near_center])])
    else:
        index = int(np.argmin(rho))
    return VertexTarget(
        index=index,
        x_um=float(x_arr[index]),
        y_um=float(y_arr[index]),
        z_um=float(z_arr[index]),
        rho_um=float(rho[index]),
    )


def infer_surface_branch_sign(z: Any, rho: Any, *, target_vertex_z_um: float) -> float:
    """Infer which sphere branch matches the data: +1 for z0+r at the vertex, else -1."""
    z_arr = np.asarray(z, dtype=float)
    rho_arr = np.asarray(rho, dtype=float)
    edge = _edge_mask(rho_arr)
    edge_z = z_arr[edge] if np.any(edge) else z_arr
    edge_level = float(np.median(edge_z))
    # When the aperture edge lies below the selected vertex, the sphere center must
    # lie below that vertex too, which corresponds to the z0 + r branch at rho=0.
    return 1.0 if edge_level <= target_vertex_z_um else -1.0


def radial_weight_profile(rho_um: Any, observed_radius_um: float, *, center_weight: float) -> Any:
    """
    Smooth radial weighting for center-prioritized sphere fits.

    The weight profile is `exp(-rho_norm^2 / (2 * sigma_norm^2))` with `rho_norm = rho / observed_radius`.
    `center_weight` scales the Gaussian width smoothly:
    - values below 0.05 are treated as flat weighting
    - `center_weight=1` gives an edge weight near 1e-3
    - larger values narrow the Gaussian proportionally
    - smaller positive values widen it proportionally
    """
    weight = validate_center_weight(center_weight)
    rho_arr = np.asarray(rho_um, dtype=float)
    if observed_radius_um <= 0:
        return np.ones_like(rho_arr, dtype=float)
    if weight < 0.05:
        return np.ones_like(rho_arr, dtype=float)
    rho_norm = rho_arr / float(observed_radius_um)
    sigma_at_1 = 1.0 / math.sqrt(2.0 * math.log(1000.0))
    sigma_norm = sigma_at_1 / weight
    weights = np.exp(-np.square(rho_norm) / (2.0 * sigma_norm**2))
    return np.clip(weights, 1e-6, None)


def sphere_reference_vertex(
    *,
    x0_fit: float,
    y0_fit: float,
    z0_fit: float,
    radius_um: float,
    surface_branch_sign: float,
) -> tuple[float, float, float]:
    """Return the vertex point on the selected sphere branch."""
    return float(x0_fit), float(y0_fit), float(z0_fit + surface_branch_sign * radius_um)


def build_sphere_reference_fit(
    *,
    x0_fit: float,
    y0_fit: float,
    z0_fit: float,
    radius_um: float,
    residuals: Any,
    sphere_fit_mode: str,
    center_weight: float,
    target_vertex: VertexTarget,
    surface_branch_sign: float,
) -> SphereReferenceFit:
    """Bundle fitted sphere geometry with derived diagnostics."""
    ref_x, ref_y, ref_z = sphere_reference_vertex(
        x0_fit=float(x0_fit),
        y0_fit=float(y0_fit),
        z0_fit=float(z0_fit),
        radius_um=float(radius_um),
        surface_branch_sign=float(surface_branch_sign),
    )
    return SphereReferenceFit(
        x0_fit=float(x0_fit),
        y0_fit=float(y0_fit),
        z0_fit=float(z0_fit),
        radius_um=float(radius_um),
        residuals=residuals,
        sphere_fit_mode=sphere_fit_mode,
        center_weight=float(center_weight),
        target_vertex_x_um=target_vertex.x_um,
        target_vertex_y_um=target_vertex.y_um,
        target_vertex_z_um=target_vertex.z_um,
        reference_vertex_x_um=ref_x,
        reference_vertex_y_um=ref_y,
        reference_vertex_z_um=ref_z,
        vertex_mismatch_z_um=float(ref_z - target_vertex.z_um),
    )


def fit_sphere_with_fixed_radius(
    x: Any,
    y: Any,
    z: Any,
    *,
    radius_um: float,
    initial_guess_center: tuple[float, float, float],
    sphere_fit_mode: str,
    center_weight: float,
    target_vertex: VertexTarget,
    surface_branch_sign: float,
) -> SphereReferenceFit:
    """Fit the reference sphere while keeping the radius fixed."""
    if radius_um <= 0:
        raise ValueError(f"Fixed radius must be positive, got {radius_um}.")

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    z_arr = np.asarray(z, dtype=float)

    rho = np.sqrt(x_arr**2 + y_arr**2)
    rho_edge = float(np.max(rho))
    if radius_um <= rho_edge:
        raise ValueError(f"Fixed radius {radius_um} um is smaller than the observed aperture radius {rho_edge} um.")

    sag_mag = radius_um - np.sqrt(max(radius_um**2 - rho_edge**2, 0.0))
    if np.isclose(sag_mag, 0.0):
        raise ValueError("Degenerate fixed-radius sphere estimate: sag is approximately zero.")

    mode = sphere_fit_mode.strip().lower()
    if mode == "vertex_locked":
        x0_fit = float(target_vertex.x_um)
        y0_fit = float(target_vertex.y_um)
        z0_fit = float(target_vertex.z_um - surface_branch_sign * radius_um)
        distances = np.sqrt((x_arr - x0_fit) ** 2 + (y_arr - y0_fit) ** 2 + (z_arr - z0_fit) ** 2)
        residuals = distances - radius_um
        return build_sphere_reference_fit(
            x0_fit=x0_fit,
            y0_fit=y0_fit,
            z0_fit=z0_fit,
            radius_um=radius_um,
            residuals=residuals,
            sphere_fit_mode=mode,
            center_weight=center_weight,
            target_vertex=target_vertex,
            surface_branch_sign=surface_branch_sign,
        )

    initial_guess = np.asarray(initial_guess_center, dtype=float)
    rho_from_vertex = np.sqrt((x_arr - target_vertex.x_um) ** 2 + (y_arr - target_vertex.y_um) ** 2)
    observed_radius_um = float(np.max(rho_from_vertex))
    weights = (
        np.ones_like(rho_from_vertex, dtype=float)
        if mode == "legacy_lsq"
        else radial_weight_profile(rho_from_vertex, observed_radius_um, center_weight=center_weight)
    )

    def residuals(params: Any) -> Any:
        x0_fit, y0_fit, z0_fit = params
        distances = np.sqrt((x_arr - x0_fit) ** 2 + (y_arr - y0_fit) ** 2 + (z_arr - z0_fit) ** 2)
        return distances - radius_um

    def weighted_residuals(params: Any) -> Any:
        return np.sqrt(weights) * residuals(params)

    result = least_squares(weighted_residuals, initial_guess)
    x0_fit, y0_fit, z0_fit = result.x
    return build_sphere_reference_fit(
        x0_fit=float(x0_fit),
        y0_fit=float(y0_fit),
        z0_fit=float(z0_fit),
        radius_um=float(radius_um),
        residuals=residuals(result.x),
        sphere_fit_mode=mode,
        center_weight=center_weight,
        target_vertex=target_vertex,
        surface_branch_sign=surface_branch_sign,
    )


def _edge_mask(rho: Any, *, relative_tol: float = 1e-3) -> Any:
    """Select points near the aperture edge for robust sphere initialization."""
    rho_arr = np.asarray(rho, dtype=float)
    rho_edge = float(np.max(rho_arr))
    if np.isclose(rho_edge, 0.0):
        return rho_arr == rho_edge
    return rho_arr >= rho_edge * (1.0 - relative_tol)


def _sphere_seed_candidates(x: Any, y: Any, z: Any) -> list[tuple[float, float, float, float]]:
    """Build symmetric sphere seeds that work for both initial/deformed surfaces."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    z_arr = np.asarray(z, dtype=float)
    rho = np.sqrt(x_arr**2 + y_arr**2)
    edge = _edge_mask(rho)
    edge_z = z_arr[edge]
    if edge_z.size == 0:
        edge_z = z_arr

    x0_seed = float(np.mean(x_arr))
    y0_seed = float(np.mean(y_arr))
    candidates: list[tuple[float, float, float, float]] = []

    for vertex_z, edge_z_ref, center_sign in (
        (float(np.min(z_arr)), float(np.max(edge_z)), +1.0),
        (float(np.max(z_arr)), float(np.min(edge_z)), -1.0),
    ):
        sag = abs(vertex_z - edge_z_ref)
        rho_edge = float(np.max(rho))
        if np.isclose(sag, 0.0):
            continue
        radius = float((sag**2 + rho_edge**2) / (2.0 * sag))
        z0_seed = float(vertex_z + center_sign * radius)
        candidates.append((x0_seed, y0_seed, z0_seed, radius))

    z_span = float(np.ptp(z_arr))
    rho_edge = float(np.max(rho))
    fallback_sag = max(z_span, rho_edge * 1e-3, 1.0)
    fallback_radius = float((fallback_sag**2 + rho_edge**2) / (2.0 * fallback_sag))
    candidates.append((x0_seed, y0_seed, float(np.mean(z_arr) - fallback_radius), fallback_radius))
    return candidates


def fit_sphere_robust(x: Any, y: Any, z: Any, *, sphere_fit_mode: str, center_weight: float) -> SphereReferenceFit:
    """Fit the sphere reference using the requested objective mode."""
    mode = sphere_fit_mode.strip().lower()
    if mode not in {"legacy_lsq", "center_weighted", "vertex_locked"}:
        raise ValueError(
            f"sphere_fit_mode must be one of legacy_lsq, center_weighted, or vertex_locked; got {sphere_fit_mode!r}."
        )
    weight = validate_center_weight(center_weight)
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    z_arr = np.asarray(z, dtype=float)
    target_vertex = select_vertex_target(x_arr, y_arr, z_arr)
    rho_from_vertex = np.sqrt((x_arr - target_vertex.x_um) ** 2 + (y_arr - target_vertex.y_um) ** 2)
    surface_branch_sign = infer_surface_branch_sign(
        z_arr,
        rho_from_vertex,
        target_vertex_z_um=target_vertex.z_um,
    )

    if mode == "legacy_lsq":
        try:
            x0_fit, y0_fit, z0_fit, radius_um, residuals = fit_sphere(x_arr, y_arr, z_arr)
        except ValueError as exc:
            if "Degenerate sphere estimate" not in str(exc):
                raise
        else:
            return build_sphere_reference_fit(
                x0_fit=x0_fit,
                y0_fit=y0_fit,
                z0_fit=z0_fit,
                radius_um=radius_um,
                residuals=residuals,
                sphere_fit_mode=mode,
                center_weight=weight,
                target_vertex=target_vertex,
                surface_branch_sign=surface_branch_sign,
            )

    def sphere_residuals(params: Any) -> Any:
        x0_fit, y0_fit, z0_fit, radius_um = params
        distances = np.sqrt((x_arr - x0_fit) ** 2 + (y_arr - y0_fit) ** 2 + (z_arr - z0_fit) ** 2)
        return distances - radius_um

    def weighted_residuals(params: Any, weights: Any) -> Any:
        return np.sqrt(weights) * sphere_residuals(params)

    def residuals_radius(radius_params: Any) -> Any:
        radius_um = float(radius_params[0])
        z0_fit = float(target_vertex.z_um - surface_branch_sign * radius_um)
        distances = np.sqrt(
            (x_arr - target_vertex.x_um) ** 2 + (y_arr - target_vertex.y_um) ** 2 + (z_arr - z0_fit) ** 2
        )
        return distances - radius_um

    if mode == "vertex_locked":
        best_radius_um: float | None = None
        best_residuals: Any = None
        best_sse: float | None = None
        for initial_guess in _sphere_seed_candidates(x_arr, y_arr, z_arr):
            result = least_squares(
                residuals_radius,
                np.asarray([float(initial_guess[3])], dtype=float),
                bounds=([1e-9], [np.inf]),
            )
            residuals = residuals_radius(result.x)
            sse = float(np.sum(np.square(residuals)))
            if best_sse is None or sse < best_sse:
                best_radius_um = float(result.x[0])
                best_residuals = residuals
                best_sse = sse
        if best_radius_um is None or best_residuals is None:
            raise ValueError("Vertex-locked sphere fit could not build an initial radius guess.")

        z0_fit = float(target_vertex.z_um - surface_branch_sign * best_radius_um)
        return build_sphere_reference_fit(
            x0_fit=target_vertex.x_um,
            y0_fit=target_vertex.y_um,
            z0_fit=z0_fit,
            radius_um=best_radius_um,
            residuals=best_residuals,
            sphere_fit_mode=mode,
            center_weight=weight,
            target_vertex=target_vertex,
            surface_branch_sign=surface_branch_sign,
        )

    observed_radius_um = float(np.max(rho_from_vertex))
    weights = radial_weight_profile(rho_from_vertex, observed_radius_um, center_weight=weight)
    best_result: SphereReferenceFit | None = None
    best_sse: float | None = None
    for initial_guess in _sphere_seed_candidates(x_arr, y_arr, z_arr):
        result = least_squares(
            weighted_residuals,
            np.asarray(initial_guess, dtype=float),
            args=(weights,),
            bounds=([-np.inf, -np.inf, -np.inf, 1e-9], [np.inf, np.inf, np.inf, np.inf]),
        )
        residuals = sphere_residuals(result.x)
        sse = float(np.sum(np.square(np.sqrt(weights) * residuals)))
        if best_sse is None or sse < best_sse:
            x0_fit, y0_fit, z0_fit, radius_um = result.x
            best_result = build_sphere_reference_fit(
                x0_fit=float(x0_fit),
                y0_fit=float(y0_fit),
                z0_fit=float(z0_fit),
                radius_um=float(radius_um),
                residuals=residuals,
                sphere_fit_mode=mode,
                center_weight=weight,
                target_vertex=target_vertex,
                surface_branch_sign=surface_branch_sign,
            )
            best_sse = sse

    if best_result is None:
        raise ValueError("Robust sphere fit failed to find a valid initialization.")
    return best_result


def get_sphere_prefit_entry(
    file_path: Path,
    *,
    surf_id: str,
    sphere_fit_mode: str = "center_weighted",
    center_weight: float = 0.5,
) -> SpherePrefitEntry:
    """Load one file and compute the robust sphere prefit used by downstream steps."""
    _ = surf_id
    x, y, z = load_xyz_point_cloud(file_path)
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    z_arr = np.asarray(z, dtype=float)
    raw_aperture_radius_um = float(np.max(np.sqrt(x_arr**2 + y_arr**2)))
    sphere_fit = fit_sphere_robust(
        x_arr,
        y_arr,
        z_arr,
        sphere_fit_mode=sphere_fit_mode,
        center_weight=center_weight,
    )
    rho, _ = cartesian_to_polar(x_arr - sphere_fit.x0_fit, y_arr - sphere_fit.y0_fit)
    return SpherePrefitEntry(
        x=x_arr,
        y=y_arr,
        z=z_arr,
        raw_aperture_radius_um=raw_aperture_radius_um,
        x0_fit=float(sphere_fit.x0_fit),
        y0_fit=float(sphere_fit.y0_fit),
        z0_fit=float(sphere_fit.z0_fit),
        best_radius_um=float(sphere_fit.radius_um),
        best_sphere_residuals=np.asarray(sphere_fit.residuals, dtype=float),
        observed_aperture_radius_um=float(np.max(rho)),
        sphere_fit_mode=sphere_fit.sphere_fit_mode,
        center_weight=sphere_fit.center_weight,
        target_vertex_x_um=sphere_fit.target_vertex_x_um,
        target_vertex_y_um=sphere_fit.target_vertex_y_um,
        target_vertex_z_um=sphere_fit.target_vertex_z_um,
        reference_vertex_x_um=sphere_fit.reference_vertex_x_um,
        reference_vertex_y_um=sphere_fit.reference_vertex_y_um,
        reference_vertex_z_um=sphere_fit.reference_vertex_z_um,
        vertex_mismatch_z_um=sphere_fit.vertex_mismatch_z_um,
    )
