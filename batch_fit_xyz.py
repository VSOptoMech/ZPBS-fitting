"""Batch-fit notebook-style XYZ surface files without previews or plots."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import re
import sys
from dataclasses import dataclass, replace
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

NOTEBOOK_FILENAME_RE = re.compile(
    r"^(?P<design_token>R\d+V(?P<design_id>[^-_]+))-(?P<fea_id>[^_]+)_(?P<force_id>F[^_]+)_FVS_(?P<surface_token>[A-Za-z]+)$"
)
SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")
FOCUS_SURF_IDS = {"AA", "AP", "PA", "PP"}


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
    sphere_residuals: Any
    zernike_surface: Any
    zernike_surface_residuals: Any
    zernike_residual_surface: Any
    zernike_residual_residuals: Any
    zpoly_fits: Any
    zpoly_fits2: Any
    x0_fit: float
    y0_fit: float
    z0_fit: float
    fitted_sphere_radius_um: float
    applied_reference_radius_um: float
    prefit_best_radius_um: float | None
    sphere_sse: float
    sphere_rms: float
    surface_zernike_sse: float
    surface_zernike_rms: float
    surface_zernike_cond: float
    sphere_residual_zernike_sse: float
    sphere_residual_zernike_rms: float
    sphere_residual_zernike_cond: float
    observed_aperture_radius_um: float
    norm_radius_um: float
    vertex_um: float
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


def parse_surface_metadata(file_path: Path) -> SurfaceMetadata:
    """Parse notebook-style metadata from an XYZ filename."""
    stem = file_path.stem
    match = NOTEBOOK_FILENAME_RE.match(stem)
    if match is not None:
        groups = match.groupdict()
        surface_token = groups["surface_token"].upper()
        if surface_token.endswith(("I", "D")) and len(surface_token) > 2:
            surf_id = surface_token[:-1]
        else:
            surf_id = surface_token
        return SurfaceMetadata(
            design_token=groups["design_token"],
            design_id=groups["design_id"],
            fea_id=groups["fea_id"],
            force_id=groups["force_id"],
            surface_token=surface_token,
            surf_id=surf_id,
        )

    prefix = stem.split("_", maxsplit=1)[0]
    design_id = prefix.split("V")[-1] if "V" in prefix else prefix
    return SurfaceMetadata(
        design_token=prefix,
        design_id=design_id,
        fea_id="UNKNOWN",
        force_id="UNKNOWN",
        surface_token="UNKNOWN",
        surf_id="UNKNOWN",
    )


def _is_float(text: str) -> bool:
    """Return True when text can be parsed as a float."""
    try:
        float(text)
    except ValueError:
        return False
    return True


def _split_xyz_line(line: str, delimiter: str | None) -> list[str]:
    """Split a candidate XYZ row using either a CSV delimiter or whitespace."""
    stripped = line.strip()
    if delimiter is None:
        return stripped.split()
    return [field.strip() for field in stripped.split(delimiter)]


def _detect_delimiter(lines: list[str]) -> str | None:
    """Detect the field delimiter used by the XYZ file."""
    sample = "\n".join(lines[:10])
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
    except csv.Error:
        return None
    return dialect.delimiter


def load_xyz_point_cloud(file_path: Path) -> tuple[list[float], list[float], list[float]]:
    """Load a 3-column XYZ point cloud from CSV- or whitespace-delimited text."""
    raw_lines = [line for line in file_path.read_text().splitlines() if line.strip()]
    if not raw_lines:
        raise ValueError(f"Point-cloud file is empty: {file_path}")

    delimiter = _detect_delimiter(raw_lines)
    rows = [_split_xyz_line(line, delimiter) for line in raw_lines]
    rows = [row for row in rows if len(row) >= 3]
    if not rows:
        raise ValueError(f"No valid 3-column rows found in {file_path}.")

    header_offset = 0
    first = rows[0][:3]
    if not all(_is_float(value) for value in first):
        lowered = [value.lower() for value in first]
        if lowered == ["x", "y", "z"]:
            header_offset = 1
        else:
            raise ValueError(f"Unsupported XYZ header in {file_path}: {first}")

    x_vals: list[float] = []
    y_vals: list[float] = []
    z_vals: list[float] = []
    for row in rows[header_offset:]:
        x_txt, y_txt, z_txt = row[:3]
        if not (_is_float(x_txt) and _is_float(y_txt) and _is_float(z_txt)):
            continue
        x_vals.append(float(x_txt))
        y_vals.append(float(y_txt))
        z_vals.append(float(z_txt))

    if not x_vals:
        raise ValueError(f"No numeric XYZ rows found in {file_path}.")

    return x_vals, y_vals, z_vals


def make_output_filename(metadata: SurfaceMetadata) -> str:
    """Mirror the notebook export naming style for coefficient CSVs."""
    return (
        f"ZPs_{metadata.design_token}-{metadata.fea_id}_{metadata.surface_token}_{metadata.force_id}_base_sphere.csv"
    )


def format_processed_label(metadata: SurfaceMetadata) -> str:
    """Build a stable processed-identity label used in reports and filenames."""
    return f"{metadata.design_token}-{metadata.fea_id}_{metadata.surface_token}_{metadata.force_id}"


def format_tension(force_id: str) -> str:
    """Convert a force token like F0.8mN to the CSV field value expected downstream."""
    if force_id.startswith("F") and force_id.endswith("mN"):
        return force_id[1:-2]
    return force_id


def force_sort_key(force_id: str) -> tuple[float, str]:
    """Sort F-style force tokens numerically and unknowns last."""
    if force_id.startswith("F") and force_id.endswith("mN"):
        try:
            return (float(force_id[1:-2]), force_id)
        except ValueError:
            pass
    return (float("inf"), force_id)


def uses_posterior_sign_convention(surf_id: str) -> bool:
    """Apply the notebook's posterior residual convention to posterior regional surfaces too."""
    return surf_id.upper().startswith("P")


def is_focus_surface_family(surf_id: str) -> bool:
    """Return True for the only surface families included in this project analysis."""
    return surf_id.upper() in FOCUS_SURF_IDS


def sanitize_h5_name(name: str) -> str:
    """Convert arbitrary file stems into safe HDF5 group names."""
    return SAFE_NAME_RE.sub("_", name).strip("_") or "unnamed"


def resolve_run_name(run_name: str | None) -> str:
    """Generate a deterministic run-folder name when none is provided."""
    if run_name:
        return run_name
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


def resolve_analysis_date() -> str:
    """Return the local analysis date used in dated artifact names and titles."""
    return datetime.now().date().isoformat()


def parse_boolish(value: object, default: bool = False) -> bool:
    """Parse workbook- or manifest-style booleans."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if not text:
        return default
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def parse_optional_int(value: object) -> int | None:
    """Parse optional integer-like workbook or manifest fields."""
    if value in (None, "", "null"):
        return None
    return int(float(str(value)))


def round_nearest_micrometer(value_um: float) -> float:
    """Round to the nearest whole micrometer using half-up semantics."""
    return float(math.floor(float(value_um) + 0.5))


def clamp_reference_radius_um(radius_um: float, min_aperture_um: float) -> float:
    """Clamp to the smallest valid integer radius strictly larger than the aperture."""
    if radius_um > min_aperture_um:
        return float(radius_um)
    return float(math.floor(min_aperture_um) + 1)


def clamp_normalization_radius_um(radius_um: float, min_aperture_um: float) -> float:
    """Clamp to the smallest valid integer normalization radius covering the aperture."""
    if radius_um >= min_aperture_um:
        return float(radius_um)
    return float(math.ceil(min_aperture_um))


def round_sigfigs_array(values: Any, sigfigs: int, *, np: Any) -> Any:
    """Round an array to the requested number of significant digits."""
    if sigfigs < 1:
        raise ValueError(f"sigfigs must be positive, got {sigfigs}")

    arr = np.asarray(values, dtype=float).copy()
    mask = np.isfinite(arr) & (arr != 0.0)
    if not np.any(mask):
        return arr

    magnitudes = np.floor(np.log10(np.abs(arr[mask])))
    scales = np.power(10.0, sigfigs - 1 - magnitudes)
    arr[mask] = np.sign(arr[mask]) * np.floor(np.abs(arr[mask]) * scales + 0.5) / scales
    return arr


def collapse_identical_initial_inputs(files: list[Path]) -> list[ProcessingInput]:
    """Collapse identical *I files to one synthetic zero-force entry per surface family."""
    initial_groups: dict[str, list[tuple[Path, SurfaceMetadata]]] = {}
    processing_inputs: list[ProcessingInput] = []

    for file_path in files:
        metadata = parse_surface_metadata(file_path)
        if metadata.surface_token.endswith("I"):
            initial_groups.setdefault(metadata.surf_id, []).append((file_path, metadata))
            continue
        processing_inputs.append(
            ProcessingInput(
                source_file=file_path,
                source_metadata=metadata,
                metadata=metadata,
            )
        )

    for surf_id, members in sorted(initial_groups.items()):
        surface_tokens = {metadata.surface_token for _, metadata in members}
        if len(surface_tokens) != 1:
            raise ValueError(
                f"Cannot collapse {surf_id} initial states because multiple surface tokens were found: "
                f"{sorted(surface_tokens)}"
            )
        payloads = {file_path.read_bytes() for file_path, _ in members}
        if len(payloads) != 1:
            raise ValueError(
                f"Cannot collapse {surf_id} initial states because the raw *I payloads are not identical."
            )

        representative_file, representative_metadata = min(
            members,
            key=lambda item: (force_sort_key(item[1].force_id), item[0].name),
        )
        processing_inputs.append(
            ProcessingInput(
                source_file=representative_file,
                source_metadata=representative_metadata,
                metadata=replace(representative_metadata, force_id="F0.0mN"),
            )
        )

    return sorted(
        processing_inputs,
        key=lambda item: (
            item.metadata.surf_id,
            force_sort_key(item.metadata.force_id),
            item.metadata.surface_token,
            item.source_file.name,
        ),
    )


@lru_cache(maxsize=1)
def _load_azp_symbols() -> dict[str, Any]:
    """Lazy-load heavy notebook dependencies from the existing core module."""
    import numpy as np
    from azp_csv_pipeline import (
        _build_initial_fit_guess,
        _build_residual_fit_guess,
        build_zernike_coefficients_rows,
        _pad_coeffs_to_45,
        cartesian_to_polar,
        export_zernike_coefficients_csv,
        fit_sphere,
        fit_zernike_lstsq,
        zernike_polar_basis,
        zernike_polar,
        zernike_polar_basis,
    )
    from scipy.optimize import curve_fit, least_squares

    return {
        "np": np,
        "curve_fit": curve_fit,
        "least_squares": least_squares,
        "_pad_coeffs_to_45": _pad_coeffs_to_45,
        "_build_initial_fit_guess": _build_initial_fit_guess,
        "_build_residual_fit_guess": _build_residual_fit_guess,
        "build_zernike_coefficients_rows": build_zernike_coefficients_rows,
        "cartesian_to_polar": cartesian_to_polar,
        "export_zernike_coefficients_csv": export_zernike_coefficients_csv,
        "fit_sphere": fit_sphere,
        "fit_zernike_lstsq": fit_zernike_lstsq,
        "zernike_polar_basis": zernike_polar_basis,
        "zernike_polar": zernike_polar,
        "zernike_polar_basis": zernike_polar_basis,
    }


def fit_sphere_with_fixed_radius(
    x: Any,
    y: Any,
    z: Any,
    *,
    radius_um: float,
    initial_guess_center: tuple[float, float, float],
    least_squares_fn: Any,
    np: Any,
) -> tuple[float, float, float, Any]:
    """Fit a sphere center while keeping the radius fixed."""
    if radius_um <= 0:
        raise ValueError(f"Fixed radius must be positive, got {radius_um}.")

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    z_arr = np.asarray(z, dtype=float)

    rho = np.sqrt(x_arr**2 + y_arr**2)
    rho_edge = float(np.max(rho))
    if radius_um <= rho_edge:
        raise ValueError(
            f"Fixed radius {radius_um} um is smaller than the observed aperture radius {rho_edge} um."
        )

    sag_mag = radius_um - np.sqrt(max(radius_um**2 - rho_edge**2, 0.0))
    if np.isclose(sag_mag, 0.0):
        raise ValueError("Degenerate fixed-radius sphere estimate: sag is approximately zero.")

    initial_guess = np.asarray(initial_guess_center, dtype=float)

    def residuals(params: Any) -> Any:
        x0_fit, y0_fit, z0_fit = params
        distances = np.sqrt((x_arr - x0_fit) ** 2 + (y_arr - y0_fit) ** 2 + (z_arr - z0_fit) ** 2)
        return distances - radius_um

    result = least_squares_fn(residuals, initial_guess)
    x0_fit, y0_fit, z0_fit = result.x
    return float(x0_fit), float(y0_fit), float(z0_fit), residuals(result.x)


def _edge_mask(rho: Any, *, np: Any, relative_tol: float = 1e-3) -> Any:
    """Select points near the aperture edge for robust sphere initialization."""
    rho_arr = np.asarray(rho, dtype=float)
    rho_edge = float(np.max(rho_arr))
    if np.isclose(rho_edge, 0.0):
        return rho_arr == rho_edge
    return rho_arr >= rho_edge * (1.0 - relative_tol)


def _sphere_seed_candidates(x: Any, y: Any, z: Any, *, np: Any) -> list[tuple[float, float, float, float]]:
    """Build symmetric sphere seeds that work for both initial/deformed surfaces."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    z_arr = np.asarray(z, dtype=float)
    rho = np.sqrt(x_arr**2 + y_arr**2)
    edge = _edge_mask(rho, np=np)
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


def fit_sphere_robust(x: Any, y: Any, z: Any, *, fit_sphere: Any, least_squares_fn: Any, np: Any) -> tuple[float, float, float, float, Any]:
    """Prefer the inherited fitter, with a fallback initializer for zero-sag real files."""
    try:
        return fit_sphere(x, y, z)
    except ValueError as exc:
        if "Degenerate sphere estimate" not in str(exc):
            raise

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    z_arr = np.asarray(z, dtype=float)

    def sphere_residuals(params: Any) -> Any:
        x0_fit, y0_fit, z0_fit, radius_um = params
        distances = np.sqrt((x_arr - x0_fit) ** 2 + (y_arr - y0_fit) ** 2 + (z_arr - z0_fit) ** 2)
        return distances - radius_um

    best_result: tuple[float, float, float, float, Any] | None = None
    best_sse: float | None = None
    for initial_guess in _sphere_seed_candidates(x_arr, y_arr, z_arr, np=np):
        result = least_squares_fn(
            sphere_residuals,
            np.asarray(initial_guess, dtype=float),
            bounds=([-np.inf, -np.inf, -np.inf, 1e-9], [np.inf, np.inf, np.inf, np.inf]),
        )
        residuals = sphere_residuals(result.x)
        sse = float(np.sum(np.square(residuals)))
        if best_sse is None or sse < best_sse:
            x0_fit, y0_fit, z0_fit, radius_um = result.x
            best_result = (
                float(x0_fit),
                float(y0_fit),
                float(z0_fit),
                float(radius_um),
                residuals,
            )
            best_sse = sse

    if best_result is None:
        raise ValueError("Robust sphere fit failed to find a valid initialization.")
    return best_result


def get_sphere_prefit_entry(
    file_path: Path,
    *,
    surf_id: str,
) -> SpherePrefitEntry:
    """Load one file and compute the robust sphere prefit used by downstream steps."""
    symbols = _load_azp_symbols()
    np = symbols["np"]
    fit_sphere = symbols["fit_sphere"]
    least_squares = symbols["least_squares"]
    cartesian_to_polar = symbols["cartesian_to_polar"]

    x, y, z = load_xyz_point_cloud(file_path)
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    z_arr = np.asarray(z, dtype=float)
    raw_aperture_radius_um = float(np.max(np.sqrt(x_arr**2 + y_arr**2)))
    x0_fit, y0_fit, z0_fit, best_radius_um, best_sphere_residuals = fit_sphere_robust(
        x_arr,
        y_arr,
        z_arr,
        fit_sphere=fit_sphere,
        least_squares_fn=least_squares,
        np=np,
    )
    rho, _ = cartesian_to_polar(x_arr - x0_fit, y_arr - y0_fit)
    return SpherePrefitEntry(
        x=x_arr,
        y=y_arr,
        z=z_arr,
        raw_aperture_radius_um=raw_aperture_radius_um,
        x0_fit=float(x0_fit),
        y0_fit=float(y0_fit),
        z0_fit=float(z0_fit),
        best_radius_um=float(best_radius_um),
        best_sphere_residuals=np.asarray(best_sphere_residuals, dtype=float),
        observed_aperture_radius_um=float(np.max(rho)),
    )


def run_fit_pipeline(
    x: list[float],
    y: list[float],
    z: list[float],
    *,
    surf_id: str,
    method: str,
    n_modes: int,
    maxfev: int,
    rcond: float | None,
    reference_radius_um: float | None,
    normalization_radius_um: float | None,
    round_radii_um: bool = False,
    zernike_coeff_sigfigs: int | None = None,
    prefit_data: SpherePrefitEntry | None = None,
) -> dict[str, Any]:
    """Run the headless sphere-plus-Zernike workflow with optional common radius."""
    symbols = _load_azp_symbols()
    np = symbols["np"]
    curve_fit = symbols["curve_fit"]
    least_squares = symbols["least_squares"]
    pad_coeffs = symbols["_pad_coeffs_to_45"]
    build_initial = symbols["_build_initial_fit_guess"]
    build_residual = symbols["_build_residual_fit_guess"]
    cartesian_to_polar = symbols["cartesian_to_polar"]
    fit_sphere = symbols["fit_sphere"]
    fit_zernike_lstsq = symbols["fit_zernike_lstsq"]
    zernike_polar_basis = symbols["zernike_polar_basis"]
    zernike_polar = symbols["zernike_polar"]
    zernike_polar_basis = symbols["zernike_polar_basis"]

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    z_arr = np.asarray(z, dtype=float)
    if len(x_arr) != len(y_arr) or len(y_arr) != len(z_arr):
        raise ValueError("x, y, z must have equal length.")
    if len(x_arr) == 0:
        raise ValueError("No data points provided.")
    if zernike_coeff_sigfigs is not None and zernike_coeff_sigfigs < 1:
        raise ValueError("--zernike-coeff-sigfigs must be positive.")

    raw_aperture_radius_um = float(np.max(np.sqrt(x_arr**2 + y_arr**2)))
    if prefit_data is None:
        best_x0, best_y0, best_z0, best_radius_um, best_sphere_residuals = fit_sphere_robust(
            x_arr,
            y_arr,
            z_arr,
            fit_sphere=fit_sphere,
            least_squares_fn=least_squares,
            np=np,
        )
        prefit_best_radius_um = float(best_radius_um)
    else:
        best_x0 = float(prefit_data.x0_fit)
        best_y0 = float(prefit_data.y0_fit)
        best_z0 = float(prefit_data.z0_fit)
        prefit_best_radius_um = float(prefit_data.best_radius_um)
        best_sphere_residuals = np.asarray(prefit_data.best_sphere_residuals, dtype=float)

    if reference_radius_um is None:
        if round_radii_um:
            applied_radius_um = clamp_reference_radius_um(
                round_nearest_micrometer(prefit_best_radius_um),
                raw_aperture_radius_um,
            )
            x0_fit, y0_fit, z0_fit, sphere_residuals = fit_sphere_with_fixed_radius(
                x_arr,
                y_arr,
                z_arr,
                radius_um=applied_radius_um,
                initial_guess_center=(float(best_x0), float(best_y0), float(best_z0)),
                least_squares_fn=least_squares,
                np=np,
            )
        else:
            x0_fit = float(best_x0)
            y0_fit = float(best_y0)
            z0_fit = float(best_z0)
            applied_radius_um = prefit_best_radius_um
            sphere_residuals = np.asarray(best_sphere_residuals, dtype=float)
    else:
        requested_radius_um = float(reference_radius_um)
        if round_radii_um:
            requested_radius_um = clamp_reference_radius_um(
                round_nearest_micrometer(requested_radius_um),
                raw_aperture_radius_um,
            )
        x0_fit, y0_fit, z0_fit, sphere_residuals = fit_sphere_with_fixed_radius(
            x_arr,
            y_arr,
            z_arr,
            radius_um=requested_radius_um,
            initial_guess_center=(float(best_x0), float(best_y0), float(best_z0)),
            least_squares_fn=least_squares,
            np=np,
        )
        applied_radius_um = requested_radius_um

    if uses_posterior_sign_convention(surf_id):
        sphere_residuals = -sphere_residuals

    sphere_sse = float(np.sum(np.square(sphere_residuals)))
    sphere_rms = float(np.sqrt(np.mean(np.square(sphere_residuals))))
    rho, phi = cartesian_to_polar(x_arr - x0_fit, y_arr - y0_fit)
    observed_rho_max = float(np.max(rho))
    if observed_rho_max <= 0:
        raise ValueError("Invalid polar radius max; all points may be coincident.")

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
    zv2 = float(sphere_residuals[idx_center])
    method_key = method.strip().lower()

    if method_key == "lstsq":
        zpoly_fits_raw, _, _, surface_rms, surface_cond = fit_zernike_lstsq(
            rho_norm, phi, z_arr, n_modes=n_modes, rcond=rcond
        )
        zpoly_fits2_raw, _, _, residual_rms, residual_cond = fit_zernike_lstsq(
            rho_norm, phi, sphere_residuals, n_modes=n_modes, rcond=rcond
        )
    elif method_key == "curve_fit":
        zpoly_fit0 = build_initial(zv=zv, r_fit=applied_radius_um)
        zpoly_fits_raw, _ = curve_fit(zernike_polar, pol_loci, z_arr, zpoly_fit0, maxfev=maxfev)
        zpoly_fit00 = build_residual(zv2=zv2)
        zpoly_fits2_raw, _ = curve_fit(
            zernike_polar,
            pol_loci,
            sphere_residuals,
            zpoly_fit00,
            maxfev=maxfev,
        )
    else:
        raise ValueError("method must be one of {'lstsq', 'curve_fit'}")

    if method_key == "curve_fit":
        surface_rms = float(np.sqrt(np.mean(zernike_surface_residuals**2)))
        residual_rms = float(np.sqrt(np.mean(zernike_residual_residuals**2)))
        A = zernike_polar_basis(rho_norm, phi, n_modes=n_modes)
        surface_cond = float(np.linalg.cond(A))
        residual_cond = surface_cond  # same design matrix for both fits

    zpoly_fits = pad_coeffs(np.asarray(zpoly_fits_raw, dtype=float))
    zpoly_fits2 = pad_coeffs(np.asarray(zpoly_fits2_raw, dtype=float))
    if zernike_coeff_sigfigs is not None:
        zpoly_fits = round_sigfigs_array(zpoly_fits, zernike_coeff_sigfigs, np=np)
        zpoly_fits2 = round_sigfigs_array(zpoly_fits2, zernike_coeff_sigfigs, np=np)

    zernike_surface = zernike_polar(pol_loci, *zpoly_fits)
    zernike_surface_residuals = z_arr - zernike_surface
    zernike_residual_surface = zernike_polar(pol_loci, *zpoly_fits2)
    zernike_residual_residuals = sphere_residuals - zernike_residual_surface

    # Condition numbers depend only on the chosen basis, not on coefficient rounding.
    _ = zernike_polar_basis  # Keep the primary-source symbol loaded for parity and tests.
    surface_rms = float(np.sqrt(np.mean(zernike_surface_residuals**2)))
    residual_rms = float(np.sqrt(np.mean(zernike_residual_residuals**2)))

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
        "sphere_residuals": sphere_residuals,
        "sphere_sse": sphere_sse,
        "sphere_rms": sphere_rms,
        "zv": zv,
        "zv2": zv2,
        "zpoly_fits": zpoly_fits,
        "zpoly_fits2": zpoly_fits2,
        "zernike_surface": zernike_surface,
        "zernike_surface_residuals": zernike_surface_residuals,
        "zernike_residual_surface": zernike_residual_surface,
        "zernike_residual_residuals": zernike_residual_residuals,
        "surface_zernike_sse": float(np.sum(zernike_surface_residuals**2)),
        "surface_zernike_rms": float(surface_rms),
        "surface_zernike_cond": float(surface_cond),
        "sphere_residual_zernike_sse": float(np.sum(zernike_residual_residuals**2)),
        "sphere_residual_zernike_rms": float(residual_rms),
        "sphere_residual_zernike_cond": float(residual_cond),
        "observed_aperture_radius_um": observed_rho_max,
        "norm_radius_um": norm_radius_um,
        "method": method_key,
        "n_modes": n_modes,
        "round_radii_um": round_radii_um,
        "zernike_coeff_sigfigs": zernike_coeff_sigfigs,
    }


def build_fit_artifacts(
    file_path: Path,
    *,
    metadata: SurfaceMetadata | None,
    source_metadata: SurfaceMetadata | None,
    output_dir: Path,
    method: str,
    n_modes: int,
    maxfev: int,
    rcond: float | None,
    roc_mode: str,
    reference_radius_um: float | None,
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
    metadata = parse_surface_metadata(file_path) if metadata is None else metadata
    source_metadata = parse_surface_metadata(file_path) if source_metadata is None else source_metadata
    if prefit_data is None:
        prefit_data = get_sphere_prefit_entry(file_path, surf_id=metadata.surf_id)

    fit_data = run_fit_pipeline(
        prefit_data.x,
        prefit_data.y,
        prefit_data.z,
        surf_id=metadata.surf_id,
        method=method,
        n_modes=n_modes,
        maxfev=maxfev,
        rcond=rcond,
        reference_radius_um=reference_radius_um,
        normalization_radius_um=normalization_radius_um,
        round_radii_um=round_radii_um,
        zernike_coeff_sigfigs=zernike_coeff_sigfigs,
        prefit_data=prefit_data,
    )

    export_coefficients_um = fit_data["zpoly_fits2"].copy()
    export_coefficients_um[0] = export_coefficients_um[0] - fit_data["zv2"]
    if zernike_coeff_sigfigs is not None:
        export_coefficients_um = round_sigfigs_array(
            export_coefficients_um,
            zernike_coeff_sigfigs,
            np=_load_azp_symbols()["np"],
        )
    export_coefficients_mm = (-1.0) * export_coefficients_um / 1000.0

    export_zernike_coefficients_csv = _load_azp_symbols()["export_zernike_coefficients_csv"]
    coeff_dir = output_dir / "coefficients"
    coeff_path = coeff_dir / make_output_filename(metadata)
    export_zernike_coefficients_csv(
        coeff_path,
        design_id=metadata.design_id,
        fea_id=metadata.fea_id,
        surf_id=metadata.surf_id,
        tension_mn=format_tension(metadata.force_id),
        base_sphere_radius_um=fit_data["applied_reference_radius_um"],
        vertex_um=fit_data["zv"],
        vertex_residual_um=fit_data["zv2"],
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
        sphere_residuals=fit_data["sphere_residuals"],
        zernike_surface=fit_data["zernike_surface"],
        zernike_surface_residuals=fit_data["zernike_surface_residuals"],
        zernike_residual_surface=fit_data["zernike_residual_surface"],
        zernike_residual_residuals=fit_data["zernike_residual_residuals"],
        zpoly_fits=fit_data["zpoly_fits"],
        zpoly_fits2=fit_data["zpoly_fits2"],
        x0_fit=fit_data["x0_fit"],
        y0_fit=fit_data["y0_fit"],
        z0_fit=fit_data["z0_fit"],
        fitted_sphere_radius_um=fit_data["fitted_sphere_radius_um"],
        applied_reference_radius_um=fit_data["applied_reference_radius_um"],
        prefit_best_radius_um=fit_data["prefit_best_radius_um"],
        sphere_sse=fit_data["sphere_sse"],
        sphere_rms=fit_data["sphere_rms"],
        surface_zernike_sse=fit_data["surface_zernike_sse"],
        surface_zernike_rms=fit_data["surface_zernike_rms"],
        surface_zernike_cond=fit_data["surface_zernike_cond"],
        sphere_residual_zernike_sse=fit_data["sphere_residual_zernike_sse"],
        sphere_residual_zernike_rms=fit_data["sphere_residual_zernike_rms"],
        sphere_residual_zernike_cond=fit_data["sphere_residual_zernike_cond"],
        observed_aperture_radius_um=fit_data["observed_aperture_radius_um"],
        norm_radius_um=fit_data["norm_radius_um"],
        vertex_um=fit_data["zv"],
        vertex_residual_um=fit_data["zv2"],
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


def artifacts_to_summary_row(artifacts: FitArtifacts) -> dict[str, Any]:
    """Flatten fit artifacts for batch summary CSV output."""
    return {
        "run_name": artifacts.run_name,
        "roc_mode": artifacts.roc_mode,
        "normalization_mode": artifacts.normalization_mode,
        "round_radii_um": artifacts.round_radii_um,
        "zernike_coeff_sigfigs": artifacts.zernike_coeff_sigfigs,
        "source_file": str(artifacts.source_file),
        "source_force_id": artifacts.source_metadata.force_id,
        "design_token": artifacts.metadata.design_token,
        "design_id": artifacts.metadata.design_id,
        "fea_id": artifacts.metadata.fea_id,
        "force_id": artifacts.metadata.force_id,
        "surf_id": artifacts.metadata.surf_id,
        "surface_token": artifacts.metadata.surface_token,
        "points_used": artifacts.points_used,
        "n_modes": artifacts.n_modes,
        "prefit_best_radius_um": artifacts.prefit_best_radius_um,
        "fitted_sphere_radius_um": artifacts.fitted_sphere_radius_um,
        "applied_reference_radius_um": artifacts.applied_reference_radius_um,
        "common_reference_radius_um": artifacts.common_reference_radius_um,
        "observed_aperture_radius_um": artifacts.observed_aperture_radius_um,
        "applied_normalization_radius_um": artifacts.norm_radius_um,
        "common_normalization_radius_um": artifacts.common_normalization_radius_um,
        "sphere_sse": artifacts.sphere_sse,
        "sphere_rms": artifacts.sphere_rms,
        "surface_zernike_sse": artifacts.surface_zernike_sse,
        "surface_zernike_rms": artifacts.surface_zernike_rms,
        "surface_zernike_cond": artifacts.surface_zernike_cond,
        "sphere_residual_zernike_sse": artifacts.sphere_residual_zernike_sse,
        "sphere_residual_zernike_rms": artifacts.sphere_residual_zernike_rms,
        "sphere_residual_zernike_cond": artifacts.sphere_residual_zernike_cond,
        "norm_radius_um": artifacts.norm_radius_um,
        "vertex_um": artifacts.vertex_um,
        "vertex_residual_um": artifacts.vertex_residual_um,
        "output_coefficients_csv": str(artifacts.output_coefficients_csv),
        "method": artifacts.method,
    }


def write_csv(file_path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a list of dictionary rows to CSV."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        fieldnames = ["message"]
        with file_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow({"message": "no rows"})
        return

    fieldnames = list(rows[0].keys())
    with file_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_batch_zp_report(file_path: Path, artifacts: list[FitArtifacts]) -> Path:
    """Write one wide CSV containing the per-surface ZP report for every processed file."""
    build_rows = _load_azp_symbols()["build_zernike_coefficients_rows"]
    columns: list[tuple[str, list[tuple[str, object]]]] = []

    for item in artifacts:
        export_coefficients_um = item.zpoly_fits2.copy()
        export_coefficients_um[0] = export_coefficients_um[0] - item.vertex_residual_um
        if item.zernike_coeff_sigfigs is not None:
            export_coefficients_um = round_sigfigs_array(
                export_coefficients_um,
                item.zernike_coeff_sigfigs,
                np=_load_azp_symbols()["np"],
            )
        export_coefficients_mm = (-1.0) * export_coefficients_um / 1000.0
        rows = build_rows(
            design_id=item.metadata.design_id,
            fea_id=item.metadata.fea_id,
            surf_id=item.metadata.surf_id,
            tension_mn=format_tension(item.metadata.force_id),
            base_sphere_radius_um=item.applied_reference_radius_um,
            vertex_um=item.vertex_um,
            vertex_residual_um=item.vertex_residual_um,
            norm_radius_um=item.norm_radius_um,
            zernike_coefficients_mm=export_coefficients_mm,
        )
        columns.append((format_processed_label(item.metadata), rows))

    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", newline="") as handle:
        writer = csv.writer(handle, dialect="excel")
        writer.writerow(["Field", *[label for label, _ in columns]])
        if not columns:
            writer.writerow(["message", "no rows"])
            return file_path

        row_names = [name for name, _ in columns[0][1]]
        for row_index, row_name in enumerate(row_names):
            writer.writerow(
                [
                    row_name,
                    *[
                        rows[row_index][1]
                        for _, rows in columns
                    ],
                ]
            )
    return file_path


def excel_column_name(index: int) -> str:
    """Convert a zero-based column index into Excel's A1-style column letters."""
    result = ""
    current = index + 1
    while current:
        current, remainder = divmod(current - 1, 26)
        result = chr(65 + remainder) + result
    return result


def write_xlsx(file_path: Path, rows: list[dict[str, Any]], *, sheet_name: str = "Summary") -> None:
    """Write a minimal single-sheet XLSX workbook without external spreadsheet dependencies."""
    from xml.sax.saxutils import escape
    from zipfile import ZIP_DEFLATED, ZipFile

    file_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        headers = list(rows[0].keys())
        records = [headers, *[[row.get(header, "") for header in headers] for row in rows]]
    else:
        records = [["message"], ["no rows"]]

    xml_rows: list[str] = []
    for row_index, values in enumerate(records, start=1):
        cells: list[str] = []
        for col_index, value in enumerate(values):
            cell_ref = f"{excel_column_name(col_index)}{row_index}"
            if isinstance(value, bool):
                cells.append(f'<c r="{cell_ref}" t="b"><v>{1 if value else 0}</v></c>')
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                cells.append(f'<c r="{cell_ref}"><v>{value}</v></c>')
            else:
                text = "" if value is None else str(value)
                cells.append(
                    f'<c r="{cell_ref}" t="inlineStr"><is><t xml:space="preserve">{escape(text)}</t></is></c>'
                )
        xml_rows.append(f'<row r="{row_index}">{"".join(cells)}</row>')

    worksheet_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>
    {''.join(xml_rows)}
  </sheetData>
</worksheet>'''
    workbook_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="{sheet_name}" sheetId="1" r:id="rId1"/>
  </sheets>
</workbook>'''
    workbook_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>'''
    package_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>'''
    content_types = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
</Types>'''

    with ZipFile(file_path, "w", ZIP_DEFLATED) as workbook:
        workbook.writestr("[Content_Types].xml", content_types)
        workbook.writestr("_rels/.rels", package_rels)
        workbook.writestr("xl/workbook.xml", workbook_xml)
        workbook.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        workbook.writestr("xl/worksheets/sheet1.xml", worksheet_xml)


def write_json(file_path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON manifest for one batch run."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def resolve_h5_path(h5_path: Path | None, output_root: Path) -> Path | None:
    """Resolve a requested HDF5 path relative to the output root when needed."""
    if h5_path is None:
        return None
    return h5_path if h5_path.is_absolute() else output_root / h5_path


def write_batch_h5(
    h5_path: Path,
    *,
    run_name: str,
    config: dict[str, Any],
    artifacts: list[FitArtifacts],
) -> None:
    """Append one batch run into a shared HDF5 file."""
    import h5py

    h5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "a") as handle:
        runs_group = handle.require_group("runs")
        if run_name in runs_group:
            del runs_group[run_name]
        run_group = runs_group.create_group(run_name)

        for key, value in config.items():
            if value is None:
                continue
            run_group.attrs[key] = value

        files_group = run_group.create_group("files")
        for index, item in enumerate(artifacts):
            group_name = f"{index:04d}_{sanitize_h5_name(format_processed_label(item.metadata))}"
            file_group = files_group.create_group(group_name)
            file_group.attrs["source_file"] = str(item.source_file)
            file_group.attrs["output_coefficients_csv"] = str(item.output_coefficients_csv)
            file_group.attrs["source_force_id"] = item.source_metadata.force_id
            file_group.attrs["design_token"] = item.metadata.design_token
            file_group.attrs["design_id"] = item.metadata.design_id
            file_group.attrs["fea_id"] = item.metadata.fea_id
            file_group.attrs["force_id"] = item.metadata.force_id
            file_group.attrs["surface_token"] = item.metadata.surface_token
            file_group.attrs["surf_id"] = item.metadata.surf_id
            file_group.attrs["roc_mode"] = item.roc_mode
            file_group.attrs["normalization_mode"] = item.normalization_mode
            file_group.attrs["round_radii_um"] = item.round_radii_um
            if item.zernike_coeff_sigfigs is not None:
                file_group.attrs["zernike_coeff_sigfigs"] = item.zernike_coeff_sigfigs
            file_group.attrs["method"] = item.method
            file_group.attrs["n_modes"] = item.n_modes
            file_group.attrs["points_used"] = item.points_used
            if item.prefit_best_radius_um is not None:
                file_group.attrs["prefit_best_radius_um"] = item.prefit_best_radius_um
            file_group.attrs["fitted_sphere_radius_um"] = item.fitted_sphere_radius_um
            file_group.attrs["applied_reference_radius_um"] = item.applied_reference_radius_um
            if item.common_reference_radius_um is not None:
                file_group.attrs["common_reference_radius_um"] = item.common_reference_radius_um
            file_group.attrs["observed_aperture_radius_um"] = item.observed_aperture_radius_um
            file_group.attrs["applied_normalization_radius_um"] = item.norm_radius_um
            if item.common_normalization_radius_um is not None:
                file_group.attrs["common_normalization_radius_um"] = item.common_normalization_radius_um

            file_group.create_dataset("x_um", data=item.x)
            file_group.create_dataset("y_um", data=item.y)
            file_group.create_dataset("z_um", data=item.z)
            file_group.create_dataset("rho_um", data=item.rho)
            file_group.create_dataset("phi_rad", data=item.phi)
            file_group.create_dataset("rho_norm", data=item.rho_norm)
            file_group.create_dataset("sphere_residual_um", data=item.sphere_residuals)
            file_group.create_dataset("zernike_surface_um", data=item.zernike_surface)
            file_group.create_dataset("zernike_surface_residual_um", data=item.zernike_surface_residuals)
            file_group.create_dataset("zernike_residual_surface_um", data=item.zernike_residual_surface)
            file_group.create_dataset(
                "zernike_residual_surface_residual_um",
                data=item.zernike_residual_residuals,
            )
            file_group.create_dataset("zernike_surface_coefficients", data=item.zpoly_fits)
            file_group.create_dataset("zernike_residual_coefficients", data=item.zpoly_fits2)


def sphere_profile_z(rho: Any, *, z0_fit: float, radius_um: float, posterior_surface: bool, np: Any) -> Any:
    """Evaluate the fitted sphere branch used by the notebook's rho-z plot."""
    term = np.sqrt(np.clip(radius_um**2 - rho**2, 0.0, None))
    return z0_fit - term if posterior_surface else z0_fit + term


def radial_bin_profile(rho: Any, values: Any, *, bins: int, np: Any) -> tuple[Any, Any]:
    """Bin scattered rho/value pairs into a compact median profile for thumbnails."""
    rho_arr = np.asarray(rho, dtype=float)
    values_arr = np.asarray(values, dtype=float)
    mask = np.isfinite(rho_arr) & np.isfinite(values_arr)
    rho_arr = rho_arr[mask]
    values_arr = values_arr[mask]
    if rho_arr.size == 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    rho_min = float(np.min(rho_arr))
    rho_max = float(np.max(rho_arr))
    if np.isclose(rho_min, rho_max):
        return np.asarray([rho_min], dtype=float), np.asarray([float(np.median(values_arr))], dtype=float)

    edges = np.linspace(rho_min, rho_max, bins + 1)
    centers: list[float] = []
    medians: list[float] = []
    for start, end in zip(edges[:-1], edges[1:], strict=False):
        if end == edges[-1]:
            bucket = (rho_arr >= start) & (rho_arr <= end)
        else:
            bucket = (rho_arr >= start) & (rho_arr < end)
        if not np.any(bucket):
            continue
        centers.append(float(np.median(rho_arr[bucket])))
        medians.append(float(np.median(values_arr[bucket])))

    return np.asarray(centers, dtype=float), np.asarray(medians, dtype=float)


def write_thumbnail_plot(file_path: Path, artifacts: FitArtifacts, *, bins: int = 64) -> None:
    """Write one compact QA thumbnail with measured/fitted radial profiles and residuals."""
    mpl_config_dir = file_path.parent.parent / ".matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    symbols = _load_azp_symbols()
    np = symbols["np"]

    posterior_surface = uses_posterior_sign_convention(artifacts.metadata.surf_id)
    sphere_surface = sphere_profile_z(
        artifacts.rho,
        z0_fit=artifacts.z0_fit,
        radius_um=artifacts.applied_reference_radius_um,
        posterior_surface=posterior_surface,
        np=np,
    )

    rho_meas, z_meas = radial_bin_profile(artifacts.rho, artifacts.z, bins=bins, np=np)
    rho_sphere, z_sphere = radial_bin_profile(artifacts.rho, sphere_surface, bins=bins, np=np)
    rho_zernike, z_zernike = radial_bin_profile(artifacts.rho, artifacts.zernike_surface, bins=bins, np=np)
    rho_resid, z_resid = radial_bin_profile(artifacts.rho, artifacts.zernike_surface_residuals, bins=bins, np=np)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(4.8, 2.2),
        dpi=110,
        gridspec_kw={"height_ratios": [3.0, 1.2]},
        sharex=True,
    )

    axes[0].plot(rho_meas, z_meas, color="#1f2937", linewidth=1.1, label="measured")
    axes[0].plot(rho_sphere, z_sphere, color="#2563eb", linewidth=0.9, label="sphere")
    axes[0].plot(rho_zernike, z_zernike, color="#dc2626", linewidth=0.9, label="zernike")
    axes[0].set_ylabel("z (um)", fontsize=7)
    axes[0].grid(True, alpha=0.2, linewidth=0.4)
    axes[0].tick_params(labelsize=6, length=2)
    axes[0].legend(loc="best", fontsize=6, frameon=False, ncol=3, handlelength=1.5, columnspacing=0.8)
    axes[0].set_title(
        f"{artifacts.metadata.surface_token}  sse={artifacts.sphere_sse:.2e}  rms={artifacts.surface_zernike_rms:.2e}",
        fontsize=7,
    )

    axes[1].plot(rho_resid, z_resid, color="#059669", linewidth=0.9)
    axes[1].axhline(0.0, color="#6b7280", linewidth=0.6, alpha=0.6)
    axes[1].set_xlabel("rho (um)", fontsize=7)
    axes[1].set_ylabel("resid", fontsize=7)
    axes[1].grid(True, alpha=0.2, linewidth=0.4)
    axes[1].tick_params(labelsize=6, length=2)

    fig.tight_layout(pad=0.4)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(file_path, bbox_inches="tight")
    plt.close(fig)


def write_qa_report(
    run_dir: Path,
    artifacts: list[FitArtifacts],
    *,
    summary_report_path: Path,
    analysis_date: str,
) -> Path:
    """Generate a browsable HTML gallery of compact fit thumbnails for the batch."""
    qa_dir = run_dir / "qa"
    thumbnails_dir = qa_dir / "thumbnails"
    thumbnails_dir.mkdir(parents=True, exist_ok=True)

    plotted = [item for item in artifacts if is_focus_surface_family(item.metadata.surf_id)]
    skipped = len(artifacts) - len(plotted)

    for item in plotted:
        write_thumbnail_plot(thumbnails_dir / f"{sanitize_h5_name(format_processed_label(item.metadata))}.png", item)

    ranked = sorted(plotted, key=lambda item: item.sphere_sse, reverse=True)
    flagged = ranked[: min(40, len(ranked))]

    def build_rows(items: list[FitArtifacts]) -> str:
        rows: list[str] = []
        for item in items:
            thumb_name = f"{sanitize_h5_name(format_processed_label(item.metadata))}.png"
            coeff_rel = Path("..") / item.output_coefficients_csv.relative_to(run_dir)
            file_label = format_processed_label(item.metadata)
            if item.metadata.force_id != item.source_metadata.force_id:
                file_label = f"{file_label} [{item.source_file.name}]"
            rows.append(
                "<tr>"
                f"<td>{html.escape(file_label)}</td>"
                f"<td>{html.escape(item.metadata.surface_token)}</td>"
                f"<td>{item.sphere_sse:.3e}</td>"
                f"<td>{item.surface_zernike_rms:.3e}</td>"
                f"<td>{item.sphere_residual_zernike_rms:.3e}</td>"
                f"<td><a href='{html.escape(coeff_rel.as_posix())}'>coefficients</a></td>"
                f"<td><img loading='lazy' src='thumbnails/{html.escape(thumb_name)}' alt='{html.escape(item.source_file.name)}'></td>"
                "</tr>"
            )
        return "\n".join(rows)

    html_path = qa_dir / "index.html"
    payload = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Batch QA Report - {html.escape(analysis_date)}</title>
  <style>
    body {{
      font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 24px;
      color: #111827;
      background: #f8fafc;
    }}
    h1, h2 {{ margin: 0 0 12px; }}
    p {{ margin: 0 0 16px; }}
    table {{
      border-collapse: collapse;
      width: 100%;
      margin: 0 0 32px;
      background: white;
    }}
    th, td {{
      border-bottom: 1px solid #e5e7eb;
      padding: 8px 10px;
      text-align: left;
      vertical-align: top;
      font-size: 12px;
    }}
    th {{
      position: sticky;
      top: 0;
      background: #eff6ff;
      z-index: 1;
    }}
    img {{
      width: 420px;
      max-width: 100%;
      border: 1px solid #cbd5e1;
      background: white;
    }}
    .meta {{
      color: #475569;
      font-size: 13px;
    }}
  </style>
</head>
<body>
  <h1>Batch QA Report - {html.escape(analysis_date)}</h1>
  <p class="meta">Run: {html.escape(run_dir.name)} | Focus files: {len(plotted)} | Skipped non-focus families: {skipped} | Sorted flagged section by sphere_sse descending.</p>
  <p class="meta">Plotted families: {", ".join(sorted(FOCUS_SURF_IDS))}</p>
  <p><a href="../{html.escape(summary_report_path.name)}">{html.escape(summary_report_path.name)}</a> | <a href="../run_manifest.json">run_manifest.json</a></p>

  <h2>Flagged</h2>
  <table>
    <thead>
      <tr>
        <th>File</th>
        <th>Surface</th>
        <th>Sphere SSE</th>
        <th>Surface RMS</th>
        <th>Residual RMS</th>
        <th>CSV</th>
        <th>Preview</th>
      </tr>
    </thead>
    <tbody>
      {build_rows(flagged)}
    </tbody>
  </table>

  <h2>All Files</h2>
  <table>
    <thead>
      <tr>
        <th>File</th>
        <th>Surface</th>
        <th>Sphere SSE</th>
        <th>Surface RMS</th>
        <th>Residual RMS</th>
        <th>CSV</th>
        <th>Preview</th>
      </tr>
    </thead>
    <tbody>
      {build_rows(artifacts)}
    </tbody>
  </table>
</body>
</html>
"""
    html_path.write_text(payload)
    return html_path


def precompute_best_radii(
    items: list[ProcessingInput],
) -> tuple[dict[Path, float], list[dict[str, str]]]:
    """Fit a best sphere ROC for each file ahead of a common-radius batch run."""
    radii: dict[Path, float] = {}
    failures: list[dict[str, str]] = []

    for item in items:
        try:
            entry = get_sphere_prefit_entry(item.source_file, surf_id=item.metadata.surf_id)
            radii[item.source_file] = entry.best_radius_um
        except Exception as exc:
            failures.append({"source_file": str(item.source_file), "error": f"prefit radius failed: {exc}"})

    return radii, failures


def precompute_common_normalization_radii_by_surf_id(
    items: list[ProcessingInput],
    *,
    round_radii_um: bool,
) -> tuple[dict[str, float], dict[Path, float], list[dict[str, str]]]:
    """Derive one shared normalization radius per surf_id, chosen to cover all files in that family."""
    per_file_observed: dict[Path, float] = {}
    per_surf_id: dict[str, float] = {}
    failures: list[dict[str, str]] = []

    for item in items:
        try:
            entry = get_sphere_prefit_entry(item.source_file, surf_id=item.metadata.surf_id)
            observed_radius = entry.observed_aperture_radius_um
            per_file_observed[item.source_file] = observed_radius
            current = per_surf_id.get(item.metadata.surf_id)
            per_surf_id[item.metadata.surf_id] = (
                observed_radius if current is None else max(current, observed_radius)
            )
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


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the batch CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Batch-fit notebook-style XYZ point clouds without previews. Only AA/AP/PA/PP files are included."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing notebook-style .xyz files.",
    )
    parser.add_argument(
        "--glob",
        default="*.xyz",
        help="File glob to match inside input-dir. Default: %(default)s",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("batch_outputs"),
        help="Root directory for batch runs. Each run is written into a separate folder.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run folder / HDF5 group name. Defaults to a timestamp.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search recursively under input-dir.",
    )
    parser.add_argument(
        "--method",
        choices=("curve_fit", "lstsq"),
        default="lstsq",
        help="Zernike fitting method. Default: %(default)s",
    )
    parser.add_argument(
        "--roc-mode",
        choices=("fit-per-file", "fixed", "average-best-fit"),
        default="fit-per-file",
        help="How to choose the reference radius of curvature for the batch.",
    )
    parser.add_argument(
        "--fixed-roc-um",
        type=float,
        default=None,
        help="Fixed radius of curvature in micrometers for --roc-mode fixed.",
    )
    parser.add_argument(
        "--h5-path",
        type=Path,
        default=None,
        help="Optional HDF5 file that stores all runs together under separate run groups.",
    )
    parser.add_argument(
        "--n-modes",
        type=int,
        default=45,
        help="Number of Zernike modes to fit. Default: %(default)s",
    )
    parser.add_argument(
        "--maxfev",
        type=int,
        default=10000,
        help="Maximum function evaluations for curve-fit mode. Default: %(default)s",
    )
    parser.add_argument(
        "--rcond",
        type=float,
        default=None,
        help="Least-squares rcond override for lstsq mode.",
    )
    parser.add_argument(
        "--normalization-mode",
        choices=("per-file", "common-per-surf-id"),
        default="per-file",
        help="How to choose the Zernike normalization radius. Default: %(default)s",
    )
    parser.add_argument(
        "--round-radii-um",
        dest="round_radii_um",
        action="store_true",
        help="Round applied sphere and normalization radii to the nearest whole micrometer before fitting.",
    )
    parser.add_argument(
        "--no-round-radii-um",
        dest="round_radii_um",
        action="store_false",
        help="Disable radius rounding and keep full-precision reference and normalization radii.",
    )
    parser.add_argument(
        "--zernike-coeff-sigfigs",
        type=int,
        default=6,
        help="Round fitted Zernike coefficients to the requested significant digits before recomputing residuals.",
    )
    parser.add_argument(
        "--no-round-zernike-coeffs",
        dest="zernike_coeff_sigfigs",
        action="store_const",
        const=None,
        help="Disable Zernike coefficient rounding and keep full-precision coefficients.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of files processed.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first processing error instead of writing a failure report.",
    )
    parser.add_argument(
        "--qa-report",
        action="store_true",
        help="Write an HTML QA gallery with compact radial-profile thumbnails for each fitted file.",
    )
    parser.set_defaults(round_radii_um=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    input_dir = args.input_dir.resolve()
    if not input_dir.exists():
        parser.error(f"Input directory does not exist: {input_dir}")
    if not 1 <= args.n_modes <= 45:
        parser.error("--n-modes must be between 1 and 45.")
    if args.roc_mode == "fixed" and args.fixed_roc_um is None:
        parser.error("--fixed-roc-um is required when --roc-mode fixed is selected.")
    if args.fixed_roc_um is not None and args.fixed_roc_um <= 0:
        parser.error("--fixed-roc-um must be positive.")
    if args.zernike_coeff_sigfigs is not None and args.zernike_coeff_sigfigs < 1:
        parser.error("--zernike-coeff-sigfigs must be positive.")

    matcher = input_dir.rglob if args.recursive else input_dir.glob
    matched_files = sorted(path for path in matcher(args.glob) if path.is_file())
    files = [
        path
        for path in matched_files
        if is_focus_surface_family(parse_surface_metadata(path).surf_id)
    ]
    excluded_files = len(matched_files) - len(files)
    if not files:
        parser.error(
            f"No AA/AP/PA/PP files matched {args.glob!r} under {input_dir}. "
            f"Matched files before filtering: {len(matched_files)}"
        )
    if excluded_files:
        print(
            f"excluded {excluded_files} non-focus files; processing only {len(files)} AA/AP/PA/PP files",
            file=sys.stderr,
        )

    try:
        processing_inputs = collapse_identical_initial_inputs(files)
    except ValueError as exc:
        parser.error(str(exc))
    if args.limit is not None:
        processing_inputs = processing_inputs[: args.limit]
    if not processing_inputs:
        parser.error("No effective inputs remain after collapsing duplicate *I states.")

    run_name = resolve_run_name(args.run_name)
    analysis_date = resolve_analysis_date()
    output_root = args.output_dir.resolve()
    run_dir = output_root / run_name
    h5_path = resolve_h5_path(args.h5_path, output_root)
    summary_report_path = run_dir / f"batch_summary_{analysis_date}.xlsx"

    files_to_process = processing_inputs
    common_radius_um: float | None = None
    common_radius_by_surf_id: dict[str, float] = {}
    normalization_radius_by_surf_id: dict[str, float] = {}
    prefit_radii: dict[Path, float] = {}
    failures: list[dict[str, str]] = []

    if args.roc_mode == "average-best-fit":
        prefit_radii, prefit_failures = precompute_best_radii(files_to_process)
        failures.extend(prefit_failures)
        files_to_process = [item for item in files_to_process if item.source_file in prefit_radii]
        if not files_to_process:
            write_csv(run_dir / "batch_failures.csv", failures)
            print("No files survived the prefit-radius stage.", file=sys.stderr)
            return 1
        # Compute per-surf_id average radii so physically different surface
        # families are not mixed into one global average.
        radii_by_surf: dict[str, list[float]] = {}
        for path, radius in prefit_radii.items():
            sid = parse_surface_metadata(path).surf_id
            radii_by_surf.setdefault(sid, []).append(radius)
        for sid in sorted(radii_by_surf):
            avg = sum(radii_by_surf[sid]) / len(radii_by_surf[sid])
            common_radius_by_surf_id[sid] = avg
            print(f"common best ROC {sid}: {avg:.6f} um")
        # Keep a single global average for backward-compatible manifest output.
        common_radius_um = sum(prefit_radii.values()) / len(prefit_radii)
        if args.round_radii_um:
            common_radius_um = clamp_reference_radius_um(
                round_nearest_micrometer(common_radius_um),
                max(
                    get_sphere_prefit_entry(
                        item.source_file,
                        surf_id=item.metadata.surf_id,
                    ).raw_aperture_radius_um
                    for item in files_to_process
                ),
            )
        print(f"common best ROC: {common_radius_um:.6f} um")

    if args.normalization_mode == "common-per-surf-id":
        normalization_radius_by_surf_id, _, normalization_failures = precompute_common_normalization_radii_by_surf_id(
            files_to_process,
            round_radii_um=args.round_radii_um,
        )
        failures.extend(normalization_failures)
        files_to_process = [
            item
            for item in files_to_process
            if item.metadata.surf_id in normalization_radius_by_surf_id
        ]
        if not files_to_process:
            write_csv(run_dir / "batch_failures.csv", failures)
            print("No files survived the normalization-radius prefit stage.", file=sys.stderr)
            return 1
        for surf_id in sorted(normalization_radius_by_surf_id):
            print(
                f"common normalization radius {surf_id}: {normalization_radius_by_surf_id[surf_id]:.6f} um"
            )

    artifacts: list[FitArtifacts] = []
    summary_rows: list[dict[str, Any]] = []

    for item_input in files_to_process:
        metadata = item_input.metadata
        file_path = item_input.source_file
        if args.roc_mode == "fit-per-file":
            reference_radius_um = None
        elif args.roc_mode == "fixed":
            reference_radius_um = args.fixed_roc_um
        else:
            reference_radius_um = common_radius_by_surf_id.get(metadata.surf_id, common_radius_um)
        if args.normalization_mode == "common-per-surf-id":
            normalization_radius_um = normalization_radius_by_surf_id.get(metadata.surf_id)
        else:
            normalization_radius_um = None

        try:
            prefit_data = get_sphere_prefit_entry(file_path, surf_id=metadata.surf_id)
            item = build_fit_artifacts(
                file_path,
                metadata=metadata,
                source_metadata=item_input.source_metadata,
                output_dir=run_dir,
                method=args.method,
                n_modes=args.n_modes,
                maxfev=args.maxfev,
                rcond=args.rcond,
                roc_mode=args.roc_mode,
                reference_radius_um=reference_radius_um,
                normalization_mode=args.normalization_mode,
                normalization_radius_um=normalization_radius_um,
                run_name=run_name,
                common_reference_radius_um=reference_radius_um,
                common_normalization_radius_um=normalization_radius_um,
                round_radii_um=args.round_radii_um,
                zernike_coeff_sigfigs=args.zernike_coeff_sigfigs,
                prefit_data=prefit_data,
            )
            if args.roc_mode == "average-best-fit":
                item.prefit_best_radius_um = prefit_radii.get(file_path)
            artifacts.append(item)
            summary_rows.append(artifacts_to_summary_row(item))
            print(f"processed: {file_path}")
        except Exception as exc:  # pragma: no cover - batch reporting path
            failure = {"source_file": str(file_path), "error": str(exc)}
            failures.append(failure)
            print(f"failed: {file_path} :: {exc}", file=sys.stderr)
            if args.fail_fast:
                raise

    write_xlsx(summary_report_path, summary_rows)
    batch_zp_report_path = write_batch_zp_report(run_dir / "coefficients" / "ZPs_batch_report.csv", artifacts)
    if failures:
        write_csv(run_dir / "batch_failures.csv", failures)

    write_json(
        run_dir / "run_manifest.json",
        {
            "run_name": run_name,
            "input_dir": str(input_dir),
            "glob": args.glob,
            "recursive": args.recursive,
            "focus_surf_ids": sorted(FOCUS_SURF_IDS),
            "matched_files_before_filter": len(matched_files),
            "excluded_non_focus_files": excluded_files,
            "effective_inputs_after_collapse": len(processing_inputs),
            "method": args.method,
            "roc_mode": args.roc_mode,
            "normalization_mode": args.normalization_mode,
            "fixed_roc_um": args.fixed_roc_um,
            "common_reference_radius_um": common_radius_um,
            "common_reference_radius_by_surf_id": common_radius_by_surf_id or None,
            "summary_report_path": str(summary_report_path),
            "batch_zp_report_path": str(batch_zp_report_path),
            "n_modes": args.n_modes,
            "maxfev": args.maxfev,
            "rcond": args.rcond,
            "round_radii_um": args.round_radii_um,
            "zernike_coeff_sigfigs": args.zernike_coeff_sigfigs,
            "h5_path": str(h5_path) if h5_path is not None else None,
            "qa_report": args.qa_report,
            "processed_files": len(artifacts),
            "failed_files": len(failures),
        },
    )

    if h5_path is not None and artifacts:
        write_batch_h5(
            h5_path,
            run_name=run_name,
            config={
                "input_dir": str(input_dir),
                "glob": args.glob,
                "method": args.method,
                "roc_mode": args.roc_mode,
                "normalization_mode": args.normalization_mode,
                "fixed_roc_um": args.fixed_roc_um,
                "common_reference_radius_um": common_radius_um,
                "common_reference_radius_by_surf_id": json.dumps(common_radius_by_surf_id) if common_radius_by_surf_id else None,
                "n_modes": args.n_modes,
                "maxfev": args.maxfev,
                "rcond": args.rcond,
                "round_radii_um": args.round_radii_um,
                "zernike_coeff_sigfigs": args.zernike_coeff_sigfigs,
                "run_dir": str(run_dir),
            },
            artifacts=artifacts,
        )

    if args.qa_report and artifacts:
        qa_path = write_qa_report(
            run_dir,
            artifacts,
            summary_report_path=summary_report_path,
            analysis_date=analysis_date,
        )
        print(f"qa report: {qa_path}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
