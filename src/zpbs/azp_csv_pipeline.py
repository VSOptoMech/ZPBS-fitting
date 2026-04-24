"""Core functions for Zernike fitting on point-cloud and matrix CSV inputs."""

# Last revision date: 2026-04-03
# Latest git commit: 74f1f04

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

try:
    import plotly.graph_objects as go
except ModuleNotFoundError:  # pragma: no cover - plotting is optional for batch-only environments
    go = None

N_ZERNIKE_TERMS = 45
NOLL_TO_J_INDEX = np.asarray(
    [
        0,
        2,
        1,
        4,
        3,
        5,
        7,
        8,
        6,
        9,
        12,
        13,
        11,
        14,
        10,
        18,
        17,
        19,
        16,
        20,
        15,
        24,
        23,
        25,
        22,
        26,
        21,
        27,
        31,
        32,
        30,
        33,
        29,
        34,
        28,
        35,
        40,
        41,
        39,
        42,
        38,
        43,
        37,
        44,
        36,
    ]
)


def uses_posterior_sign_convention(surf_id: str) -> bool:
    """Apply the notebook's posterior residual convention to posterior regional surfaces too."""
    return surf_id.upper().startswith("P")


@dataclass
class HeightMapData:
    """Parsed Keyence height map represented as grid and flattened point cloud."""

    source_file: Path
    metadata: dict[str, Any]
    z_grid: np.ndarray
    valid_mask: np.ndarray
    x_grid: np.ndarray
    y_grid: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    pixel_size_um: float
    z_unit: str


@dataclass
class SurfaceFitResults:
    """All outputs produced by the sphere + Zernike fitting pipeline."""

    x0_fit: float
    y0_fit: float
    z0_fit: float
    r_fit: float
    best_sphere_resid: np.ndarray
    sphere_sse_um2: float
    sphere_mae_um: float
    sphere_rms_um: float
    rho: np.ndarray
    phi: np.ndarray
    rho_norm: np.ndarray
    pol_loci: np.ndarray
    zv: float
    zv2: float
    zpbs_residual_coefficients_um: np.ndarray
    zpbs_residual_surface_um: np.ndarray
    zpbs_residual_residuals_um: np.ndarray
    zpbs_residual_sse_um2: float
    zpbs_residual_mae_um: float
    zpbs_residual_rms_um: float
    zpbs_residual_cond: float
    zpbs_to_data_surface_um: np.ndarray
    zpbs_to_data_residuals_um: np.ndarray
    zpbs_to_data_sse_um2: float
    zpbs_to_data_mae_um: float
    zpbs_to_data_rms_um: float
    zernike_method: str

    @property
    def sum_square_resid(self) -> float:
        return self.sphere_sse_um2

    @property
    def sphere_rms(self) -> float:
        return self.sphere_rms_um

    @property
    def zpoly_fits(self) -> np.ndarray:
        return self.zpbs_residual_coefficients_um

    @property
    def zpoly_fits2(self) -> np.ndarray:
        return self.zpbs_residual_coefficients_um

    @property
    def zpoly_surf2(self) -> np.ndarray:
        return self.zpbs_residual_surface_um

    @property
    def zpoly_surf_resid2(self) -> np.ndarray:
        return self.zpbs_residual_residuals_um

    @property
    def zpoly_surf_sse2(self) -> float:
        return self.zpbs_residual_sse_um2

    @property
    def zpoly_surf_rms2(self) -> float:
        return self.zpbs_residual_rms_um

    @property
    def zpoly_surf_cond2(self) -> float:
        return self.zpbs_residual_cond

    @property
    def zpoly_surf(self) -> np.ndarray:
        return self.zpbs_to_data_surface_um

    @property
    def zpoly_surf_resid(self) -> np.ndarray:
        return self.zpbs_to_data_residuals_um

    @property
    def zpoly_surf_sse(self) -> float:
        return self.zpbs_to_data_sse_um2

    @property
    def zpoly_surf_rms(self) -> float:
        return self.zpbs_to_data_rms_um

    @property
    def zpoly_surf_cond(self) -> float:
        return self.zpbs_residual_cond


def _sphere_surface_from_reference(
    rho: np.ndarray, *, z0_fit: float, radius_um: float, branch_sign: float
) -> np.ndarray:
    """Evaluate the fitted sphere branch in the measured z frame."""
    term = np.sqrt(np.clip(radius_um**2 - rho**2, 0.0, None))
    return z0_fit + (float(branch_sign) * term)


def _clean_field(value: str) -> str:
    """Normalize CSV text cells by stripping whitespace and CR/LF artifacts."""
    return value.replace("\r", "").replace("\n", "").strip()


def _to_float(value: str) -> float:
    """Convert CSV scalar to float with a clear error message."""
    cleaned = _clean_field(value)
    if cleaned == "":
        raise ValueError("Cannot convert empty string to float.")
    return float(cleaned)


def _to_int(value: str) -> int:
    """Convert CSV scalar to int with support for numeric-looking floats."""
    return int(round(_to_float(value)))


def load_keyence_height_csv(
    file_path: str | Path,
    *,
    center_origin: bool = True,
    downsample_stride: int = 1,
    max_points: int | None = None,
    random_seed: int = 42,
) -> HeightMapData:
    """
    Load a Keyence height-map CSV and return both grid and flattened point-cloud views.

    The input format is expected to contain a metadata block followed by a "Height" marker
    row and then a 2D matrix of height values where empty fields represent invalid points.
    """
    source_file = Path(file_path)
    if not source_file.exists():
        raise FileNotFoundError(f"CSV file not found: {source_file}")

    with source_file.open(newline="") as handle:
        rows = list(csv.reader(handle))

    if not rows:
        raise ValueError(f"CSV file is empty: {source_file}")

    height_row_idx: int | None = None
    for idx, row in enumerate(rows):
        if len(row) == 1 and _clean_field(row[0]) == "Height":
            height_row_idx = idx
            break

    if height_row_idx is None:
        raise ValueError("Could not find 'Height' marker row in CSV.")

    metadata_rows = rows[:height_row_idx]
    matrix_rows = rows[height_row_idx + 1 :]

    metadata: dict[str, Any] = {}
    for row in metadata_rows:
        if len(row) < 2:
            continue
        key = _clean_field(row[0])
        value = _clean_field(row[1])
        if key == "":
            continue
        metadata[key] = value
        if len(row) >= 3 and _clean_field(row[2]) != "":
            metadata[f"{key} Unit"] = _clean_field(row[2])

    required_keys = ("Horizontal", "Vertical", "XY Calibration", "Unit")
    missing = [k for k in required_keys if k not in metadata]
    if missing:
        raise ValueError(f"Missing required metadata fields: {missing}")

    horizontal = _to_int(str(metadata["Horizontal"]))
    vertical = _to_int(str(metadata["Vertical"]))
    xy_calibration = _to_float(str(metadata["XY Calibration"]))
    xy_calibration_unit = str(metadata.get("XY Calibration Unit", "um")).lower()

    if xy_calibration_unit == "nm":
        pixel_size_um = xy_calibration / 1000.0
    elif xy_calibration_unit in {"um", "micrometer", "micrometers"}:
        pixel_size_um = xy_calibration
    else:
        raise ValueError(f"Unsupported XY Calibration unit: {xy_calibration_unit}")

    z_unit = str(metadata["Unit"])

    if len(matrix_rows) < vertical:
        raise ValueError(f"Height matrix has fewer rows ({len(matrix_rows)}) than expected ({vertical}).")
    if len(matrix_rows) > vertical:
        matrix_rows = matrix_rows[:vertical]

    z_grid = np.full((vertical, horizontal), np.nan, dtype=float)
    for r_idx, row in enumerate(matrix_rows):
        col_count = min(len(row), horizontal)
        for c_idx in range(col_count):
            value = _clean_field(row[c_idx])
            if value == "":
                continue
            z_grid[r_idx, c_idx] = float(value)

    valid_mask = np.isfinite(z_grid)

    x_axis = np.arange(horizontal, dtype=float) * pixel_size_um
    y_axis = np.arange(vertical, dtype=float) * pixel_size_um
    if center_origin:
        x_axis -= x_axis.mean()
        y_axis -= y_axis.mean()

    # Positive y points upward in plots.
    y_axis = y_axis[::-1]

    x_grid, y_grid = np.meshgrid(x_axis, y_axis)

    sample_mask = valid_mask.copy()
    if downsample_stride > 1:
        stride_mask = np.zeros_like(sample_mask, dtype=bool)
        stride_mask[::downsample_stride, ::downsample_stride] = True
        sample_mask &= stride_mask

    if max_points is not None:
        flat_idx = np.flatnonzero(sample_mask.ravel())
        if len(flat_idx) > max_points:
            rng = np.random.default_rng(random_seed)
            keep = rng.choice(flat_idx, size=max_points, replace=False)
            sampled = np.zeros(sample_mask.size, dtype=bool)
            sampled[keep] = True
            sample_mask = sampled.reshape(sample_mask.shape)

    x = x_grid[sample_mask]
    y = y_grid[sample_mask]
    z = z_grid[sample_mask]

    return HeightMapData(
        source_file=source_file,
        metadata=metadata,
        z_grid=z_grid,
        valid_mask=valid_mask,
        x_grid=x_grid,
        y_grid=y_grid,
        x=x,
        y=y,
        z=z,
        pixel_size_um=pixel_size_um,
        z_unit=z_unit,
    )


def fit_sphere(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[float, float, float, float, np.ndarray]:
    """Fit a sphere to 3D points and return center, radius, and signed residuals."""
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    def sphere_residuals(params: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, z_val: np.ndarray) -> np.ndarray:
        x0, y0, z0, r = params
        distances = np.sqrt((x_val - x0) ** 2 + (y_val - y0) ** 2 + (z_val - z0) ** 2)
        return distances - r

    rho = np.sqrt(x**2 + y**2)
    rho_edge = np.max(rho)
    edge_idx = np.where(rho == rho_edge)[0]
    ze = np.max(z[edge_idx])
    zv = np.min(z)
    sag = zv - ze
    if np.isclose(sag, 0.0):
        raise ValueError("Degenerate sphere estimate: sag is approximately zero.")

    r0 = (sag**2 + rho_edge**2) / (2.0 * sag)
    z0 = zv - r0
    # Keep the signed notebook-style seed so the optimizer can converge to the
    # physically correct sphere branch for AP/PP-style caps.
    initial_guess = (0.0, 0.0, z0, r0)

    result = least_squares(sphere_residuals, initial_guess, args=(x, y, z))
    x0_fit, y0_fit, z0_fit, r_fit = result.x
    residuals = sphere_residuals(result.x, x, y, z)
    return x0_fit, y0_fit, z0_fit, r_fit, residuals


def cartesian_to_polar(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert Cartesian coordinates to polar coordinates (rho, phi)."""
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi


def zernike_polar_basis(
    rho: np.ndarray,
    phi: np.ndarray,
    n_modes: int = N_ZERNIKE_TERMS,
) -> np.ndarray:
    """Build the fixed Noll-order Zernike design matrix A."""
    if n_modes < 1 or n_modes > N_ZERNIKE_TERMS:
        raise ValueError(f"n_modes must be in [1, {N_ZERNIKE_TERMS}], got {n_modes}")

    r = np.asarray(rho)
    u = np.asarray(phi)

    r2 = r * r
    r3 = r2 * r
    r4 = r2 * r2
    r5 = r4 * r
    r6 = r3 * r3
    r7 = r6 * r
    r8 = r4 * r4

    cos_u = np.cos(u)
    sin_u = np.sin(u)
    cos_2u = np.cos(2 * u)
    sin_2u = np.sin(2 * u)
    cos_3u = np.cos(3 * u)
    sin_3u = np.sin(3 * u)
    cos_4u = np.cos(4 * u)
    sin_4u = np.sin(4 * u)
    cos_5u = np.cos(5 * u)
    sin_5u = np.sin(5 * u)
    cos_6u = np.cos(6 * u)
    sin_6u = np.sin(6 * u)
    cos_7u = np.cos(7 * u)
    sin_7u = np.sin(7 * u)
    cos_8u = np.cos(8 * u)
    sin_8u = np.sin(8 * u)

    sqrt3 = np.sqrt(3)
    sqrt5 = np.sqrt(5)
    sqrt6 = np.sqrt(6)
    sqrt7 = np.sqrt(7)
    sqrt8 = np.sqrt(8)
    sqrt10 = np.sqrt(10)
    sqrt12 = np.sqrt(12)
    sqrt14 = np.sqrt(14)
    sqrt18 = np.sqrt(18)

    basis = np.column_stack(
        [
            np.ones_like(r),
            2 * r * cos_u,
            2 * r * sin_u,
            sqrt3 * (2 * r2 - 1),
            sqrt6 * r2 * sin_2u,
            sqrt6 * r2 * cos_2u,
            sqrt8 * (3 * r3 - 2 * r) * sin_u,
            sqrt8 * (3 * r3 - 2 * r) * cos_u,
            sqrt8 * r3 * sin_3u,
            sqrt8 * r3 * cos_3u,
            sqrt5 * (6 * r4 - 6 * r2 + 1),
            sqrt10 * (4 * r4 - 3 * r2) * cos_2u,
            sqrt10 * (4 * r4 - 3 * r2) * sin_2u,
            sqrt10 * r4 * cos_4u,
            sqrt10 * r4 * sin_4u,
            sqrt12 * (10 * r5 - 12 * r3 + 3 * r) * cos_u,
            sqrt12 * (10 * r5 - 12 * r3 + 3 * r) * sin_u,
            sqrt12 * (5 * r5 - 4 * r3) * cos_3u,
            sqrt12 * (5 * r5 - 4 * r3) * sin_3u,
            sqrt12 * r5 * cos_5u,
            sqrt12 * r5 * sin_5u,
            sqrt7 * (20 * r6 - 30 * r4 + 12 * r2 - 1),
            sqrt14 * (15 * r6 - 20 * r4 + 6 * r2) * sin_2u,
            sqrt14 * (15 * r6 - 20 * r4 + 6 * r2) * cos_2u,
            sqrt14 * (6 * r6 - 5 * r4) * sin_4u,
            sqrt14 * (6 * r6 - 5 * r4) * cos_4u,
            sqrt14 * r6 * sin_6u,
            sqrt14 * r6 * cos_6u,
            4 * (35 * r7 - 60 * r5 + 30 * r3 - 4 * r) * sin_u,
            4 * (35 * r7 - 60 * r5 + 30 * r3 - 4 * r) * cos_u,
            4 * (21 * r7 - 30 * r5 + 10 * r3) * sin_3u,
            4 * (21 * r7 - 30 * r5 + 10 * r3) * cos_3u,
            4 * (7 * r7 - 6 * r5) * sin_5u,
            4 * (7 * r7 - 6 * r5) * cos_5u,
            4 * r7 * sin_7u,
            4 * r7 * cos_7u,
            3 * (70 * r8 - 140 * r6 + 90 * r4 - 20 * r2 + 1),
            sqrt18 * (56 * r8 - 105 * r6 + 60 * r4 - 10 * r2) * cos_2u,
            sqrt18 * (56 * r8 - 105 * r6 + 60 * r4 - 10 * r2) * sin_2u,
            sqrt18 * (28 * r8 - 42 * r6 + 15 * r4) * cos_4u,
            sqrt18 * (28 * r8 - 42 * r6 + 15 * r4) * sin_4u,
            sqrt18 * (8 * r8 - 7 * r6) * cos_6u,
            sqrt18 * (8 * r8 - 7 * r6) * sin_6u,
            sqrt18 * r8 * cos_8u,
            sqrt18 * r8 * sin_8u,
        ]
    )
    return basis[:, :n_modes]


def fit_zernike_lstsq(
    rho: np.ndarray,
    phi: np.ndarray,
    wavefront: np.ndarray,
    n_modes: int = N_ZERNIKE_TERMS,
    rcond: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Fit Zernike coefficients with linear least squares."""
    A = zernike_polar_basis(rho, phi, n_modes=n_modes)
    coeffs, *_ = np.linalg.lstsq(A, wavefront, rcond=rcond)
    fit = A @ coeffs
    residuals = wavefront - fit
    rms = float(np.sqrt(np.mean(residuals**2)))
    cond_number = float(np.linalg.cond(A))
    return coeffs, fit, residuals, rms, cond_number


def _pad_coeffs_to_45(coeffs: np.ndarray) -> np.ndarray:
    """Pad coefficient vector to fixed 45 terms for plotting/export compatibility."""
    if len(coeffs) == N_ZERNIKE_TERMS:
        return coeffs
    if len(coeffs) > N_ZERNIKE_TERMS:
        return coeffs[:N_ZERNIKE_TERMS]
    padded = np.zeros(N_ZERNIKE_TERMS, dtype=float)
    padded[: len(coeffs)] = coeffs
    return padded


def zernike_polar(
    loc: np.ndarray,
    z1: float,
    z2: float,
    z3: float,
    z4: float,
    z5: float,
    z6: float,
    z7: float,
    z8: float,
    z9: float,
    z10: float,
    z11: float,
    z12: float,
    z13: float,
    z14: float,
    z15: float,
    z16: float,
    z17: float,
    z18: float,
    z19: float,
    z20: float,
    z21: float,
    z22: float,
    z23: float,
    z24: float,
    z25: float,
    z26: float,
    z27: float,
    z28: float,
    z29: float,
    z30: float,
    z31: float,
    z32: float,
    z33: float,
    z34: float,
    z35: float,
    z36: float,
    z37: float,
    z38: float,
    z39: float,
    z40: float,
    z41: float,
    z42: float,
    z43: float,
    z44: float,
    z45: float,
) -> np.ndarray:
    """Compute the combined wavefront using Zernike coefficients."""
    coeffs = np.asarray(
        [
            z1,
            z2,
            z3,
            z4,
            z5,
            z6,
            z7,
            z8,
            z9,
            z10,
            z11,
            z12,
            z13,
            z14,
            z15,
            z16,
            z17,
            z18,
            z19,
            z20,
            z21,
            z22,
            z23,
            z24,
            z25,
            z26,
            z27,
            z28,
            z29,
            z30,
            z31,
            z32,
            z33,
            z34,
            z35,
            z36,
            z37,
            z38,
            z39,
            z40,
            z41,
            z42,
            z43,
            z44,
            z45,
        ],
        dtype=float,
    )
    basis = zernike_polar_basis(loc[:, 0], loc[:, 1], n_modes=N_ZERNIKE_TERMS)
    return basis @ coeffs


def print_Zernike_coeffs(
    design_id: str,
    surf_id: str,
    zpoly_fits: np.ndarray,
    zv: float,
    zpoly_surf_sse: float,
    short_list: bool = False,
) -> None:
    """Display a table with calculated Zernike coefficients."""
    zernike_names = {
        1: "Piston",
        2: "X-tilt",
        3: "Y-tilt",
        4: "Defocus",
        5: "Oblique Astigmatism",
        6: "Vertical Astigmatism",
        7: "Vertical Coma",
        8: "Horizontal Coma",
        9: "Vertical Trefoil",
        10: "Oblique Trefoil",
        11: "Primary Spherical",
        12: "Vertical Secondary Astigmatism",
        13: "Oblique Secondary Astigmatism",
        14: "Vertical Quadrafoil",
        15: "Oblique Quadrafoil",
    }

    print(f"Design ID = {design_id}")
    print(f"{surf_id} surface Zernike polynomials fit results (Noll numbering)")
    print("-" * 59)

    for i, coeff in enumerate(zpoly_fits, start=1):
        if np.abs(coeff) > 1e-4 or not short_list:
            name = zernike_names.get(i, "")
            label = f"Z{i}:"
            print(f"{label} {coeff:10.4f}   {name}".rstrip())

    print("-" * 59)
    print(f"{zpoly_fits[0] - zv:10.4f} = Z1 for vertex reference")
    print("-" * 59)
    print(f"{zpoly_surf_sse:10.2e} = sum squared error")


def _build_initial_fit_guess(zv: float, r_fit: float) -> np.ndarray:
    return np.array(
        [
            zv,
            0,
            0,
            -r_fit,
            0,
            0,
            0,
            0,
            0,
            0,
            (10000 / r_fit),
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
        dtype=float,
    )


def _build_residual_fit_guess(zv2: float) -> np.ndarray:
    guess = np.zeros(N_ZERNIKE_TERMS, dtype=float)
    guess[0] = zv2
    return guess


def fit_surface_with_zernike(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    *,
    surf_id: str,
    flip_posterior_sign: bool = True,
    method: str = "lstsq",
    n_modes: int = N_ZERNIKE_TERMS,
    rcond: float | None = None,
) -> SurfaceFitResults:
    """Run the complete sphere + Zernike fitting pipeline for one surface."""
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    if len(x) != len(y) or len(y) != len(z):
        raise ValueError("x, y, z must have equal length.")
    if len(x) == 0:
        raise ValueError("No data points provided.")

    x0_fit, y0_fit, z0_fit, r_fit, best_sphere_resid = fit_sphere(x, y, z)

    if flip_posterior_sign and uses_posterior_sign_convention(surf_id):
        best_sphere_resid = -best_sphere_resid

    sphere_sse_um2 = float(np.sum(np.square(best_sphere_resid)))
    sphere_mae_um = float(np.mean(np.abs(best_sphere_resid)))
    sphere_rms_um = float(np.sqrt(np.mean(np.square(best_sphere_resid))))
    rho, phi = cartesian_to_polar(x - x0_fit, y - y0_fit)
    rho_max = np.max(rho)
    if rho_max <= 0:
        raise ValueError("Invalid polar radius max; all points may be coincident.")

    rho_norm = rho / rho_max
    pol_loci = np.stack((rho_norm, phi), axis=1)

    idx_center = int(np.argmin(rho))
    zv = float(z[idx_center])
    method_key = method.strip().lower()
    if method_key != "lstsq":
        raise ValueError(
            f"Unsupported Zernike fitting method in maintained code: {method!r}. Only 'lstsq' is supported."
        )
    zv2 = float(best_sphere_resid[idx_center])
    (
        zpbs_residual_coefficients_lin,
        _zpbs_residual_surface_direct,
        _zpbs_residual_residuals_direct,
        _zpbs_residual_rms_direct,
        zpbs_residual_cond,
    ) = fit_zernike_lstsq(
        rho_norm,
        phi,
        best_sphere_resid,
        n_modes=n_modes,
        rcond=rcond,
    )
    zpbs_residual_coefficients_um = _pad_coeffs_to_45(zpbs_residual_coefficients_lin)
    zpbs_residual_surface_um = zernike_polar(pol_loci, *zpbs_residual_coefficients_um)
    zpbs_residual_residuals_um = best_sphere_resid - zpbs_residual_surface_um
    zpbs_residual_sse_um2 = float(np.sum(zpbs_residual_residuals_um**2))
    zpbs_residual_mae_um = float(np.mean(np.abs(zpbs_residual_residuals_um)))
    zpbs_residual_rms_um = float(np.sqrt(np.mean(zpbs_residual_residuals_um**2)))

    branch_sign = 1.0 if zv >= float(z0_fit) else -1.0
    sphere_surface_um = _sphere_surface_from_reference(
        rho,
        z0_fit=float(z0_fit),
        radius_um=float(r_fit),
        branch_sign=branch_sign,
    )
    residual_sign = -1.0 if uses_posterior_sign_convention(surf_id) else 1.0
    zpbs_to_data_surface_um = sphere_surface_um + (residual_sign * zpbs_residual_surface_um)
    zpbs_to_data_residuals_um = z - zpbs_to_data_surface_um
    zpbs_to_data_sse_um2 = float(np.sum(zpbs_to_data_residuals_um**2))
    zpbs_to_data_mae_um = float(np.mean(np.abs(zpbs_to_data_residuals_um)))
    zpbs_to_data_rms_um = float(np.sqrt(np.mean(zpbs_to_data_residuals_um**2)))

    return SurfaceFitResults(
        x0_fit=float(x0_fit),
        y0_fit=float(y0_fit),
        z0_fit=float(z0_fit),
        r_fit=float(r_fit),
        best_sphere_resid=best_sphere_resid,
        sphere_sse_um2=sphere_sse_um2,
        sphere_mae_um=sphere_mae_um,
        sphere_rms_um=sphere_rms_um,
        rho=rho,
        phi=phi,
        rho_norm=rho_norm,
        pol_loci=pol_loci,
        zv=zv,
        zv2=zv2,
        zpbs_residual_coefficients_um=zpbs_residual_coefficients_um,
        zpbs_residual_surface_um=zpbs_residual_surface_um,
        zpbs_residual_residuals_um=zpbs_residual_residuals_um,
        zpbs_residual_sse_um2=zpbs_residual_sse_um2,
        zpbs_residual_mae_um=zpbs_residual_mae_um,
        zpbs_residual_rms_um=zpbs_residual_rms_um,
        zpbs_residual_cond=zpbs_residual_cond,
        zpbs_to_data_surface_um=zpbs_to_data_surface_um,
        zpbs_to_data_residuals_um=zpbs_to_data_residuals_um,
        zpbs_to_data_sse_um2=zpbs_to_data_sse_um2,
        zpbs_to_data_mae_um=zpbs_to_data_mae_um,
        zpbs_to_data_rms_um=zpbs_to_data_rms_um,
        zernike_method=method_key,
    )


def visualize_zernike_slice(
    pol_loci: np.ndarray,
    z_residuals: np.ndarray,
    rho_max: float,
    zernike_coeffs: np.ndarray,
    slice_angle: float,
    angular_tolerance: float = np.pi / 12,
    n_points: int = 200,
) -> Any:
    """Visualize a 2D slice through a Zernike surface with overlaid data points."""
    if go is None:
        raise ModuleNotFoundError("plotly is required to visualize Zernike slices.")
    rho_norm_slice = np.linspace(0, 1, n_points)
    phi_slice = np.full(n_points, slice_angle)
    slice_loci = np.stack((rho_norm_slice, phi_slice), axis=1)
    z_fit_slice = zernike_polar(slice_loci, *zernike_coeffs)
    r_signed_slice = rho_norm_slice * rho_max

    phi_slice_opposite = np.full(n_points, slice_angle + np.pi)
    slice_loci_opposite = np.stack((rho_norm_slice, phi_slice_opposite), axis=1)
    z_fit_slice_opposite = zernike_polar(slice_loci_opposite, *zernike_coeffs)
    r_signed_slice_opposite = -rho_norm_slice * rho_max

    r_signed_full = np.concatenate([r_signed_slice_opposite[::-1], r_signed_slice])
    z_fit_full = np.concatenate([z_fit_slice_opposite[::-1], z_fit_slice])

    rho_norm_data = pol_loci[:, 0]
    phi_data = pol_loci[:, 1]
    phi_diff = np.abs(np.angle(np.exp(1j * (phi_data - slice_angle))))
    phi_diff_opposite = np.abs(np.angle(np.exp(1j * (phi_data - slice_angle - np.pi))))
    mask_positive = phi_diff <= angular_tolerance
    mask_negative = phi_diff_opposite <= angular_tolerance
    mask = mask_positive | mask_negative

    r_signed_data = np.zeros(len(phi_data))
    r_signed_data[mask_positive] = rho_norm_data[mask_positive] * rho_max
    r_signed_data[mask_negative] = -rho_norm_data[mask_negative] * rho_max

    r_signed_filtered = r_signed_data[mask]
    z_filtered = z_residuals[mask]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=r_signed_full,
            y=z_fit_full,
            mode="lines",
            name="Zernike Fit",
            line={"color": "blue", "width": 1.5},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=r_signed_filtered,
            y=z_filtered,
            mode="markers",
            name="Data Points",
            marker={"color": "red", "size": 8, "opacity": 0.6},
        )
    )

    fig.update_layout(
        width=960,
        height=720,
        xaxis_title="Radial Distance (um)",
        yaxis_title="Height Residual (um)",
        title=(f"Zernike Surface Slice at φ = {slice_angle:.3f} rad (±{np.degrees(angular_tolerance):.1f}°)"),
        showlegend=True,
        hovermode="closest",
        template="plotly_white",
    )
    return fig


def export_zernike_coefficients_csv(
    output_file: str | Path,
    *,
    design_id: str,
    fea_id: str,
    surf_id: str,
    tension_mn: float | str,
    base_sphere_roc_um: float,
    vertex_um: float,
    vertex_residual_um: float,
    norm_radius_um: float,
    zernike_coefficients_mm: np.ndarray,
) -> Path:
    """Write Zernike coefficients to CSV in the same format as the original notebook."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile, dialect="excel")
        for row in build_zernike_coefficients_rows(
            design_id=design_id,
            fea_id=fea_id,
            surf_id=surf_id,
            tension_mn=tension_mn,
            base_sphere_roc_um=base_sphere_roc_um,
            vertex_um=vertex_um,
            vertex_residual_um=vertex_residual_um,
            norm_radius_um=norm_radius_um,
            zernike_coefficients_mm=zernike_coefficients_mm,
        ):
            writer.writerow(row)

    return output_file


def _format_coefficient_csv_value(label: str, value: object) -> object:
    """Format selected coefficient-report floats as stable CSV strings."""
    if label in {"Vertex (mm)", "Vertex residual (mm)"}:
        return f"{float(value):.6f}"
    if label.startswith("Z"):
        return np.format_float_positional(
            float(value),
            precision=15,
            unique=True,
            fractional=False,
            trim="-",
        )
    return value


def build_zernike_coefficients_rows(
    *,
    design_id: str,
    fea_id: str,
    surf_id: str,
    tension_mn: float | str,
    base_sphere_roc_um: float,
    vertex_um: float,
    vertex_residual_um: float,
    norm_radius_um: float,
    zernike_coefficients_mm: np.ndarray,
) -> list[tuple[str, object]]:
    """Build the notebook-style coefficient report rows shared by per-surface and batch exports."""
    rows: list[tuple[str, object]] = [
        ("Design", f"R01V{design_id}"),
        ("FEA", fea_id),
        ("Surface", surf_id),
        ("Tension (mN)", str(tension_mn)),
        ("Base sphere ROC (mm)", base_sphere_roc_um / 1000.0),
        ("Vertex (mm)", vertex_um / 1000.0),
        ("Vertex residual (mm)", vertex_residual_um / 1000.0),
        ("No. ZP terms", N_ZERNIKE_TERMS),
        ("Norm. Radius (mm)", norm_radius_um / 1000.0),
    ]
    for i, term in enumerate(zernike_coefficients_mm, start=1):
        rows.append((f"Z{i}", term))
    return [(label, _format_coefficient_csv_value(label, value)) for label, value in rows]
