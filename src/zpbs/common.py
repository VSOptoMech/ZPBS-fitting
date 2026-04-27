"""Small shared helpers and validation logic."""

from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Any

FOCUS_SURF_IDS = {"AA", "AP", "PA", "PP"}
SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def make_output_filename(metadata: Any) -> str:
    """Mirror the notebook export naming style for coefficient CSVs."""
    return f"ZPs_{metadata.design_token}-{metadata.fea_id}_{metadata.surface_token}_{metadata.force_id}_base_sphere.csv"


def format_processed_label(metadata: Any) -> str:
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


def signed_sphere_radius_um(radius_um: float, *, reference_vertex_z_um: float, z0_fit: float) -> float:
    """Return the signed sphere radius using the maintained export/display branch convention."""
    sign = 1.0 if float(reference_vertex_z_um) >= float(z0_fit) else -1.0
    return sign * abs(float(radius_um))


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


def format_mae_rms_um(value: object, *, decimals_small: int = 6, decimals_large: int = 3) -> str:
    """Format MAE/RMS-style micrometer metrics with fixed 6 decimals below 1.0 um."""
    if value is None:
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        text = str(value)
        if not text.strip():
            return ""
        return text

    magnitude = abs(numeric)
    if magnitude < 1.0:
        return f"{numeric:.{decimals_small}f}"
    if magnitude >= 1000.0:
        return f"{numeric:.{decimals_large}e}"
    return f"{numeric:.{decimals_large}f}"


def format_mae_rms_display(value: object, *, precision: int = 3) -> str:
    """Backward-compatible wrapper for MAE/RMS display formatting."""
    return format_mae_rms_um(value, decimals_small=6, decimals_large=precision)


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


def validate_center_weight(center_weight: float) -> float:
    """Validate the center-weight control used by the center-weighted sphere fit."""
    weight = float(center_weight)
    if not 0.0 <= weight <= 5.0:
        raise ValueError(f"center_weight must be in [0, 5], got {center_weight}.")
    return weight


def validate_sphere_reference_configuration(*, roc_mode: str, sphere_fit_mode: str) -> None:
    """Reject sphere-reference combinations that are intentionally unsupported."""
    mode = sphere_fit_mode.strip().lower()
    roc = roc_mode.strip().lower()
    if mode != "legacy_lsq" and roc != "fit-per-file":
        raise ValueError(
            "Only sphere_fit_mode=legacy_lsq supports fixed ROC or roc_mode average-best-fit; "
            f"got roc_mode={roc_mode!r} with sphere_fit_mode={sphere_fit_mode!r}."
        )


def validate_zernike_method(method: str) -> str:
    """Accept only the maintained least-squares Zernike fitting path."""
    method_key = str(method).strip().lower()
    if method_key != "lstsq":
        raise ValueError(
            f"Unsupported Zernike fitting method in maintained code: {method!r}. Only 'lstsq' is supported."
        )
    return method_key
