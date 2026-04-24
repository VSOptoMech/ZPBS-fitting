from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class PathResolution:
    """Resolved viewer path plus the remap strategy that produced it."""

    original: Path
    resolved: Path
    strategy: str
    exists: bool


def _coerce_path(value: object) -> Path | None:
    """Convert non-empty path-like values into `Path` objects."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text)


def _append_unique_candidate(
    candidates: list[tuple[Path, str]],
    seen: set[str],
    path: Path | None,
    strategy: str,
) -> None:
    """Append a candidate path once while preserving insertion order."""
    if path is None:
        return
    key = str(path)
    if key in seen:
        return
    seen.add(key)
    candidates.append((path, strategy))


def _select_path_resolution(candidates: list[tuple[Path, str]]) -> PathResolution:
    """Choose the first existing path and otherwise keep the last fallback."""
    for path, strategy in candidates:
        if path.exists():
            return PathResolution(original=candidates[0][0], resolved=path, strategy=strategy, exists=True)
    fallback_path, fallback_strategy = candidates[-1]
    return PathResolution(
        original=candidates[0][0],
        resolved=fallback_path,
        strategy=fallback_strategy,
        exists=False,
    )


def remap_path_prefix(
    source_path: Path,
    *,
    original_root: Path | None,
    replacement_root: Path | None,
) -> Path | None:
    """Rebase a source path onto a replacement root when possible."""
    if original_root is None or replacement_root is None:
        return None
    try:
        relative = source_path.relative_to(original_root)
    except ValueError:
        return None
    return replacement_root / relative


def infer_original_run_dir(summary_file: Path, manifest: dict[str, object]) -> Path | None:
    """Infer the original run directory recorded alongside a summary workbook."""
    manifest_summary_path = _coerce_path(manifest.get("summary_report_path"))
    if manifest_summary_path is not None:
        return manifest_summary_path.parent
    return None


def infer_original_source_root(
    rows: list[dict[str, str]],
    manifest: dict[str, object],
) -> Path | None:
    """Infer the original XYZ root from the manifest or the workbook rows."""
    manifest_input_dir = _coerce_path(manifest.get("input_dir"))
    if manifest_input_dir is not None:
        return manifest_input_dir

    source_paths = [Path(row["source_file"]) for row in rows if row.get("source_file", "").strip()]
    absolute_paths = [path for path in source_paths if path.is_absolute()]
    if not absolute_paths:
        return None

    try:
        common_path = Path(os.path.commonpath([str(path) for path in absolute_paths]))
    except ValueError:
        return None
    return common_path


def resolve_summary_coefficients_path(
    coeff_path_text: str,
    *,
    summary_file: Path,
    manifest: dict[str, object],
) -> PathResolution:
    """Resolve a coefficients CSV after the run directory has moved."""
    original = Path(coeff_path_text)
    candidates: list[tuple[Path, str]] = []
    seen: set[str] = set()
    current_run_dir = summary_file.parent

    _append_unique_candidate(candidates, seen, original, "workbook")
    if not original.is_absolute():
        _append_unique_candidate(candidates, seen, current_run_dir / original, "summary-relative")

    remapped_run_path = remap_path_prefix(
        original,
        original_root=infer_original_run_dir(summary_file, manifest),
        replacement_root=current_run_dir,
    )
    _append_unique_candidate(candidates, seen, remapped_run_path, "run-folder-remap")
    _append_unique_candidate(
        candidates,
        seen,
        current_run_dir / "coefficients" / original.name,
        "coefficients-filename-fallback",
    )
    return _select_path_resolution(candidates)


def resolve_summary_source_path(
    source_path_text: str,
    *,
    summary_file: Path,
    rows: list[dict[str, str]],
    manifest: dict[str, object],
    original_source_root: Path | None,
    local_source_root: Path | None,
) -> PathResolution:
    """Resolve a source XYZ path after the workbook is opened on another machine."""
    original = Path(source_path_text)
    candidates: list[tuple[Path, str]] = []
    seen: set[str] = set()

    _append_unique_candidate(candidates, seen, original, "workbook")
    if not original.is_absolute():
        _append_unique_candidate(candidates, seen, summary_file.parent / original, "summary-relative")

    effective_original_root = original_source_root or infer_original_source_root(rows, manifest)
    remapped_source_path = remap_path_prefix(
        original,
        original_root=effective_original_root,
        replacement_root=local_source_root,
    )
    _append_unique_candidate(candidates, seen, remapped_source_path, "source-root-remap")
    _append_unique_candidate(
        candidates,
        seen,
        local_source_root / original.name if local_source_root is not None else None,
        "source-filename-fallback",
    )
    return _select_path_resolution(candidates)


def prepare_summary_row_for_preview(
    row: dict[str, str],
    *,
    summary_file: Path,
    rows: list[dict[str, str]],
    manifest: dict[str, object],
    original_source_root_text: str = "",
    local_source_root_text: str = "",
) -> tuple[dict[str, str], dict[str, str]]:
    """Attach resolved source and coefficients paths to one preview row."""
    original_source_root = _coerce_path(original_source_root_text)
    local_source_root = _coerce_path(local_source_root_text)

    source_resolution = resolve_summary_source_path(
        row.get("source_file", ""),
        summary_file=summary_file,
        rows=rows,
        manifest=manifest,
        original_source_root=original_source_root,
        local_source_root=local_source_root,
    )
    coeff_resolution = resolve_summary_coefficients_path(
        row.get("output_coefficients_csv", ""),
        summary_file=summary_file,
        manifest=manifest,
    )

    preview_row = dict(row)
    preview_row["_resolved_source_file"] = str(source_resolution.resolved)
    preview_row["_resolved_output_coefficients_csv"] = str(coeff_resolution.resolved)

    details = {
        "original_source_file": str(source_resolution.original),
        "resolved_source_file": str(source_resolution.resolved),
        "source_resolution_strategy": source_resolution.strategy,
        "source_exists": "1" if source_resolution.exists else "0",
        "original_coeff_file": str(coeff_resolution.original),
        "resolved_coeff_file": str(coeff_resolution.resolved),
        "coeff_resolution_strategy": coeff_resolution.strategy,
        "coeff_exists": "1" if coeff_resolution.exists else "0",
        "effective_original_source_root": str(original_source_root or infer_original_source_root(rows, manifest) or ""),
        "local_source_root": str(local_source_root or ""),
    }
    return preview_row, details
