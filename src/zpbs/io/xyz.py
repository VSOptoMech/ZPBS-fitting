"""Maintained XYZ IO and metadata parsing helpers."""

from __future__ import annotations

import csv
import re
from dataclasses import replace
from pathlib import Path

from zpbs.common import force_sort_key
from zpbs.models import ProcessingInput, SurfaceMetadata

NOTEBOOK_FILENAME_RE = re.compile(
    r"^(?P<design_token>R\d+V(?P<design_id>[^-_]+))-(?P<fea_id>[^_]+)_(?P<force_id>F[^_]+)_FVS_(?P<surface_token>[A-Za-z]+)$"
)


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
    """Return True when text parses cleanly as a float."""
    try:
        float(text)
    except ValueError:
        return False
    return True


def _split_xyz_line(line: str, delimiter: str | None) -> list[str]:
    """Split one XYZ line using CSV-style delimiters or whitespace."""
    stripped = line.strip()
    if delimiter is None:
        return stripped.split()
    return [field.strip() for field in stripped.split(delimiter)]


def _detect_delimiter(lines: list[str]) -> str | None:
    """Detect the delimiter used by a sample of XYZ lines."""
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


def collapse_identical_initial_inputs(files: list[Path]) -> list[ProcessingInput]:
    """Collapse identical *I files to one synthetic zero-force entry per design and surface state."""
    initial_groups: dict[tuple[str, str, str, str], list[tuple[Path, SurfaceMetadata]]] = {}
    processing_inputs: list[ProcessingInput] = []

    for file_path in files:
        metadata = parse_surface_metadata(file_path)
        if metadata.surface_token.endswith("I"):
            group_key = (metadata.design_token, metadata.fea_id, metadata.surf_id, metadata.surface_token)
            initial_groups.setdefault(group_key, []).append((file_path, metadata))
            continue
        processing_inputs.append(
            ProcessingInput(
                source_file=file_path,
                source_metadata=metadata,
                metadata=metadata,
            )
        )

    for group_key, members in sorted(initial_groups.items()):
        _design_token, _fea_id, surf_id, _surface_token = group_key
        surface_tokens = {metadata.surface_token for _, metadata in members}
        if len(surface_tokens) != 1:
            raise ValueError(
                f"Cannot collapse {surf_id} initial states because multiple surface tokens were found: "
                f"{sorted(surface_tokens)}"
            )
        payloads = {file_path.read_bytes() for file_path, _ in members}
        if len(payloads) != 1:
            raise ValueError(f"Cannot collapse {surf_id} initial states because the raw *I payloads are not identical.")

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
