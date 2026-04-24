"""Writers and batch report helpers."""

from __future__ import annotations

import csv
import html
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from zpbs.azp_csv_pipeline import build_zernike_coefficients_rows
from zpbs.common import (
    FOCUS_SURF_IDS,
    format_mae_rms_display,
    format_processed_label,
    format_tension,
    is_focus_surface_family,
    round_sigfigs_array,
    sanitize_h5_name,
    uses_posterior_sign_convention,
)
from zpbs.models import FitArtifacts


def artifacts_to_summary_row(artifacts: FitArtifacts) -> dict[str, Any]:
    """Flatten fit artifacts for batch summary CSV output."""
    return {
        "source_file": str(artifacts.source_file),
        "output_coefficients_csv": str(artifacts.output_coefficients_csv),
        "force_id": artifacts.metadata.force_id,
        "surf_id": artifacts.metadata.surf_id,
        "surface_token": artifacts.metadata.surface_token,
        "applied_reference_radius_um": artifacts.applied_reference_radius_um,
        "applied_normalization_radius_um": artifacts.norm_radius_um,
        "observed_aperture_radius_um": artifacts.observed_aperture_radius_um,
        "sphere_rms_um": artifacts.sphere_rms_um,
        "sphere_mae_um": artifacts.sphere_mae_um,
        "zpbs_residual_rms_um": artifacts.zpbs_residual_rms_um,
        "zpbs_residual_cond": artifacts.zpbs_residual_cond,
        "vertex_mismatch_z_um": artifacts.vertex_mismatch_z_um,
        "vertex_um": artifacts.vertex_um,
        "vertex_residual_um": artifacts.vertex_residual_um,
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
    columns: list[tuple[str, list[tuple[str, object]]]] = []

    for item in artifacts:
        export_coefficients_um = item.zpbs_residual_coefficients_um.copy()
        export_coefficients_um[0] = export_coefficients_um[0] - item.sphere_vertex_residual_um
        if item.zernike_coeff_sigfigs is not None:
            export_coefficients_um = round_sigfigs_array(export_coefficients_um, item.zernike_coeff_sigfigs, np=np)
        # Keep the historical anterior sign and invert posterior surfaces once more so
        # the batch-wide report matches the per-file Zemax-facing CSV export.
        export_sign = 1.0 if uses_posterior_sign_convention(item.metadata.surf_id) else -1.0
        export_coefficients_mm = export_sign * export_coefficients_um / 1000.0
        signed_roc_um = (1.0 if float(item.reference_vertex_z_um) >= float(item.z0_fit) else -1.0) * float(
            item.applied_reference_radius_um
        )
        rows = build_zernike_coefficients_rows(
            design_id=item.metadata.design_id,
            fea_id=item.metadata.fea_id,
            surf_id=item.metadata.surf_id,
            tension_mn=format_tension(item.metadata.force_id),
            base_sphere_roc_um=signed_roc_um,
            vertex_um=item.vertex_um,
            vertex_residual_um=item.vertex_residual_um,
            norm_radius_um=item.norm_radius_um,
            zernike_coefficients_mm=export_coefficients_mm,
        )
        columns.append((format_processed_label(item.metadata), rows))

    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", newline="") as handle:
        writer = csv.writer(handle, dialect="excel")
        if not columns:
            writer.writerow(["message", "no rows"])
            return file_path

        row_count = len(columns[0][1])
        for row_index in range(row_count):
            output_row: list[object] = []
            for _label, rows in columns:
                row_name, row_value = rows[row_index]
                output_row.extend([row_name, row_value])
            writer.writerow(output_row)
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
                cells.append(f'<c r="{cell_ref}" t="inlineStr"><is><t xml:space="preserve">{escape(text)}</t></is></c>')
        xml_rows.append(f'<row r="{row_index}">{"".join(cells)}</row>')

    worksheet_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>
    {"".join(xml_rows)}
  </sheetData>
</worksheet>"""
    workbook_xml = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="{sheet_name}" sheetId="1" r:id="rId1"/>
  </sheets>
</workbook>'''
    workbook_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>"""
    package_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>"""
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
</Types>"""

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


def write_batch_h5(h5_path: Path, *, run_name: str, config: dict[str, Any], artifacts: list[FitArtifacts]) -> None:
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
            file_group.attrs["sphere_fit_mode"] = item.sphere_fit_mode
            file_group.attrs["center_weight"] = item.center_weight
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
            file_group.attrs["target_vertex_x_um"] = item.target_vertex_x_um
            file_group.attrs["target_vertex_y_um"] = item.target_vertex_y_um
            file_group.attrs["target_vertex_z_um"] = item.target_vertex_z_um
            file_group.attrs["reference_vertex_x_um"] = item.reference_vertex_x_um
            file_group.attrs["reference_vertex_y_um"] = item.reference_vertex_y_um
            file_group.attrs["reference_vertex_z_um"] = item.reference_vertex_z_um
            file_group.attrs["vertex_mismatch_z_um"] = item.vertex_mismatch_z_um
            file_group.attrs["vertex_um"] = item.vertex_um
            file_group.attrs["sphere_vertex_residual_um"] = item.sphere_vertex_residual_um
            file_group.attrs["vertex_residual_um"] = item.vertex_residual_um
            file_group.attrs["sphere_sse_um2"] = item.sphere_sse_um2
            file_group.attrs["sphere_mae_um"] = item.sphere_mae_um
            file_group.attrs["sphere_rms_um"] = item.sphere_rms_um
            file_group.attrs["zpbs_residual_sse_um2"] = item.zpbs_residual_sse_um2
            file_group.attrs["zpbs_residual_mae_um"] = item.zpbs_residual_mae_um
            file_group.attrs["zpbs_residual_rms_um"] = item.zpbs_residual_rms_um
            file_group.attrs["zpbs_residual_cond"] = item.zpbs_residual_cond

            file_group.create_dataset("x_um", data=item.x)
            file_group.create_dataset("y_um", data=item.y)
            file_group.create_dataset("z_um", data=item.z)
            file_group.create_dataset("rho_um", data=item.rho)
            file_group.create_dataset("phi_rad", data=item.phi)
            file_group.create_dataset("rho_norm", data=item.rho_norm)
            file_group.create_dataset("sphere_residual_um", data=item.sphere_residuals_um)
            file_group.create_dataset("zpbs_to_data_surface_um", data=item.zpbs_to_data_surface_um)
            file_group.create_dataset("zpbs_to_data_residual_um", data=item.zpbs_to_data_residuals_um)
            file_group.create_dataset("zpbs_residual_surface_um", data=item.zpbs_residual_surface_um)
            file_group.create_dataset("zpbs_residual_residual_um", data=item.zpbs_residual_residuals_um)
            file_group.create_dataset("zpbs_residual_coefficients_um", data=item.zpbs_residual_coefficients_um)


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
        bucket = (rho_arr >= start) & (rho_arr <= end) if end == edges[-1] else (rho_arr >= start) & (rho_arr < end)
        if not np.any(bucket):
            continue
        centers.append(float(np.median(rho_arr[bucket])))
        medians.append(float(np.median(values_arr[bucket])))

    return np.asarray(centers, dtype=float), np.asarray(medians, dtype=float)


def _thumbnail_x_limit_um(rho: Any, *, np: Any) -> float:
    """Snap the compact QA x-axis to 100 um increments from zero."""
    rho_arr = np.asarray(rho, dtype=float)
    finite = rho_arr[np.isfinite(rho_arr)]
    if finite.size == 0:
        return 100.0

    span_um = float(np.max(finite))
    if span_um <= 0.0:
        return 100.0
    return float(np.ceil(span_um / 100.0) * 100.0)


def _build_overview_plot_series(artifacts: FitArtifacts, *, bins: int, np: Any) -> dict[str, Any]:
    """Compute the shared profile/residual series used by QA-style overview plots."""
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
    rho_zpbs_to_data, z_zpbs_to_data = radial_bin_profile(
        artifacts.rho, artifacts.zpbs_to_data_surface_um, bins=bins, np=np
    )
    rho_sphere_resid, z_sphere_resid = radial_bin_profile(
        artifacts.rho, artifacts.sphere_residuals_um, bins=bins, np=np
    )
    rho_resid, z_resid = radial_bin_profile(artifacts.rho, artifacts.zpbs_to_data_residuals_um, bins=bins, np=np)
    x_max = float(max(np.max(artifacts.rho), artifacts.observed_aperture_radius_um, 0.0))
    return {
        "rho_meas": rho_meas,
        "z_meas": z_meas,
        "rho_sphere": rho_sphere,
        "z_sphere": z_sphere,
        "rho_zpbs_to_data": rho_zpbs_to_data,
        "z_zpbs_to_data": z_zpbs_to_data,
        "rho_sphere_resid": rho_sphere_resid,
        "z_sphere_resid": z_sphere_resid,
        "rho_resid": rho_resid,
        "z_resid": z_resid,
        "x_max": x_max,
    }


def render_overview_plot(
    figure: Any,
    artifacts: FitArtifacts,
    *,
    bins: int = 96,
    compact: bool = False,
) -> None:
    """Populate a figure with a QA-style overview plot for one fitted file."""
    import numpy as np

    series = _build_overview_plot_series(artifacts, bins=bins, np=np)
    figure.clear()
    if compact:
        axes = figure.subplots(
            2,
            1,
            sharex=True,
            gridspec_kw={"height_ratios": [3.0, 1.2]},
        )
        ax_profile, ax_resid = axes
        ax_profile.plot(series["rho_meas"], series["z_meas"], color="#1f2937", linewidth=1.1, label="measured")
        ax_profile.plot(series["rho_sphere"], series["z_sphere"], color="#2563eb", linewidth=0.9, label="sphere")
        ax_profile.plot(
            series["rho_zpbs_to_data"],
            series["z_zpbs_to_data"],
            color="#dc2626",
            linewidth=0.9,
            linestyle="--",
            label="zpbs fit",
        )
        ax_profile.set_ylabel("z (um)", fontsize=7)
        ax_profile.grid(True, alpha=0.2, linewidth=0.4)
        ax_profile.tick_params(labelsize=6, length=2)
        ax_profile.legend(loc="best", fontsize=6, frameon=False, ncol=3, handlelength=1.5, columnspacing=0.8)
        ax_profile.set_title(
            (
                f"{artifacts.metadata.surface_token}  sphere_rms="
                f"{format_mae_rms_display(artifacts.sphere_rms_um, precision=3)}  "
                f"zpbs_residual_cond={artifacts.zpbs_residual_cond:.2e}"
            ),
            fontsize=7,
        )

        ax_resid.plot(series["rho_resid"], series["z_resid"], color="#059669", linewidth=0.9)
        ax_resid.axhline(0.0, color="#6b7280", linewidth=0.6, alpha=0.6)
        ax_resid.set_xlabel("rho (um)", fontsize=7)
        ax_resid.set_ylabel("ZPBS Fit Resid. (um)", fontsize=7)
        ax_resid.grid(True, alpha=0.2, linewidth=0.4)
        ax_resid.tick_params(labelsize=6, length=2)
        compact_x_max = _thumbnail_x_limit_um(artifacts.rho, np=np)
        for axis in (ax_profile, ax_resid):
            axis.set_xlim(0.0, compact_x_max)
            axis.margins(x=0.0)
        return

    grid = figure.add_gridspec(3, 1, height_ratios=[2.3, 1.0, 1.0])
    ax_profile = figure.add_subplot(grid[0, 0])
    ax_sphere_resid = figure.add_subplot(grid[1, 0], sharex=ax_profile)
    ax_resid = figure.add_subplot(grid[2, 0], sharex=ax_profile)

    ax_profile.plot(series["rho_meas"], series["z_meas"], color="#111827", linewidth=1.5, label="Measured")
    ax_profile.plot(series["rho_sphere"], series["z_sphere"], color="#2563eb", linewidth=1.2, label="Sphere")
    ax_profile.plot(
        series["rho_zpbs_to_data"],
        series["z_zpbs_to_data"],
        color="#dc2626",
        linewidth=1.2,
        linestyle="--",
        label="ZPBS Fit",
    )
    ax_profile.set_title(
        f"{artifacts.metadata.surface_token} | {artifacts.source_file.name}",
        fontsize=11,
    )
    ax_profile.set_ylabel("z (um)")
    ax_profile.grid(True, alpha=0.2)
    ax_profile.legend(loc="upper right", frameon=False, fontsize=8)
    metrics_text = "\n".join(
        [
            f"Norm radius: {artifacts.norm_radius_um:.2f} um",
            f"Observed aperture: {artifacts.observed_aperture_radius_um:.2f} um",
            f"Sphere RMS: {format_mae_rms_display(artifacts.sphere_rms_um, precision=3)} um",
            f"ZPBS residual RMS: {format_mae_rms_display(artifacts.zpbs_residual_rms_um, precision=3)} um",
            f"ZPBS residual cond: {artifacts.zpbs_residual_cond:.2e}",
            f"Vertex mismatch z: {artifacts.vertex_mismatch_z_um:.2e} um",
        ]
    )
    metrics_y = 0.02 if artifacts.metadata.surf_id in {"AA", "AP"} else 0.98
    metrics_va = "bottom" if artifacts.metadata.surf_id in {"AA", "AP"} else "top"
    ax_profile.text(
        0.015,
        metrics_y,
        metrics_text,
        transform=ax_profile.transAxes,
        ha="left",
        va=metrics_va,
        fontsize=8,
        family="monospace",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1", "alpha": 0.9},
    )

    ax_sphere_resid.plot(series["rho_sphere_resid"], series["z_sphere_resid"], color="#7c3aed", linewidth=1.2)
    ax_sphere_resid.axhline(0.0, color="#6b7280", linewidth=0.8, alpha=0.8)
    ax_sphere_resid.set_ylabel("Sphere Resid. (um)")
    ax_sphere_resid.grid(True, alpha=0.2)

    ax_resid.plot(series["rho_resid"], series["z_resid"], color="#059669", linewidth=1.2)
    ax_resid.axhline(0.0, color="#6b7280", linewidth=0.8, alpha=0.8)
    ax_resid.set_xlabel("rho (um)")
    ax_resid.set_ylabel("ZPBS Fit Resid. (um)")
    ax_resid.grid(True, alpha=0.2)

    for axis in (ax_profile, ax_sphere_resid, ax_resid):
        axis.set_xlim(0.0, series["x_max"])
        axis.margins(x=0.0)


def write_thumbnail_plot(file_path: Path, artifacts: FitArtifacts, *, bins: int = 64) -> None:
    """Write one compact QA thumbnail with measured/fitted radial profiles and residuals."""
    mpl_config_dir = file_path.parent.parent / ".matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(4.8, 2.2), dpi=110)
    render_overview_plot(fig, artifacts, bins=bins, compact=True)
    fig.tight_layout(pad=0.4)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(file_path, bbox_inches="tight")
    plt.close(fig)


def write_overview_plot(file_path: Path, artifacts: FitArtifacts, *, bins: int = 128) -> Path:
    """Write a high-resolution QA-style overview plot for one fitted file."""
    mpl_config_dir = file_path.parent.parent / ".matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    fig = Figure(figsize=(9.0, 5.6), constrained_layout=True)
    render_overview_plot(fig, artifacts, bins=bins, compact=False)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(file_path, bbox_inches="tight", dpi=150)
    return file_path


def write_qa_report(
    run_dir: Path, artifacts: list[FitArtifacts], *, summary_report_path: Path, analysis_date: str
) -> Path:
    """Generate a browsable HTML gallery of compact fit thumbnails for the batch."""
    qa_dir = run_dir / "qa"
    thumbnails_dir = qa_dir / "thumbnails"
    thumbnails_dir.mkdir(parents=True, exist_ok=True)

    plotted = [item for item in artifacts if is_focus_surface_family(item.metadata.surf_id)]
    skipped = len(artifacts) - len(plotted)

    for item in plotted:
        write_thumbnail_plot(thumbnails_dir / f"{sanitize_h5_name(format_processed_label(item.metadata))}.png", item)

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
                f"<td>{format_mae_rms_display(item.sphere_rms_um, precision=3)}</td>"
                f"<td>{format_mae_rms_display(item.zpbs_residual_rms_um, precision=3)}</td>"
                f"<td>{item.zpbs_residual_cond:.3e}</td>"
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
  <p class="meta">Run: {html.escape(run_dir.name)} | Focus files: {len(plotted)} | Skipped non-focus families: {skipped}</p>
  <p class="meta">Plotted families: {", ".join(sorted(FOCUS_SURF_IDS))}</p>
  <p><a href="../{html.escape(summary_report_path.name)}">{html.escape(summary_report_path.name)}</a> | <a href="../run_manifest.json">run_manifest.json</a></p>

  <h2>All Files</h2>
  <table>
    <thead>
      <tr>
        <th>File</th>
        <th>Surface</th>
        <th>Sphere RMS (um)</th>
        <th>ZPBS Residual RMS (um)</th>
        <th>ZPBS Residual Cond</th>
        <th>CSV</th>
        <th>Preview</th>
      </tr>
    </thead>
    <tbody>
      {build_rows(plotted)}
    </tbody>
  </table>
</body>
</html>
"""
    html_path.write_text(payload)
    return html_path
