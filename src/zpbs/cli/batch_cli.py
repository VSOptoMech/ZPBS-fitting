"""Maintained batch CLI entrypoint."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from zpbs.common import (
    FOCUS_SURF_IDS,
    clamp_normalization_radius_um,
    clamp_reference_radius_um,
    is_focus_surface_family,
    resolve_analysis_date,
    resolve_run_name,
    round_nearest_micrometer,
    validate_center_weight,
    validate_sphere_reference_configuration,
)
from zpbs.io.xyz import collapse_identical_initial_inputs, parse_surface_metadata
from zpbs.pipeline.surface_fit import build_fit_artifacts, get_sphere_prefit_entry
from zpbs.pipeline.tilt_correction import (
    VertexTiltCorrection,
    apply_vertex_tilt_correction_to_artifacts,
    export_coefficient_rows_for_artifacts,
)
from zpbs.reporting.batch_reports import (
    artifacts_to_summary_row,
    resolve_h5_path,
    write_batch_h5,
    write_batch_zp_report,
    write_csv,
    write_json,
    write_qa_report,
    write_xlsx,
)


def _write_name_value_rows(path: Path, rows: list[tuple[str, str]]) -> None:
    """Write two-column name/value rows to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, dialect="excel")
        writer.writerows(rows)


def _tilt_correction_summary_fields(correction: VertexTiltCorrection) -> dict[str, Any]:
    """Return compact per-file summary fields for optional vertex tilt correction."""
    return {
        "vertex_tilt_correction": "on",
        "original_center_slope_x_mrad": correction.original_x_mrad,
        "original_center_slope_y_mrad": correction.original_y_mrad,
        "original_center_slope_magnitude_mrad": correction.original_magnitude_mrad,
        "corrected_center_slope_x_mrad": correction.corrected_x_mrad,
        "corrected_center_slope_y_mrad": correction.corrected_y_mrad,
        "corrected_center_slope_magnitude_mrad": correction.corrected_magnitude_mrad,
        "delta_z2_um": correction.delta_z2_um,
        "delta_z3_um": correction.delta_z3_um,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Create the batch CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Batch-fit notebook-style XYZ point clouds without previews. Only AA/AP/PA/PP files are included."
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing notebook-style .xyz files.")
    parser.add_argument("--glob", default="*.xyz", help="File glob to match inside input-dir. Default: %(default)s")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("batch_outputs"),
        help="Root directory for batch runs. Each run is written into a separate folder.",
    )
    parser.add_argument(
        "--run-name", default=None, help="Optional run folder / HDF5 group name. Defaults to a timestamp."
    )
    parser.add_argument("--recursive", action="store_true", help="Search recursively under input-dir.")
    parser.add_argument(
        "--roc-mode",
        choices=("fit-per-file", "average-best-fit"),
        default="fit-per-file",
        help="How to choose the reference radius of curvature for the batch. Use --fixed-roc-um for a fixed ROC.",
    )
    parser.add_argument(
        "--fixed-roc-um",
        type=float,
        default=None,
        help="Fixed radius of curvature in micrometers. Supplying this selects fixed ROC mode.",
    )
    parser.add_argument(
        "--sphere-fit-mode",
        choices=("legacy_lsq", "center_weighted", "vertex_locked"),
        default="center_weighted",
        help="How to fit the sphere reference before residual formation. Default: %(default)s",
    )
    parser.add_argument(
        "--center-weight",
        type=float,
        default=0.5,
        help=(
            "Center-weight control in [0, 5] for --sphere-fit-mode center_weighted. "
            "Values below 0.05 behave like uniform weighting, 0.5 uses a Gaussian about 2x wider than the "
            "w=1 profile, 1 drives the aperture edge near zero weight, and larger values narrow the Gaussian "
            "proportionally toward the center."
        ),
    )
    parser.add_argument(
        "--h5-path",
        type=Path,
        default=None,
        help="Optional HDF5 file that stores all runs together under separate run groups.",
    )
    parser.add_argument("--n-modes", type=int, default=45, help="Number of Zernike modes to fit. Default: %(default)s")
    parser.add_argument("--rcond", type=float, default=None, help="Least-squares rcond override for lstsq mode.")
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
        "--zero-vertex-tilt",
        action="store_true",
        help=(
            "Adjust residual Z2/Z3 after fitting so the exported Zernike model has zero net center slope. "
            "This is an optional Zemax-facing correction."
        ),
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of files processed.")
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
    if args.fixed_roc_um is not None and args.fixed_roc_um <= 0:
        parser.error("--fixed-roc-um must be positive.")
    if args.fixed_roc_um is not None:
        if args.roc_mode != "fit-per-file":
            parser.error("--fixed-roc-um cannot be combined with --roc-mode average-best-fit.")
        args.roc_mode = "fixed"
    try:
        validate_center_weight(args.center_weight)
        validate_sphere_reference_configuration(roc_mode=args.roc_mode, sphere_fit_mode=args.sphere_fit_mode)
    except ValueError as exc:
        parser.error(str(exc))
    if args.zernike_coeff_sigfigs is not None and args.zernike_coeff_sigfigs < 1:
        parser.error("--zernike-coeff-sigfigs must be positive.")

    matcher = input_dir.rglob if args.recursive else input_dir.glob
    matched_files = sorted(path for path in matcher(args.glob) if path.is_file())
    files = [path for path in matched_files if is_focus_surface_family(parse_surface_metadata(path).surf_id)]
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
    prefit_entries: dict[Path, Any] = {}
    failures: list[dict[str, str]] = []

    def load_prefit(item: Any) -> Any:
        """Cache each file's sphere prefit so multi-stage batch plans reuse it."""
        source_file = item.source_file
        entry = prefit_entries.get(source_file)
        if entry is None:
            entry = get_sphere_prefit_entry(
                source_file,
                surf_id=item.metadata.surf_id,
                sphere_fit_mode=args.sphere_fit_mode,
                center_weight=args.center_weight,
            )
            prefit_entries[source_file] = entry
        return entry

    if args.roc_mode == "average-best-fit":
        surviving_inputs = []
        for item in files_to_process:
            try:
                entry = load_prefit(item)
            except Exception as exc:
                failures.append({"source_file": str(item.source_file), "error": f"prefit radius failed: {exc}"})
                continue
            prefit_radii[item.source_file] = entry.best_radius_um
            surviving_inputs.append(item)
        files_to_process = surviving_inputs
        if not files_to_process:
            write_csv(run_dir / "batch_failures.csv", failures)
            print("No files survived the prefit-radius stage.", file=sys.stderr)
            return 1
        radii_by_surf: dict[str, list[float]] = {}
        raw_apertures_by_surf: dict[str, list[float]] = {}
        raw_apertures_all: list[float] = []
        for path, radius in prefit_radii.items():
            entry = prefit_entries[path]
            sid = parse_surface_metadata(path).surf_id
            radii_by_surf.setdefault(sid, []).append(radius)
            raw_apertures_by_surf.setdefault(sid, []).append(float(entry.raw_aperture_radius_um))
            raw_apertures_all.append(float(entry.raw_aperture_radius_um))
        for sid in sorted(radii_by_surf):
            avg = sum(radii_by_surf[sid]) / len(radii_by_surf[sid])
            if args.round_radii_um:
                avg = clamp_reference_radius_um(
                    round_nearest_micrometer(avg),
                    max(raw_apertures_by_surf[sid]),
                )
            common_radius_by_surf_id[sid] = avg
            print(f"common best ROC {sid}: {avg:.6f} um")
        common_radius_um = sum(prefit_radii.values()) / len(prefit_radii)
        if args.round_radii_um:
            common_radius_um = clamp_reference_radius_um(
                round_nearest_micrometer(common_radius_um),
                max(raw_apertures_all),
            )
        print(f"common best ROC: {common_radius_um:.6f} um")

    if args.normalization_mode == "common-per-surf-id":
        per_surf_id: dict[str, float] = {}
        surviving_inputs = []
        for item in files_to_process:
            try:
                entry = load_prefit(item)
            except Exception as exc:
                failures.append(
                    {"source_file": str(item.source_file), "error": f"prefit normalization radius failed: {exc}"}
                )
                continue
            current = per_surf_id.get(item.metadata.surf_id)
            observed_radius = entry.observed_aperture_radius_um
            per_surf_id[item.metadata.surf_id] = observed_radius if current is None else max(current, observed_radius)
            surviving_inputs.append(item)
        normalization_radius_by_surf_id = (
            {
                surf_id: clamp_normalization_radius_um(round_nearest_micrometer(radius_um), radius_um)
                for surf_id, radius_um in per_surf_id.items()
            }
            if args.round_radii_um
            else per_surf_id
        )
        files_to_process = [
            item for item in surviving_inputs if item.metadata.surf_id in normalization_radius_by_surf_id
        ]
        if not files_to_process:
            write_csv(run_dir / "batch_failures.csv", failures)
            print("No files survived the normalization-radius prefit stage.", file=sys.stderr)
            return 1
        for surf_id in sorted(normalization_radius_by_surf_id):
            print(f"common normalization radius {surf_id}: {normalization_radius_by_surf_id[surf_id]:.6f} um")

    artifacts = []
    summary_rows = []
    for item_input in files_to_process:
        metadata = item_input.metadata
        file_path = item_input.source_file
        if args.roc_mode == "fit-per-file":
            reference_radius_um = None
        elif args.roc_mode == "fixed":
            reference_radius_um = args.fixed_roc_um
        else:
            reference_radius_um = common_radius_by_surf_id.get(metadata.surf_id, common_radius_um)
        normalization_radius_um = (
            normalization_radius_by_surf_id.get(metadata.surf_id)
            if args.normalization_mode == "common-per-surf-id"
            else None
        )

        try:
            prefit_data = load_prefit(item_input)
            item = build_fit_artifacts(
                file_path,
                metadata=metadata,
                source_metadata=item_input.source_metadata,
                output_dir=run_dir,
                n_modes=args.n_modes,
                rcond=args.rcond,
                roc_mode=args.roc_mode,
                reference_radius_um=reference_radius_um,
                sphere_fit_mode=args.sphere_fit_mode,
                center_weight=args.center_weight,
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
            correction = None
            if args.zero_vertex_tilt:
                item, correction = apply_vertex_tilt_correction_to_artifacts(item)
                _write_name_value_rows(item.output_coefficients_csv, export_coefficient_rows_for_artifacts(item))
            artifacts.append(item)
            summary_row = artifacts_to_summary_row(item)
            if correction is not None:
                summary_row.update(_tilt_correction_summary_fields(correction))
            summary_rows.append(summary_row)
            print(f"processed: {file_path}")
        except Exception as exc:
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
            "summary_schema_version": 2,
            "run_name": run_name,
            "input_dir": str(input_dir),
            "glob": args.glob,
            "recursive": args.recursive,
            "focus_surf_ids": sorted(FOCUS_SURF_IDS),
            "matched_files_before_filter": len(matched_files),
            "excluded_non_focus_files": excluded_files,
            "effective_inputs_after_collapse": len(processing_inputs),
            "method": "lstsq",
            "roc_mode": args.roc_mode,
            "sphere_fit_mode": args.sphere_fit_mode,
            "center_weight": args.center_weight,
            "normalization_mode": args.normalization_mode,
            "fixed_roc_um": args.fixed_roc_um,
            "common_reference_radius_um": common_radius_um,
            "common_reference_radius_by_surf_id": common_radius_by_surf_id or None,
            "summary_report_path": str(summary_report_path),
            "batch_zp_report_path": str(batch_zp_report_path),
            "n_modes": args.n_modes,
            "rcond": args.rcond,
            "round_radii_um": args.round_radii_um,
            "zernike_coeff_sigfigs": args.zernike_coeff_sigfigs,
            "zero_vertex_tilt": args.zero_vertex_tilt,
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
                "method": "lstsq",
                "roc_mode": args.roc_mode,
                "sphere_fit_mode": args.sphere_fit_mode,
                "center_weight": args.center_weight,
                "normalization_mode": args.normalization_mode,
                "fixed_roc_um": args.fixed_roc_um,
                "common_reference_radius_um": common_radius_um,
                "common_reference_radius_by_surf_id": json.dumps(common_radius_by_surf_id)
                if common_radius_by_surf_id
                else None,
                "n_modes": args.n_modes,
                "rcond": args.rcond,
                "round_radii_um": args.round_radii_um,
                "zernike_coeff_sigfigs": args.zernike_coeff_sigfigs,
                "zero_vertex_tilt": args.zero_vertex_tilt,
                "run_dir": str(run_dir),
            },
            artifacts=artifacts,
        )

    if args.qa_report and artifacts:
        qa_path = write_qa_report(
            run_dir, artifacts, summary_report_path=summary_report_path, analysis_date=analysis_date
        )
        print(f"qa report: {qa_path}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
