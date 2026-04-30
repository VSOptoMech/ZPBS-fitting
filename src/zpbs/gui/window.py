from __future__ import annotations

import json
import os
import shlex
import shutil
import sys
import tempfile
from datetime import datetime
from html import escape
from collections.abc import Callable
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str((Path(__file__).resolve().parents[3] / ".matplotlib").resolve()))

from PyQt5.QtCore import QEvent, QProcess, QProcessEnvironment, QThread, QTimer, Qt
from PyQt5.QtGui import QCloseEvent, QColor, QPalette
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QSpinBox,
    QSizePolicy,
    QTableWidget,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..common import (
    format_mae_rms_display,
    is_focus_surface_family,
    parse_boolish,
    validate_sphere_reference_configuration,
)
from ..io.xyz import collapse_identical_initial_inputs, parse_surface_metadata
from .canvases import FitPreviewCanvas, NumericTableWidgetItem, OverviewPlotCanvas
from .single_file import (
    SingleFileAnalysisRequest,
    SingleFileAnalysisResult,
    SingleFileAnalysisWorker,
    VertexTiltCorrection,
    build_tilt_corrected_single_file_result,
    build_single_file_candidates,
    tilt_correction_summary_rows,
    validate_single_file_source,
)
from .support import (
    NORMALIZATION_MODE_LABELS,
    ROC_MODE_LABELS,
    SPHERE_FIT_MODE_LABELS,
    format_metric,
    parse_optional_float,
)
from ..io.remap import infer_original_run_dir, infer_original_source_root, prepare_summary_row_for_preview
from ..io.workbook import (
    is_compact_summary_rows,
    parse_inline_xlsx_rows,
    parse_run_manifest,
)

SCRIPT_DIR = Path(__file__).resolve().parents[3]
CLI_MODULE = "zpbs.cli.batch_cli"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "batch_outputs"
FOCUS_SURF_IDS = ("AA", "AP", "PA", "PP")
PATH_FIELD_MIN_WIDTH = 360
FIT_CONTROL_WIDTH = 250
BROWSE_BUTTON_WIDTH = 92
LARGE_BATCH_WARNING_THRESHOLD = 1000


class BatchFitWindow(QMainWindow):
    """Launcher, single-file inspector, and summary viewer."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Batch Fit Launcher and Viewer")
        self.resize(1380, 940)
        self.process: QProcess | None = None
        self._process_started_at: datetime | None = None
        self._active_run_dir: Path | None = None
        self._active_total_inputs: int | None = None
        self._processed_progress_count = 0
        self._process_output_buffer = ""
        self.summary_rows: list[dict[str, str]] = []
        self.summary_manifest: dict[str, object] = {}
        self.summary_workbook_path: Path | None = None
        self.common_rho_axis_limit_um: float | None = None
        self.single_file_temp_root = Path(tempfile.mkdtemp(prefix="zpbs_single_file_"))
        self.single_file_candidates: list[Path] = []
        self.single_file_results_cache: dict[tuple[object, ...], SingleFileAnalysisResult] = {}
        self.single_file_result: SingleFileAnalysisResult | None = None
        self._single_file_request_id = 0
        self._single_file_active_request_id = 0
        self._single_file_active_request: SingleFileAnalysisRequest | None = None
        self._single_file_pending_request: SingleFileAnalysisRequest | None = None
        self._single_file_ignore_cache_for_request_id: int | None = None
        self._single_file_thread: QThread | None = None
        self._single_file_worker: SingleFileAnalysisWorker | None = None
        self._single_file_debounce_timer = QTimer(self)
        self._single_file_debounce_timer.setSingleShot(True)
        self._single_file_debounce_timer.setInterval(225)
        self._single_file_debounce_timer.timeout.connect(self._dispatch_single_file_analysis)

        tabs = QTabWidget(self)
        self.setCentralWidget(tabs)
        tabs.addTab(self._build_runner_tab(), "Run Batch")
        tabs.addTab(self._build_single_file_tab(), "Single File")
        tabs.addTab(self._build_viewer_tab(), "Inspect Summary")

        self.refresh_command_preview()

    def _build_runner_tab(self) -> QWidget:
        tab = QWidget(self)
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)
        self._apply_runner_tab_theme(tab)
        layout.addWidget(self._build_header())

        splitter = QSplitter(Qt.Horizontal, tab)
        splitter.setChildrenCollapsible(False)

        left = QWidget(splitter)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)
        left_layout.addWidget(self._build_paths_group())
        left_layout.addWidget(self._build_fit_group())
        left_layout.addWidget(self._build_runtime_group())
        left_layout.addWidget(self._build_runner_actions())
        left_layout.addStretch(1)

        right = QWidget(splitter)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)
        right_layout.addWidget(self._build_command_preview_group())
        right_layout.addWidget(self._build_log_group(), stretch=1)

        left.setMinimumWidth(600)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([700, 920])

        layout.addWidget(splitter, stretch=1)
        return tab

    def _build_single_file_tab(self) -> QWidget:
        tab = QWidget(self)
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)
        self._apply_runner_tab_theme(tab)

        layout.addWidget(self._build_single_file_controls_group())

        splitter = QSplitter(Qt.Horizontal, tab)
        splitter.setChildrenCollapsible(False)

        left = QWidget(splitter)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)
        left_layout.addWidget(self._build_single_file_fit_data_group(), stretch=1)
        splitter.addWidget(left)

        right = QWidget(splitter)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)
        overview_group = QGroupBox("Overview Plot", right)
        overview_layout = QVBoxLayout(overview_group)
        self.single_file_overview_canvas = OverviewPlotCanvas(overview_group)
        self.single_file_overview_canvas.clear_message("Select a maintained AA/AP/PA/PP file to analyze.")
        overview_layout.addWidget(self.single_file_overview_canvas, stretch=1)
        right_layout.addWidget(overview_group, stretch=1)
        splitter.addWidget(right)
        splitter.setSizes([560, 840])

        layout.addWidget(splitter, stretch=1)
        self._refresh_single_file_export_buttons()
        return tab

    def _build_single_file_controls_group(self) -> QGroupBox:
        group = QGroupBox("Single File Controls", self)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(14, 16, 14, 14)
        layout.setSpacing(10)

        self.single_file_path_edit = QLineEdit(self)
        self.single_file_folder_edit = QLineEdit(self)
        self.single_file_prev_button = QPushButton("Prev", self)
        self.single_file_prev_button.setMaximumWidth(72)
        self.single_file_force_combo = QComboBox(self)
        self.single_file_force_combo.setMinimumWidth(100)
        self.single_file_next_button = QPushButton("Next", self)
        self.single_file_next_button.setMaximumWidth(72)
        self.single_file_refresh_button = QPushButton("Refresh", self)
        self.single_file_refresh_button.setMaximumWidth(92)
        self.single_file_surf_buttons: dict[str, QPushButton] = {}
        self.single_file_state_buttons: dict[str, QPushButton] = {}

        self.single_file_sphere_fit_mode_combo = QComboBox(self)
        for value, label in SPHERE_FIT_MODE_LABELS.items():
            self.single_file_sphere_fit_mode_combo.addItem(label, value)
        self.single_file_sphere_fit_mode_combo.setCurrentIndex(
            self.single_file_sphere_fit_mode_combo.findData("center_weighted")
        )

        self.single_file_center_weight_spin = QDoubleSpinBox(self)
        self.single_file_center_weight_spin.setRange(0.0, 5.0)
        self.single_file_center_weight_spin.setDecimals(2)
        self.single_file_center_weight_spin.setSingleStep(0.05)
        self.single_file_center_weight_spin.setValue(0.5)

        self.single_file_n_modes_spin = QSpinBox(self)
        self.single_file_n_modes_spin.setRange(1, 45)
        self.single_file_n_modes_spin.setValue(45)

        self.single_file_round_radii_check = QCheckBox("Round radii", self)
        self.single_file_round_coeffs_check = QCheckBox("Round coefficients", self)
        self.single_file_tilt_correction_check = QCheckBox("Tilt correction", self)
        self.single_file_round_radii_check.setChecked(True)
        self.single_file_round_coeffs_check.setChecked(True)
        self.single_file_tilt_correction_check.setToolTip(
            "Single File only: adjust displayed/exported Z2/Z3 so the net Zernike center slope is zero."
        )
        self.single_file_save_csv_button = QPushButton("Save CSV...", self)
        self.single_file_save_overview_button = QPushButton("Save Overview Plot...", self)
        self.single_file_save_bundle_button = QPushButton("Save Bundle...", self)
        self.single_file_save_csv_button.clicked.connect(self.save_single_file_csv)
        self.single_file_save_overview_button.clicked.connect(self.save_single_file_overview_plot)
        self.single_file_save_bundle_button.clicked.connect(self.save_single_file_bundle)

        for widget in (
            self.single_file_path_edit,
            self.single_file_folder_edit,
        ):
            widget.setMinimumWidth(PATH_FIELD_MIN_WIDTH)

        self.single_file_sphere_fit_mode_combo.setFixedWidth(220)

        for control in (
            self.single_file_center_weight_spin,
            self.single_file_n_modes_spin,
        ):
            control.setFixedWidth(80)

        file_row = QWidget(self)
        file_layout = QGridLayout(file_row)
        file_layout.setContentsMargins(0, 0, 0, 0)
        file_layout.setHorizontalSpacing(12)
        file_layout.setVerticalSpacing(8)
        file_layout.addWidget(QLabel("Current File"), 0, 0)
        file_layout.addWidget(
            self._with_browse_button(
                self.single_file_path_edit,
                directory=False,
                on_selected=lambda: self._on_single_file_path_committed(show_warning=True),
            ),
            0,
            1,
        )
        file_layout.addWidget(QLabel("Preload Folder"), 1, 0)
        file_layout.addWidget(
            self._with_browse_button(
                self.single_file_folder_edit,
                directory=True,
                on_selected=self._on_single_file_folder_committed,
            ),
            1,
            1,
        )
        file_layout.setColumnStretch(1, 1)

        selector_row = QWidget(self)
        selector_layout = QHBoxLayout(selector_row)
        selector_layout.setContentsMargins(0, 0, 0, 0)
        selector_layout.setSpacing(12)

        surf_widget = QWidget(selector_row)
        surf_layout = QHBoxLayout(surf_widget)
        surf_layout.setContentsMargins(0, 0, 0, 0)
        surf_layout.setSpacing(8)
        surf_layout.addWidget(QLabel("Surface Family", surf_widget))
        for surf_id in FOCUS_SURF_IDS:
            button = QPushButton(surf_id, surf_widget)
            button.setCheckable(True)
            button.setAutoExclusive(True)
            button.setEnabled(False)
            button.clicked.connect(lambda _checked, _surf_id=surf_id: self._refresh_single_file_force_options())
            self.single_file_surf_buttons[surf_id] = button
            surf_layout.addWidget(button)
        selector_layout.addWidget(surf_widget)

        selector_layout.addSpacing(8)
        selector_layout.addWidget(QLabel("Force", selector_row))
        selector_layout.addWidget(self.single_file_prev_button)
        selector_layout.addWidget(self.single_file_force_combo)
        selector_layout.addWidget(self.single_file_next_button)

        selector_layout.addSpacing(8)
        state_widget = QWidget(selector_row)
        state_layout = QHBoxLayout(state_widget)
        state_layout.setContentsMargins(0, 0, 0, 0)
        state_layout.setSpacing(8)
        state_layout.addWidget(QLabel("Surface State", state_widget))
        for suffix, label in (("I", "Initial"), ("D", "Deformed")):
            button = QPushButton(label, state_widget)
            button.setCheckable(True)
            button.setAutoExclusive(True)
            button.setEnabled(False)
            button.clicked.connect(lambda _checked, _suffix=suffix: self._apply_single_file_selection_from_controls())
            self.single_file_state_buttons[suffix] = button
            state_layout.addWidget(button)
        state_layout.addWidget(self.single_file_tilt_correction_check)
        selector_layout.addWidget(state_widget)
        selector_layout.addStretch(1)
        selector_layout.addWidget(self.single_file_refresh_button)

        options_row = QWidget(self)
        options_layout = QHBoxLayout(options_row)
        options_layout.setContentsMargins(0, 0, 0, 0)
        options_layout.setSpacing(12)
        options_layout.addWidget(QLabel("Sphere Fit"))
        options_layout.addWidget(self.single_file_sphere_fit_mode_combo)
        options_layout.addWidget(QLabel("Center Weight"))
        options_layout.addWidget(self.single_file_center_weight_spin)
        options_layout.addWidget(QLabel("N Modes"))
        options_layout.addWidget(self.single_file_n_modes_spin)
        options_layout.addWidget(self.single_file_round_radii_check)
        options_layout.addWidget(self.single_file_round_coeffs_check)
        options_layout.addSpacing(8)
        options_layout.addWidget(self.single_file_save_csv_button)
        options_layout.addWidget(self.single_file_save_overview_button)
        options_layout.addWidget(self.single_file_save_bundle_button)
        options_layout.addStretch(1)

        self.single_file_status_label = QLabel("Select a source file or preload a folder.")
        self.single_file_status_label.setWordWrap(True)
        self.single_file_status_label.setProperty("themeRole", "mutedText")
        self.single_file_status_label.setStyleSheet(f"color: {self._muted_text_color()};")

        layout.addWidget(file_row)
        layout.addWidget(selector_row)
        layout.addWidget(options_row)
        layout.addWidget(self.single_file_status_label)

        self.single_file_path_edit.editingFinished.connect(
            lambda: self._on_single_file_path_committed(show_warning=True)
        )
        self.single_file_folder_edit.editingFinished.connect(self._on_single_file_folder_committed)
        self.single_file_force_combo.currentTextChanged.connect(lambda _text: self._refresh_single_file_state_options())
        self.single_file_prev_button.clicked.connect(lambda: self._navigate_single_file_force(-1))
        self.single_file_next_button.clicked.connect(lambda: self._navigate_single_file_force(1))
        self.single_file_refresh_button.clicked.connect(self._refresh_single_file_selection)
        self.single_file_sphere_fit_mode_combo.currentIndexChanged.connect(self._on_single_file_option_changed)
        self.single_file_center_weight_spin.valueChanged.connect(self._on_single_file_option_changed)
        self.single_file_n_modes_spin.valueChanged.connect(self._on_single_file_option_changed)
        self.single_file_round_radii_check.stateChanged.connect(self._on_single_file_option_changed)
        self.single_file_round_coeffs_check.stateChanged.connect(self._on_single_file_option_changed)
        self.single_file_tilt_correction_check.stateChanged.connect(self._on_single_file_tilt_correction_changed)

        self._sync_single_file_controls()
        return group

    def _build_single_file_fit_data_group(self) -> QGroupBox:
        group = QGroupBox("Diagnostics and Fit Data", self)
        layout = QVBoxLayout(group)
        self.single_file_fit_data_table = QTableWidget(self)
        self.single_file_fit_data_table.setColumnCount(2)
        self.single_file_fit_data_table.setHorizontalHeaderLabels(["Field", "Value"])
        self.single_file_fit_data_table.verticalHeader().setVisible(False)
        self.single_file_fit_data_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.single_file_fit_data_table.setSelectionMode(QTableWidget.NoSelection)
        self.single_file_fit_data_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.single_file_fit_data_table)
        return group

    def _set_single_file_status(self, text: str) -> None:
        self.single_file_status_label.setText(text)

    def _clear_single_file_result_views(self, message: str) -> None:
        self.single_file_result = None
        self.single_file_overview_canvas.clear_message(message)
        self._populate_name_value_table(self.single_file_fit_data_table, [])
        self._refresh_single_file_export_buttons()

    def _populate_name_value_table(self, table: QTableWidget, rows: list[tuple[str, str]]) -> None:
        table.setRowCount(len(rows))
        for row_index, (name, value) in enumerate(rows):
            table.setItem(row_index, 0, NumericTableWidgetItem(name, name))
            table.setItem(row_index, 1, NumericTableWidgetItem(value, value))
        table.resizeColumnsToContents()

    def _diagnostic_rows_for_result(self, result: SingleFileAnalysisResult) -> list[tuple[str, str]]:
        artifacts = result.artifacts
        rows = [
            ("Source file", str(artifacts.source_file)),
            ("Temp result directory", str(result.result_dir)),
            ("Points used", str(artifacts.points_used)),
            ("Method", artifacts.method),
            ("Sphere fit mode", artifacts.sphere_fit_mode),
            ("Center weight", f"{artifacts.center_weight:.2f}"),
            ("Normalization mode", artifacts.normalization_mode),
            ("Round radii", "on" if artifacts.round_radii_um else "off"),
            ("Round coefficients", str(artifacts.zernike_coeff_sigfigs or "off")),
            ("Observed aperture radius (um)", format_metric(artifacts.observed_aperture_radius_um, precision=4)),
            ("Reference vertex z (um)", format_metric(artifacts.reference_vertex_z_um, precision=4)),
            ("Sphere MAE (um)", format_mae_rms_display(artifacts.sphere_mae_um, precision=3)),
            ("Sphere RMS (um)", format_mae_rms_display(artifacts.sphere_rms_um, precision=3)),
            ("ZPBS residual MAE (um)", format_mae_rms_display(artifacts.zpbs_residual_mae_um, precision=3)),
            ("ZPBS residual RMS (um)", format_mae_rms_display(artifacts.zpbs_residual_rms_um, precision=3)),
            ("ZPBS residual cond", format_metric(artifacts.zpbs_residual_cond, precision=3)),
            (
                "ZPBS residual on-axis m0 (um)",
                format_mae_rms_display(artifacts.zpbs_residual_on_axis_m0_um, precision=3),
            ),
            ("Coefficient CSV", str(result.csv_path)),
            ("Overview plot", str(result.overview_plot_path)),
            ("Analysis JSON", str(result.analysis_json_path)),
        ]
        if artifacts.metadata.filename_kind == "suffixed":
            rows.append(("Filename suffix", artifacts.metadata.filename_suffix))
        elif artifacts.metadata.filename_kind == "generic":
            rows.extend(
                [
                    ("Filename kind", "generic"),
                    ("Surface convention", artifacts.metadata.surf_id),
                ]
            )
        if result.summary_row.get("vertex_tilt_correction") == "on":
            rows.extend(
                tilt_correction_summary_rows(
                    VertexTiltCorrection(
                        original_x_mrad=float(result.summary_row["original_center_slope_x_mrad"]),
                        original_y_mrad=float(result.summary_row["original_center_slope_y_mrad"]),
                        corrected_x_mrad=float(result.summary_row["corrected_center_slope_x_mrad"]),
                        corrected_y_mrad=float(result.summary_row["corrected_center_slope_y_mrad"]),
                        delta_z2_um=float(result.summary_row["delta_z2_um"]),
                        delta_z3_um=float(result.summary_row["delta_z3_um"]),
                    )
                )
            )
        return rows

    def _fit_data_rows_for_result(self, result: SingleFileAnalysisResult) -> list[tuple[str, str]]:
        coeff_rows = list(result.coeff_rows)
        diagnostic_rows = self._diagnostic_rows_for_result(result)
        if not diagnostic_rows:
            return coeff_rows
        insert_at = next(
            (index + 1 for index, (name, _value) in enumerate(coeff_rows) if name == "Norm. Radius (mm)"),
            len(coeff_rows),
        )
        return [*coeff_rows[:insert_at], *diagnostic_rows, *coeff_rows[insert_at:]]

    def _sync_single_file_controls(self) -> None:
        sphere_fit_mode = str(self.single_file_sphere_fit_mode_combo.currentData())
        self.single_file_center_weight_spin.setEnabled(sphere_fit_mode == "center_weighted")
        self._update_single_file_force_navigation_buttons()

    def _set_single_file_generic_navigation_state(self) -> None:
        for button in self.single_file_surf_buttons.values():
            button.blockSignals(True)
            button.setChecked(False)
            button.setEnabled(False)
            button.blockSignals(False)
        self.single_file_force_combo.blockSignals(True)
        self.single_file_force_combo.clear()
        self.single_file_force_combo.setEnabled(False)
        self.single_file_force_combo.blockSignals(False)
        for button in self.single_file_state_buttons.values():
            button.blockSignals(True)
            button.setChecked(False)
            button.setEnabled(False)
            button.blockSignals(False)
        self.single_file_prev_button.setEnabled(False)
        self.single_file_next_button.setEnabled(False)

    def _refresh_single_file_export_buttons(self) -> None:
        enabled = self.single_file_result is not None
        for button in (
            self.single_file_save_csv_button,
            self.single_file_save_overview_button,
            self.single_file_save_bundle_button,
        ):
            button.setEnabled(enabled)

    def _current_single_file_source(self) -> Path | None:
        text = self.single_file_path_edit.text().strip()
        return Path(text) if text else None

    def _single_file_sigfigs(self) -> int | None:
        return 6 if self.single_file_round_coeffs_check.isChecked() else None

    def _build_single_file_request(self) -> SingleFileAnalysisRequest | None:
        source_file = self._current_single_file_source()
        if source_file is None:
            return None
        self._single_file_request_id += 1
        return SingleFileAnalysisRequest(
            request_id=self._single_file_request_id,
            source_file=source_file,
            sphere_fit_mode=str(self.single_file_sphere_fit_mode_combo.currentData()),
            center_weight=float(self.single_file_center_weight_spin.value()),
            n_modes=int(self.single_file_n_modes_spin.value()),
            round_radii_um=self.single_file_round_radii_check.isChecked(),
            zernike_coeff_sigfigs=self._single_file_sigfigs(),
            temp_root=self.single_file_temp_root,
        )

    def _queue_single_file_analysis(self, *, force: bool = False) -> None:
        request = self._build_single_file_request()
        if request is None:
            self._set_single_file_status("Select a source file or preload a folder.")
            self._clear_single_file_result_views("Select a maintained AA/AP/PA/PP file to analyze.")
            return
        self._single_file_pending_request = request
        if force:
            self._single_file_ignore_cache_for_request_id = request.request_id
            if request.source_file.exists():
                self.single_file_results_cache.pop(request.cache_key(), None)
        self._single_file_debounce_timer.start()

    def _dispatch_single_file_analysis(self) -> None:
        request = self._single_file_pending_request
        self._single_file_pending_request = None
        if request is None:
            return
        try:
            validate_single_file_source(request.source_file)
        except ValueError as exc:
            self._set_single_file_status(str(exc))
            self._clear_single_file_result_views(str(exc))
            return

        cache_key = request.cache_key()
        ignore_cache = self._single_file_ignore_cache_for_request_id == request.request_id
        if not ignore_cache:
            cached = self.single_file_results_cache.get(cache_key)
            if cached is not None:
                self._single_file_active_request_id = request.request_id
                self._render_single_file_result(cached, cached_result=True)
                return

        if self._single_file_thread is not None:
            self._single_file_pending_request = request
            self._set_single_file_status(f"Queued latest analysis for {request.source_file.name}...")
            return

        self._single_file_active_request_id = request.request_id
        self._single_file_active_request = request
        self._single_file_ignore_cache_for_request_id = None
        self._set_single_file_status(f"Analyzing {request.source_file.name}...")
        worker = SingleFileAnalysisWorker(request)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._single_file_analysis_finished)
        worker.failed.connect(self._single_file_analysis_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(self._single_file_thread_finished)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._single_file_worker = worker
        self._single_file_thread = thread
        thread.start()

    def _render_single_file_result(self, result: SingleFileAnalysisResult, *, cached_result: bool) -> None:
        self.single_file_result = result
        self.single_file_results_cache[result.cache_key] = result
        display_result = self._single_file_display_result()
        if display_result is None:
            return
        self.single_file_overview_canvas.plot_artifacts(display_result.artifacts)
        self._populate_name_value_table(self.single_file_fit_data_table, self._fit_data_rows_for_result(display_result))
        status_prefix = "Loaded cached analysis" if cached_result else "Analysis ready"
        correction_suffix = " | tilt correction on" if display_result is not result else ""
        self._set_single_file_status(
            f"{status_prefix}: {result.artifacts.source_file.name} | temp: {result.result_dir}{correction_suffix}"
        )
        self._refresh_single_file_export_buttons()

    def _single_file_display_result(self) -> SingleFileAnalysisResult | None:
        if self.single_file_result is None:
            return None
        if self.single_file_tilt_correction_check.isChecked():
            return build_tilt_corrected_single_file_result(self.single_file_result)
        return self.single_file_result

    def _on_single_file_tilt_correction_changed(self) -> None:
        if self.single_file_result is None:
            return
        self._render_single_file_result(self.single_file_result, cached_result=True)

    def _single_file_analysis_finished(self, result: SingleFileAnalysisResult) -> None:
        if result.request_id != self._single_file_active_request_id:
            return
        self._render_single_file_result(result, cached_result=False)

    def _single_file_analysis_failed(self, request_id: int, message: str) -> None:
        if request_id != self._single_file_active_request_id:
            return
        self._set_single_file_status(message)
        self._clear_single_file_result_views(message)

    def _single_file_thread_finished(self) -> None:
        self._single_file_worker = None
        self._single_file_thread = None
        self._single_file_active_request = None
        pending_request = self._single_file_pending_request
        if pending_request is not None:
            self._single_file_debounce_timer.start()

    def _on_single_file_option_changed(self) -> None:
        self._sync_single_file_controls()
        if self._current_single_file_source() is not None:
            self._queue_single_file_analysis()

    def _single_file_metadata(self, source_file: Path) -> object:
        return parse_surface_metadata(source_file)

    def _current_single_file_is_generic(self) -> bool:
        source_file = self._current_single_file_source()
        return source_file is not None and self._single_file_metadata(source_file).filename_kind == "generic"

    def _single_file_selected_surf_id(self) -> str:
        for surf_id, button in self.single_file_surf_buttons.items():
            if button.isChecked():
                return surf_id
        return ""

    def _single_file_selected_state_suffix(self) -> str:
        for suffix, button in self.single_file_state_buttons.items():
            if button.isChecked():
                return suffix
        return ""

    def _single_file_current_candidate(self) -> Path | None:
        if self._current_single_file_is_generic():
            return None
        surf_id = self._single_file_selected_surf_id()
        force_id = self.single_file_force_combo.currentText()
        suffix = self._single_file_selected_state_suffix()
        if not surf_id or not force_id or not suffix:
            return None
        for path in self.single_file_candidates:
            metadata = self._single_file_metadata(path)
            if (
                metadata.surf_id == surf_id
                and metadata.force_id == force_id
                and metadata.surface_token.endswith(suffix)
            ):
                return path
        return None

    def _set_single_file_candidates(self, candidates: list[Path], preferred: Path | None = None) -> None:
        self.single_file_candidates = list(candidates)
        self._refresh_single_file_surf_options(preferred=preferred)

    def _refresh_single_file_surf_options(self, preferred: Path | None = None) -> None:
        preferred_metadata = (
            self._single_file_metadata(preferred) if preferred is not None and preferred.exists() else None
        )
        if preferred_metadata is not None and preferred_metadata.filename_kind == "generic":
            self._set_single_file_generic_navigation_state()
            return
        available = {self._single_file_metadata(path).surf_id for path in self.single_file_candidates}
        preferred_surf_id = preferred_metadata.surf_id if preferred_metadata is not None else ""
        current = self._single_file_selected_surf_id()
        target = (
            current
            if current in available
            else preferred_surf_id
            if preferred_surf_id in available
            else next(
                (surf_id for surf_id in FOCUS_SURF_IDS if surf_id in available),
                None,
            )
        )
        self.single_file_force_combo.setEnabled(bool(target))
        for surf_id, button in self.single_file_surf_buttons.items():
            button.blockSignals(True)
            button.setEnabled(surf_id in available)
            button.setChecked(surf_id == target)
            button.blockSignals(False)
        self._refresh_single_file_force_options(preferred=preferred)

    def _refresh_single_file_force_options(self, preferred: Path | None = None) -> None:
        self.single_file_force_combo.setEnabled(True)
        surf_id = self._single_file_selected_surf_id()
        matching = [path for path in self.single_file_candidates if self._single_file_metadata(path).surf_id == surf_id]
        force_ids = sorted({self._single_file_metadata(path).force_id for path in matching}, key=self._force_sort_key)
        preferred_force_id = (
            self._single_file_metadata(preferred).force_id
            if preferred is not None and preferred.exists() and self._single_file_metadata(preferred).surf_id == surf_id
            else ""
        )
        current = self.single_file_force_combo.currentText()
        target = current if current in force_ids else preferred_force_id if preferred_force_id in force_ids else ""
        self.single_file_force_combo.blockSignals(True)
        self.single_file_force_combo.clear()
        self.single_file_force_combo.addItems(force_ids)
        if target:
            self.single_file_force_combo.setCurrentText(target)
        elif force_ids:
            self.single_file_force_combo.setCurrentIndex(0)
        self.single_file_force_combo.blockSignals(False)
        self._refresh_single_file_state_options(preferred=preferred)

    def _refresh_single_file_state_options(self, preferred: Path | None = None) -> None:
        surf_id = self._single_file_selected_surf_id()
        force_id = self.single_file_force_combo.currentText()
        matching = [
            path
            for path in self.single_file_candidates
            if self._single_file_metadata(path).surf_id == surf_id
            and self._single_file_metadata(path).force_id == force_id
        ]
        available_suffixes = {
            self._single_file_metadata(path).surface_token[-1].upper()
            for path in matching
            if self._single_file_metadata(path).surface_token
        }
        preferred_suffix = (
            self._single_file_metadata(preferred).surface_token[-1].upper()
            if preferred is not None
            and preferred.exists()
            and self._single_file_metadata(preferred).surf_id == surf_id
            and self._single_file_metadata(preferred).force_id == force_id
            and self._single_file_metadata(preferred).surface_token
            else ""
        )
        current = self._single_file_selected_state_suffix()
        target = (
            current
            if current in available_suffixes
            else preferred_suffix
            if preferred_suffix in available_suffixes
            else next(
                (suffix for suffix in ("I", "D") if suffix in available_suffixes),
                None,
            )
        )
        for suffix, button in self.single_file_state_buttons.items():
            button.blockSignals(True)
            button.setEnabled(suffix in available_suffixes)
            button.setChecked(suffix == target)
            button.blockSignals(False)
        self._update_single_file_force_navigation_buttons()
        self._apply_single_file_selection_from_controls()

    def _apply_single_file_selection_from_controls(self) -> None:
        source_file = self._single_file_current_candidate()
        self._sync_single_file_controls()
        if source_file is None:
            return
        self.single_file_path_edit.blockSignals(True)
        self.single_file_path_edit.setText(str(source_file))
        self.single_file_path_edit.blockSignals(False)
        self._queue_single_file_analysis()

    def _navigate_single_file_force(self, step: int) -> None:
        if self.single_file_force_combo.count() == 0:
            return
        current = self.single_file_force_combo.currentIndex()
        if current < 0:
            current = 0
        target = min(max(current + step, 0), self.single_file_force_combo.count() - 1)
        if target != current:
            self.single_file_force_combo.setCurrentIndex(target)
            self._refresh_single_file_state_options()
        self._update_single_file_force_navigation_buttons()

    def _update_single_file_force_navigation_buttons(self) -> None:
        if self._current_single_file_is_generic():
            self.single_file_prev_button.setEnabled(False)
            self.single_file_next_button.setEnabled(False)
            return
        count = self.single_file_force_combo.count()
        current = self.single_file_force_combo.currentIndex()
        has_items = count > 0 and current >= 0
        self.single_file_prev_button.setEnabled(has_items and current > 0)
        self.single_file_next_button.setEnabled(has_items and current < count - 1)

    def _on_single_file_folder_committed(self) -> None:
        folder_text = self.single_file_folder_edit.text().strip()
        if not folder_text:
            self.single_file_candidates = []
            self._set_single_file_candidates([], None)
            return
        folder = Path(folder_text)
        if not folder.exists():
            QMessageBox.warning(self, "Invalid Folder", f"Folder does not exist:\n{folder}")
            return
        candidates = build_single_file_candidates(folder)
        preferred = self._current_single_file_source()
        self._set_single_file_candidates(candidates, preferred=preferred)
        if candidates:
            candidate_resolved = {path.resolve() for path in candidates}
            selected_path = preferred if preferred is not None and preferred.resolve() in candidate_resolved else None
            if selected_path is None:
                selected_path = next(
                    (
                        path
                        for path in candidates
                        if self._single_file_metadata(path).filename_kind != "generic"
                        and is_focus_surface_family(self._single_file_metadata(path).surf_id)
                    ),
                    candidates[0],
                )
            self._apply_single_file_source(selected_path, queue_analysis=True)
            self._set_single_file_status(f"Loaded {len(candidates)} Single File-compatible .xyz files from {folder}.")
        else:
            self._set_single_file_status(f"No Single File-compatible .xyz files were found in {folder}.")
            self._clear_single_file_result_views(
                "No Single File-compatible .xyz files were found in the selected folder."
            )

    def _apply_single_file_source(self, source_file: Path, *, queue_analysis: bool) -> None:
        self.single_file_path_edit.blockSignals(True)
        self.single_file_path_edit.setText(str(source_file))
        self.single_file_path_edit.blockSignals(False)
        self._refresh_single_file_surf_options(preferred=source_file)
        if queue_analysis and self._single_file_current_candidate() is None:
            self._queue_single_file_analysis()

    def _on_single_file_path_committed(self, *, show_warning: bool) -> None:
        source_file = self._current_single_file_source()
        if source_file is None:
            self._clear_single_file_result_views("Select a maintained AA/AP/PA/PP file to analyze.")
            return
        try:
            validate_single_file_source(source_file)
        except ValueError as exc:
            if show_warning:
                QMessageBox.warning(self, "Unsupported File", str(exc))
            self._set_single_file_status(str(exc))
            self._clear_single_file_result_views(str(exc))
            return

        self.single_file_folder_edit.setText(str(source_file.parent))
        candidates = build_single_file_candidates(source_file.parent)
        self._set_single_file_candidates(candidates, preferred=source_file)
        self._apply_single_file_source(source_file, queue_analysis=True)

    def _refresh_single_file_selection(self) -> None:
        folder_text = self.single_file_folder_edit.text().strip()
        if folder_text:
            folder = Path(folder_text)
            if folder.exists():
                candidates = build_single_file_candidates(folder)
                self._set_single_file_candidates(candidates, preferred=self._current_single_file_source())
        if self._current_single_file_source() is not None:
            self._queue_single_file_analysis(force=True)

    def _copy_single_file_artifact(
        self, source_path: Path, *, caption: str, default_name: str, file_filter: str
    ) -> None:
        selected, _ = QFileDialog.getSaveFileName(self, caption, str(source_path.with_name(default_name)), file_filter)
        if not selected:
            return
        shutil.copy2(source_path, selected)

    def save_single_file_csv(self) -> None:
        display_result = self._single_file_display_result()
        if display_result is None:
            return
        self._copy_single_file_artifact(
            display_result.csv_path,
            caption="Save Coefficient CSV",
            default_name=display_result.csv_path.name,
            file_filter="CSV Files (*.csv);;All Files (*)",
        )

    def save_single_file_overview_plot(self) -> None:
        display_result = self._single_file_display_result()
        if display_result is None:
            return
        self._copy_single_file_artifact(
            display_result.overview_plot_path,
            caption="Save Overview Plot",
            default_name=display_result.overview_plot_path.name,
            file_filter="PNG Files (*.png);;All Files (*)",
        )

    def save_single_file_bundle(self) -> None:
        if self.single_file_result is None:
            return
        target_parent = QFileDialog.getExistingDirectory(
            self,
            "Select Bundle Destination",
            str(self.single_file_result.result_dir.parent),
        )
        if not target_parent:
            return
        target_dir = Path(target_parent) / self.single_file_result.export_dir_name
        if target_dir.exists():
            QMessageBox.warning(self, "Bundle Exists", f"Destination already exists:\n{target_dir}")
            return
        shutil.copytree(self.single_file_result.result_dir, target_dir)
        self._set_single_file_status(f"Saved bundle to {target_dir}")

    def changeEvent(self, event: QEvent) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.PaletteChange:
            if isinstance(self.centralWidget(), QTabWidget):
                for index in (0, 1):
                    themed_tab = self.centralWidget().widget(index)
                    if isinstance(themed_tab, QWidget):
                        self._apply_runner_tab_theme(themed_tab)
            self._refresh_theme_styles()

    def _apply_runner_tab_theme(self, tab: QWidget) -> None:
        window_color = self._color_css(self.palette().color(QPalette.Base))
        border_color = self._color_css(self.palette().color(QPalette.Mid))
        title_color = self._color_css(self.palette().color(QPalette.WindowText))
        tab.setStyleSheet(
            f"""
            QGroupBox {{
                border: 1px solid {border_color};
                border-radius: 10px;
                margin-top: 14px;
                padding-top: 10px;
                background: {window_color};
                font-weight: 600;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                top: -3px;
                left: 12px;
                padding: 0 4px;
                color: {title_color};
            }}
            """
        )

    def _refresh_theme_styles(self) -> None:
        for label in self.findChildren(QLabel):
            role = label.property("themeRole")
            if role == "mutedText":
                label.setStyleSheet(f"color: {self._muted_text_color()};")
            elif role == "sectionLabel":
                label.setStyleSheet(f"font-size: 12px; font-weight: 600; color: {self._section_text_color()};")

    @staticmethod
    def _color_css(color: QColor) -> str:
        return color.name()

    def _muted_text_color(self) -> str:
        return self._color_css(self.palette().color(QPalette.Disabled, QPalette.WindowText))

    def _section_text_color(self) -> str:
        return self._color_css(self.palette().color(QPalette.WindowText))

    def _build_viewer_tab(self) -> QWidget:
        tab = QWidget(self)
        layout = QVBoxLayout(tab)
        layout.addWidget(self._build_summary_loader_group())
        layout.addWidget(self._build_summary_path_remap_group())

        splitter = QSplitter(self)
        splitter.setChildrenCollapsible(False)

        left = QWidget(self)
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(self._build_selection_group())
        left_layout.addWidget(self._build_details_group(), stretch=1)
        splitter.addWidget(left)

        right = QWidget(self)
        right_layout = QVBoxLayout(right)
        self.preview_canvas = FitPreviewCanvas(right)
        self.preview_canvas.clear_message("Load a batch summary workbook to inspect one fit.")
        right_layout.addWidget(self.preview_canvas, stretch=1)
        splitter.addWidget(right)
        splitter.setSizes([360, 980])

        layout.addWidget(splitter, stretch=1)
        return tab

    def _build_header(self) -> QWidget:
        widget = QWidget(self)
        layout = QVBoxLayout(widget)
        title = QLabel("Batch XYZ Fitter")
        title.setStyleSheet("font-size: 20px; font-weight: 600;")
        subtitle = QLabel(
            "Package-native GUI for the maintained batch runner. The runner only includes AA/AP/PA/PP files, and the viewer can load a summary workbook to inspect individual fits live."
        )
        subtitle.setWordWrap(True)
        subtitle.setProperty("themeRole", "mutedText")
        subtitle.setStyleSheet(f"color: {self._muted_text_color()};")
        layout.addWidget(title)
        layout.addWidget(subtitle)
        return widget

    def _build_paths_group(self) -> QGroupBox:
        group = QGroupBox("Paths", self)
        grid = QGridLayout(group)
        grid.setContentsMargins(14, 16, 14, 14)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(10)

        self.input_dir_edit = QLineEdit(self)
        self.output_dir_edit = QLineEdit(str(DEFAULT_OUTPUT_DIR), self)
        self.h5_path_edit = QLineEdit("batch_results.h5", self)
        self.run_name_edit = QLineEdit(self._default_run_name(), self)
        self.glob_edit = QLineEdit("*_FVS_*.xyz", self)

        for widget in (
            self.input_dir_edit,
            self.output_dir_edit,
            self.h5_path_edit,
            self.run_name_edit,
            self.glob_edit,
        ):
            widget.setMinimumWidth(PATH_FIELD_MIN_WIDTH)
            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            widget.textChanged.connect(self.refresh_command_preview)

        rows = (
            ("Input Directory", self._with_browse_button(self.input_dir_edit, directory=True)),
            ("Output Directory", self._with_browse_button(self.output_dir_edit, directory=True)),
            ("HDF5 Path", self._with_browse_button(self.h5_path_edit, directory=False, save=True)),
            ("Run Name", self.run_name_edit),
            ("Glob", self.glob_edit),
        )
        for row_index, (label_text, field_widget) in enumerate(rows):
            label = QLabel(label_text, self)
            label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            grid.addWidget(label, row_index, 0)
            grid.addWidget(field_widget, row_index, 1)
        grid.setColumnStretch(1, 1)
        return group

    def _build_fit_group(self) -> QGroupBox:
        group = QGroupBox("Fit Configuration", self)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(14, 16, 14, 14)
        layout.setSpacing(12)

        self.roc_mode_combo = QComboBox(self)
        for value, label in ROC_MODE_LABELS.items():
            self.roc_mode_combo.addItem(label, value)

        self.sphere_fit_mode_combo = QComboBox(self)
        for value, label in SPHERE_FIT_MODE_LABELS.items():
            self.sphere_fit_mode_combo.addItem(label, value)

        self.normalization_mode_combo = QComboBox(self)
        for value, label in NORMALIZATION_MODE_LABELS.items():
            self.normalization_mode_combo.addItem(label, value)

        self.fixed_roc_spin = QDoubleSpinBox(self)
        self.fixed_roc_spin.setRange(0.0, 1_000_000_000.0)
        self.fixed_roc_spin.setDecimals(6)
        self.fixed_roc_spin.setValue(0.0)
        self.fixed_roc_spin.setEnabled(False)

        self.center_weight_spin = QDoubleSpinBox(self)
        self.center_weight_spin.setRange(0.0, 5.0)
        self.center_weight_spin.setDecimals(2)
        self.center_weight_spin.setSingleStep(0.05)
        self.center_weight_spin.setValue(0.5)

        self.n_modes_spin = QSpinBox(self)
        self.n_modes_spin.setRange(1, 45)
        self.n_modes_spin.setValue(45)

        self.round_radii_check = QCheckBox("Round radii to nearest um before fitting", self)
        self.round_coeffs_check = QCheckBox("Round Zernike coefficients to 6 sig figs", self)
        self.zero_vertex_tilt_check = QCheckBox("Tilt correction", self)
        self.round_radii_check.setChecked(True)
        self.round_coeffs_check.setChecked(True)
        self.zero_vertex_tilt_check.setToolTip(
            "Adjust exported residual Z2/Z3 after fitting so the Zernike model has zero net center slope."
        )
        self.roc_mode_combo.setCurrentIndex(self.roc_mode_combo.findData("fit-per-file"))
        self.sphere_fit_mode_combo.setCurrentIndex(self.sphere_fit_mode_combo.findData("center_weighted"))
        self.normalization_mode_combo.setCurrentIndex(self.normalization_mode_combo.findData("per-file"))

        for control in (
            self.roc_mode_combo,
            self.sphere_fit_mode_combo,
            self.normalization_mode_combo,
            self.fixed_roc_spin,
            self.center_weight_spin,
            self.n_modes_spin,
        ):
            control.setFixedWidth(FIT_CONTROL_WIDTH)

        sphere_grid = QGridLayout()
        sphere_grid.setHorizontalSpacing(14)
        sphere_grid.setVerticalSpacing(8)
        sphere_grid.addWidget(self._section_label("Sphere Reference"), 0, 0, 1, 2)
        sphere_grid.addWidget(QLabel("Sphere Fit Mode"), 1, 0)
        sphere_grid.addWidget(QLabel("ROC Mode"), 1, 1)
        sphere_grid.addWidget(self.sphere_fit_mode_combo, 2, 0)
        sphere_grid.addWidget(self.roc_mode_combo, 2, 1)
        sphere_grid.addWidget(QLabel("Center Weight"), 3, 0)
        sphere_grid.addWidget(QLabel("Fixed ROC (um)"), 3, 1)
        sphere_grid.addWidget(self.center_weight_spin, 4, 0)
        sphere_grid.addWidget(self.fixed_roc_spin, 4, 1)
        sphere_grid.setColumnStretch(0, 1)
        sphere_grid.setColumnStretch(1, 1)

        residual_grid = QGridLayout()
        residual_grid.setHorizontalSpacing(14)
        residual_grid.setVerticalSpacing(8)
        residual_grid.addWidget(self._section_label("Residual Model"), 0, 0, 1, 2)
        residual_grid.addWidget(QLabel("Normalization Mode"), 1, 0)
        residual_grid.addWidget(QLabel("N Modes"), 1, 1)
        residual_grid.addWidget(self.normalization_mode_combo, 2, 0)
        residual_grid.addWidget(self.n_modes_spin, 2, 1)
        residual_grid.setColumnStretch(0, 1)
        residual_grid.setColumnStretch(1, 1)

        rounding_layout = QVBoxLayout()
        rounding_layout.setSpacing(6)
        rounding_layout.addWidget(self._section_label("Rounding"))
        rounding_layout.addWidget(self.round_radii_check)
        rounding_layout.addWidget(self.round_coeffs_check)
        rounding_layout.addWidget(self.zero_vertex_tilt_check)

        self.roc_mode_combo.currentIndexChanged.connect(self._on_roc_mode_changed)
        self.sphere_fit_mode_combo.currentIndexChanged.connect(self._on_sphere_fit_mode_changed)
        self.normalization_mode_combo.currentIndexChanged.connect(self.refresh_command_preview)
        self.fixed_roc_spin.valueChanged.connect(self.refresh_command_preview)
        self.center_weight_spin.valueChanged.connect(self.refresh_command_preview)
        self.n_modes_spin.valueChanged.connect(self.refresh_command_preview)
        self.round_radii_check.stateChanged.connect(self.refresh_command_preview)
        self.round_coeffs_check.stateChanged.connect(self.refresh_command_preview)
        self.zero_vertex_tilt_check.stateChanged.connect(self.refresh_command_preview)

        layout.addLayout(sphere_grid)
        layout.addLayout(residual_grid)
        layout.addLayout(rounding_layout)
        self._sync_sphere_reference_controls()
        return group

    def _build_runtime_group(self) -> QGroupBox:
        group = QGroupBox("Runtime Options", self)
        layout = QHBoxLayout(group)

        self.recursive_check = QCheckBox("Recursive search", self)
        self.qa_report_check = QCheckBox("Generate QA report", self)
        self.qa_report_check.setChecked(True)
        self.fail_fast_check = QCheckBox("Fail fast", self)
        self.h5_enabled_check = QCheckBox("Write HDF5", self)
        self.h5_enabled_check.setChecked(False)
        self.recursive_check.setToolTip("Search subdirectories under the input directory, not just the top level.")
        self.qa_report_check.setToolTip("Write an HTML gallery with thumbnail plots for quick visual inspection.")
        self.fail_fast_check.setToolTip("Stop the batch immediately on the first file error instead of continuing.")
        self.h5_enabled_check.setToolTip(
            "Append raw point-cloud data and fit results into an HDF5 file. Leave off for public/shareable runs."
        )

        for widget in (
            self.recursive_check,
            self.qa_report_check,
            self.fail_fast_check,
            self.h5_enabled_check,
        ):
            layout.addWidget(widget)
            widget.stateChanged.connect(self.refresh_command_preview)
        layout.addStretch(1)
        return group

    def _build_command_preview_group(self) -> QGroupBox:
        group = QGroupBox("Command Preview", self)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(14, 16, 14, 14)
        layout.setSpacing(8)

        self.command_preview = QPlainTextEdit(self)
        self.command_preview.setReadOnly(True)
        self.command_preview.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        self.command_preview.setMaximumBlockCount(200)
        self.command_preview.setMinimumHeight(140)

        layout.addWidget(self.command_preview)
        return group

    def _build_log_group(self) -> QGroupBox:
        group = QGroupBox("Live Output", self)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(14, 16, 14, 14)
        layout.setSpacing(8)

        self.log_output = QPlainTextEdit(self)
        self.log_output.setReadOnly(True)
        self.log_output.setLineWrapMode(QPlainTextEdit.NoWrap)

        layout.addWidget(self.log_output, stretch=1)
        return group

    def _build_runner_actions(self) -> QWidget:
        widget = QWidget(self)
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        self.refresh_run_name_button = QPushButton("New Run Name", self)
        self.run_button = QPushButton("Run", self)
        self.stop_button = QPushButton("Stop", self)
        self.stop_button.setEnabled(False)
        self.run_button.setDefault(True)

        for button in (self.refresh_run_name_button, self.run_button, self.stop_button):
            button.setMinimumHeight(36)

        self.refresh_run_name_button.clicked.connect(self._reset_run_name)
        self.run_button.clicked.connect(self.start_process)
        self.stop_button.clicked.connect(self.stop_process)

        layout.addWidget(self.refresh_run_name_button)
        layout.addStretch(1)
        layout.addWidget(self.run_button)
        layout.addWidget(self.stop_button)
        widget.setLayout(layout)
        return widget

    def _build_summary_loader_group(self) -> QGroupBox:
        group = QGroupBox("Summary Workbook", self)
        layout = QHBoxLayout(group)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(6)
        self.summary_file_edit = QLineEdit(self)
        self.summary_load_button = QPushButton("Load Summary", self)
        self.summary_load_button.clicked.connect(self.load_summary_workbook)
        layout.addWidget(QLabel("Batch summary(.xlsx)"), stretch=0)
        layout.addWidget(
            self._with_browse_button(
                self.summary_file_edit,
                directory=False,
                save=False,
                xlsx=True,
                on_selected=self.load_summary_workbook,
            ),
            stretch=1,
        )
        layout.addWidget(self.summary_load_button, stretch=0)
        return group

    def _build_selection_group(self) -> QGroupBox:
        group = QGroupBox("Selection", self)
        layout = QVBoxLayout(group)

        surf_widget = QWidget(self)
        surf_layout = QHBoxLayout(surf_widget)
        surf_layout.setContentsMargins(0, 0, 0, 0)
        surf_layout.addWidget(QLabel("Surface Family"))
        self.surf_id_buttons: dict[str, QPushButton] = {}
        for surf_id in FOCUS_SURF_IDS:
            button = QPushButton(surf_id, self)
            button.setCheckable(True)
            button.setAutoExclusive(True)
            button.setEnabled(False)
            button.clicked.connect(self._refresh_force_options)
            self.surf_id_buttons[surf_id] = button
            surf_layout.addWidget(button)
        surf_layout.addStretch(1)

        force_widget = QWidget(self)
        force_layout = QHBoxLayout(force_widget)
        force_layout.setContentsMargins(0, 0, 0, 0)
        force_layout.addWidget(QLabel("Force"))
        self.prev_force_button = QPushButton("Prev", self)
        self.prev_force_button.setMaximumWidth(60)
        self.force_id_combo = QComboBox(self)
        self.force_id_combo.currentTextChanged.connect(self._refresh_surface_token_options)
        self.next_force_button = QPushButton("Next", self)
        self.next_force_button.setMaximumWidth(60)
        self.prev_force_button.clicked.connect(lambda: self._navigate_force(-1))
        self.next_force_button.clicked.connect(lambda: self._navigate_force(1))
        force_layout.addWidget(self.prev_force_button)
        force_layout.addWidget(self.force_id_combo, stretch=1)
        force_layout.addWidget(self.next_force_button)

        token_widget = QWidget(self)
        token_layout = QHBoxLayout(token_widget)
        token_layout.setContentsMargins(0, 0, 0, 0)
        token_layout.addWidget(QLabel("Surface State"))
        self.surface_suffix_buttons: dict[str, QPushButton] = {}
        for suffix, label in (("I", "Initial"), ("D", "Deformed")):
            button = QPushButton(label, self)
            button.setCheckable(True)
            button.setAutoExclusive(True)
            button.setEnabled(False)
            button.clicked.connect(self.plot_current_selection)
            self.surface_suffix_buttons[suffix] = button
            token_layout.addWidget(button)
        token_layout.addStretch(1)

        layout.addWidget(surf_widget)
        layout.addWidget(force_widget)
        layout.addWidget(token_widget)
        return group

    def _build_summary_path_remap_group(self) -> QGroupBox:
        group = QGroupBox("Viewer Path Remapping", self)
        self.summary_path_remap_group = group
        layout = QHBoxLayout(group)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(6)

        self.summary_original_source_root_edit = QLineEdit(self)
        self.summary_local_source_root_edit = QLineEdit(self)

        self.summary_original_source_root_edit.setPlaceholderText(
            "Auto-filled from run_manifest.json or source_file paths"
        )
        self.summary_local_source_root_edit.setPlaceholderText("Choose the local folder containing the raw .xyz files")
        self.summary_original_source_root_edit.editingFinished.connect(self._refresh_current_summary_preview_if_loaded)
        self.summary_local_source_root_edit.editingFinished.connect(self._refresh_current_summary_preview_if_loaded)
        self.summary_original_source_root_edit.setToolTip(
            "Original raw-data root recorded by the saved run. Auto-filled when available."
        )
        self.summary_local_source_root_edit.setToolTip(
            "Local raw-data root used only when the saved workbook references .xyz files from another machine."
        )

        layout.addWidget(QLabel("Original XYZ Root"), stretch=0)
        layout.addWidget(self.summary_original_source_root_edit, stretch=1)
        layout.addWidget(QLabel("Local XYZ Root"), stretch=0)
        layout.addWidget(
            self._with_browse_button(
                self.summary_local_source_root_edit,
                directory=True,
                on_selected=self._refresh_current_summary_preview_if_loaded,
            ),
            stretch=1,
        )
        return group

    def _build_details_group(self) -> QGroupBox:
        group = QGroupBox("Selection Details", self)
        layout = QVBoxLayout(group)
        self.details_output = QTextEdit(self)
        self.details_output.setReadOnly(True)
        self.details_output.setLineWrapMode(QTextEdit.NoWrap)
        layout.addWidget(self.details_output)
        return group

    def _with_browse_button(
        self,
        line_edit: QLineEdit,
        *,
        directory: bool,
        save: bool = False,
        xlsx: bool = False,
        on_selected: Callable[[], None] | None = None,
    ) -> QWidget:
        widget = QWidget(self)
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.setStretch(0, 1)
        layout.setStretch(1, 0)
        button = QPushButton("Browse", widget)
        button.setFixedWidth(BROWSE_BUTTON_WIDTH)
        button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        line_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        def choose_path() -> None:
            if directory:
                selected = QFileDialog.getExistingDirectory(
                    self, "Select Directory", line_edit.text() or str(SCRIPT_DIR)
                )
            elif save:
                selected, _ = QFileDialog.getSaveFileName(
                    self,
                    "Select Output File",
                    line_edit.text() or str(SCRIPT_DIR / "results.h5"),
                    "HDF5 Files (*.h5);;All Files (*)",
                )
            else:
                selected, _ = QFileDialog.getOpenFileName(
                    self,
                    "Select File",
                    line_edit.text() or str(SCRIPT_DIR),
                    "Excel Files (*.xlsx)" if xlsx else "All Files (*)",
                )
            if selected:
                line_edit.setText(selected)
                if on_selected is not None:
                    on_selected()

        button.clicked.connect(choose_path)
        layout.addWidget(line_edit, stretch=1)
        layout.addWidget(button)
        return widget

    def _section_label(self, text: str) -> QLabel:
        label = QLabel(text, self)
        label.setProperty("themeRole", "sectionLabel")
        label.setStyleSheet(f"font-size: 12px; font-weight: 600; color: {self._section_text_color()};")
        return label

    def _default_run_name(self) -> str:
        return datetime.now().strftime("gui_run_%Y%m%d_%H%M%S")

    def _active_batch_run_dir(self) -> Path:
        output_root = Path(self.output_dir_edit.text().strip()).expanduser()
        return output_root / self.run_name_edit.text().strip()

    @staticmethod
    def _format_elapsed_seconds(elapsed_seconds: float) -> str:
        if elapsed_seconds < 60:
            return f"{elapsed_seconds:.1f} s"
        minutes, seconds = divmod(elapsed_seconds, 60.0)
        return f"{int(minutes)} min {seconds:.1f} s"

    def _reset_run_name(self) -> None:
        self.run_name_edit.setText(self._default_run_name())

    def _append_progress_line(self, current: int, total: int, source_file: str) -> None:
        """Append lightweight per-file progress text using just the source stem."""
        self.log_output.appendPlainText(f"processed ({current}/{total}): {Path(source_file).stem}")

    def _handle_process_output_line(self, line: str) -> None:
        """Normalize subprocess output into user-friendly live progress lines where possible."""
        stripped = line.rstrip("\n")
        if not stripped:
            return
        if stripped.startswith("processed: "):
            self._processed_progress_count += 1
            total = self._active_total_inputs or self._processed_progress_count
            self._append_progress_line(
                self._processed_progress_count,
                total,
                stripped.removeprefix("processed: ").strip(),
            )
            return
        self.log_output.appendPlainText(stripped)

    def _on_roc_mode_changed(self) -> None:
        self._sync_sphere_reference_controls()
        self.refresh_command_preview()

    def _on_sphere_fit_mode_changed(self) -> None:
        self._sync_sphere_reference_controls()
        self.refresh_command_preview()

    def _sync_sphere_reference_controls(self) -> None:
        sphere_fit_mode = str(self.sphere_fit_mode_combo.currentData())
        roc_mode = str(self.roc_mode_combo.currentData())
        legacy_mode = sphere_fit_mode == "legacy_lsq"
        if not legacy_mode and roc_mode != "fit-per-file":
            self.roc_mode_combo.blockSignals(True)
            self.roc_mode_combo.setCurrentIndex(self.roc_mode_combo.findData("fit-per-file"))
            self.roc_mode_combo.blockSignals(False)
            roc_mode = "fit-per-file"
        self.roc_mode_combo.setEnabled(legacy_mode)
        self.fixed_roc_spin.setEnabled(legacy_mode and roc_mode == "fixed")
        self.center_weight_spin.setEnabled(sphere_fit_mode == "center_weighted")

    def build_command(self) -> list[str]:
        args = ["-m", CLI_MODULE, self.input_dir_edit.text().strip()]
        if self.glob_edit.text().strip():
            args.extend(["--glob", self.glob_edit.text().strip()])
        if self.output_dir_edit.text().strip():
            args.extend(["--output-dir", self.output_dir_edit.text().strip()])
        if self.run_name_edit.text().strip():
            args.extend(["--run-name", self.run_name_edit.text().strip()])
        if self.recursive_check.isChecked():
            args.append("--recursive")

        roc_mode = str(self.roc_mode_combo.currentData())
        sphere_fit_mode = str(self.sphere_fit_mode_combo.currentData())
        normalization_mode = str(self.normalization_mode_combo.currentData())
        if roc_mode == "fixed":
            args.extend(["--fixed-roc-um", f"{self.fixed_roc_spin.value():.6f}"])
        else:
            args.extend(["--roc-mode", roc_mode])
        args.extend(["--sphere-fit-mode", sphere_fit_mode])
        args.extend(["--center-weight", f"{self.center_weight_spin.value():.2f}"])
        args.extend(["--normalization-mode", normalization_mode])
        args.extend(["--n-modes", str(self.n_modes_spin.value())])
        if not self.round_radii_check.isChecked():
            args.append("--no-round-radii-um")
        if not self.round_coeffs_check.isChecked():
            args.append("--no-round-zernike-coeffs")
        if self.zero_vertex_tilt_check.isChecked():
            args.append("--zero-vertex-tilt")
        if self.fail_fast_check.isChecked():
            args.append("--fail-fast")
        if self.qa_report_check.isChecked():
            args.append("--qa-report")
        if self.h5_enabled_check.isChecked() and self.h5_path_edit.text().strip():
            args.extend(["--h5-path", self.h5_path_edit.text().strip()])
        return args

    def refresh_command_preview(self) -> None:
        command = [sys.executable, *self.build_command()]
        self.command_preview.setPlainText(shlex.join(command))

    def validate_inputs(self) -> bool:
        input_dir = self.input_dir_edit.text().strip()
        output_dir = self.output_dir_edit.text().strip()
        if not input_dir:
            QMessageBox.warning(self, "Missing Input", "Select an input directory.")
            return False
        if not Path(input_dir).exists():
            QMessageBox.warning(self, "Invalid Input", f"Input directory does not exist:\n{input_dir}")
            return False
        if not output_dir:
            QMessageBox.warning(self, "Missing Output", "Select an output directory.")
            return False
        roc_mode = str(self.roc_mode_combo.currentData())
        sphere_fit_mode = str(self.sphere_fit_mode_combo.currentData())
        if roc_mode == "fixed" and self.fixed_roc_spin.value() <= 0:
            QMessageBox.warning(self, "Invalid Fixed ROC", "Fixed ROC must be greater than zero.")
            return False
        if not 0.0 <= self.center_weight_spin.value() <= 5.0:
            QMessageBox.warning(self, "Invalid Center Weight", "Center weight must be between 0 and 5.")
            return False
        try:
            validate_sphere_reference_configuration(
                roc_mode=roc_mode,
                sphere_fit_mode=sphere_fit_mode,
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid Sphere Reference", str(exc))
            return False
        return True

    def _effective_input_count_summary(self) -> tuple[int, int, int]:
        """Mirror batch discovery so the GUI warns on the real effective run size."""
        input_dir = Path(self.input_dir_edit.text().strip())
        matcher = input_dir.rglob if self.recursive_check.isChecked() else input_dir.glob
        matched_files = sorted(path for path in matcher(self.glob_edit.text().strip()) if path.is_file())
        focus_files = [path for path in matched_files if is_focus_surface_family(parse_surface_metadata(path).surf_id)]
        processing_inputs = collapse_identical_initial_inputs(focus_files)
        return len(processing_inputs), len(matched_files), len(focus_files)

    def _confirm_large_batch_run(self) -> bool:
        """Warn before launching a very large batch run."""
        try:
            effective_count, matched_count, focus_count = self._effective_input_count_summary()
        except ValueError as exc:
            QMessageBox.warning(self, "Batch Discovery Error", str(exc))
            return False
        if effective_count <= LARGE_BATCH_WARNING_THRESHOLD:
            return True
        response = QMessageBox.question(
            self,
            "Confirm Large Batch",
            (
                f"This run will process {effective_count} effective inputs.\n\n"
                f"Matched files: {matched_count}\n"
                f"Focus-family files: {focus_count}\n"
                f"Input directory: {self.input_dir_edit.text().strip()}\n"
                f"Glob: {self.glob_edit.text().strip()}\n\n"
                "Large runs may take substantial time and produce large outputs.\n"
                "Do you want to continue?"
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return response == QMessageBox.Yes

    def start_process(self) -> None:
        if self.process is not None:
            QMessageBox.information(self, "Already Running", "A batch process is already running.")
            return
        if not self.validate_inputs():
            return
        effective_count = None
        try:
            effective_count, _matched_count, _focus_count = self._effective_input_count_summary()
        except ValueError:
            effective_count = None
        if not self._confirm_large_batch_run():
            return

        self.log_output.clear()
        self.refresh_command_preview()
        self._active_run_dir = self._active_batch_run_dir()
        self._process_started_at = datetime.now()
        self._active_total_inputs = effective_count
        self._processed_progress_count = 0
        self._process_output_buffer = ""

        process = QProcess(self)
        process.setProgram(sys.executable)
        process.setArguments(self.build_command())
        process.setWorkingDirectory(str(SCRIPT_DIR))
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        process.setProcessEnvironment(env)
        process.setProcessChannelMode(QProcess.MergedChannels)
        process.readyReadStandardOutput.connect(self._consume_output)
        process.finished.connect(self._process_finished)
        process.started.connect(self._process_started)

        self.process = process
        process.start()

    def stop_process(self) -> None:
        if self.process is not None:
            self.log_output.appendPlainText("\nStopping process...\n")
            self.process.kill()

    def _process_started(self) -> None:
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.log_output.appendPlainText(f"$ {self.command_preview.toPlainText()}\n")

    def _consume_output(self) -> None:
        if self.process is None:
            return
        chunk = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if chunk:
            self._process_output_buffer += chunk
            while "\n" in self._process_output_buffer:
                line, self._process_output_buffer = self._process_output_buffer.split("\n", 1)
                self._handle_process_output_line(line)

    def _process_finished(self, exit_code: int, _exit_status: QProcess.ExitStatus) -> None:
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if self._process_output_buffer:
            self._handle_process_output_line(self._process_output_buffer)
            self._process_output_buffer = ""
        elapsed_seconds = (
            (datetime.now() - self._process_started_at).total_seconds()
            if self._process_started_at is not None
            else None
        )
        if exit_code == 0:
            processed_files = None
            failed_files = None
            run_dir = self._active_run_dir
            manifest_path = run_dir / "run_manifest.json" if run_dir is not None else None
            summary_report_path: Path | None = None
            if manifest_path is not None and manifest_path.exists():
                try:
                    manifest = json.loads(manifest_path.read_text())
                except json.JSONDecodeError:
                    manifest = {}
                processed_files = manifest.get("processed_files")
                failed_files = manifest.get("failed_files")
                summary_report_text = manifest.get("summary_report_path")
                if summary_report_text:
                    summary_report_path = Path(str(summary_report_text))
            if summary_report_path is not None:
                self.summary_file_edit.setText(str(summary_report_path))

            if isinstance(processed_files, int) and isinstance(failed_files, int):
                success_line = (
                    f"\nCompleted successfully: processed {processed_files} file{'' if processed_files == 1 else 's'}"
                )
                if failed_files:
                    success_line += f", {failed_files} failed"
                if elapsed_seconds is not None:
                    success_line += f" in {self._format_elapsed_seconds(elapsed_seconds)}"
                if run_dir is not None:
                    success_line += f".\nResults: {run_dir}\n"
                else:
                    success_line += ".\n"
            else:
                success_line = "\nCompleted successfully."
                if elapsed_seconds is not None:
                    success_line += f" Total time: {self._format_elapsed_seconds(elapsed_seconds)}."
                if run_dir is not None:
                    success_line += f"\nResults: {run_dir}\n"
                else:
                    success_line += "\n"
            self.log_output.appendPlainText(success_line)
            if summary_report_path is not None:
                self.log_output.appendPlainText(f"Summary workbook: {summary_report_path}")
            if manifest_path is not None and manifest_path.exists():
                self.log_output.appendPlainText(f"Run manifest: {manifest_path}")
        else:
            self.log_output.appendPlainText(f"\nProcess exited with code {exit_code}.\n")
        self.process = None
        self._process_started_at = None
        self._active_run_dir = None
        self._active_total_inputs = None
        self._processed_progress_count = 0

    def _refresh_current_summary_preview_if_loaded(self) -> None:
        if self.summary_rows:
            self.plot_current_selection()

    def _update_summary_path_remap_hint(self) -> None:
        if self.summary_workbook_path is None:
            tooltip = (
                "Coefficient paths under the original run folder are remapped automatically to the loaded workbook folder. "
                "Set Local XYZ Root only when the raw .xyz tree moved to a different absolute path."
            )
            self.summary_path_remap_group.setToolTip(tooltip)
            self.summary_original_source_root_edit.setToolTip(
                "Original raw-data root recorded by the saved run. Auto-filled when available."
            )
            self.summary_local_source_root_edit.setToolTip(
                "Local raw-data root used only when the saved workbook references .xyz files from another machine."
            )
            return

        original_run_dir = infer_original_run_dir(self.summary_workbook_path, self.summary_manifest)
        current_run_dir = self.summary_workbook_path.parent
        original_xyz_root = self.summary_original_source_root_edit.text().strip() or "n/a"
        if original_run_dir is None:
            coeff_line = f"Results folder remap: current workbook folder = {current_run_dir}"
        else:
            coeff_line = f"Results folder remap: {original_run_dir} -> {current_run_dir}"
        group_tooltip = (
            f"{coeff_line}\n"
            f"Original XYZ root: {original_xyz_root}\n"
            "Set Local XYZ Root only if the saved workbook references raw data on another machine."
        )
        self.summary_path_remap_group.setToolTip(group_tooltip)
        self.summary_original_source_root_edit.setToolTip(
            f"Original XYZ root inferred from the saved run.\nCurrent value: {original_xyz_root}"
        )
        self.summary_local_source_root_edit.setToolTip(
            f"{coeff_line}\nSet this only when the raw .xyz files live elsewhere on this machine."
        )

    def load_summary_workbook(self) -> None:
        path_text = self.summary_file_edit.text().strip()
        if not path_text:
            QMessageBox.warning(self, "Missing Summary", "Choose a batch summary workbook.")
            return
        path = Path(path_text)
        if not path.exists():
            QMessageBox.warning(self, "Missing Summary", f"Summary workbook not found:\n{path}")
            return

        rows = parse_inline_xlsx_rows(path)
        rows = [row for row in rows if row.get("surf_id") in FOCUS_SURF_IDS]
        if not rows:
            QMessageBox.warning(self, "No Rows", "No AA/AP/PA/PP rows were found in the workbook.")
            return

        manifest = parse_run_manifest(path)
        if is_compact_summary_rows(rows) and not manifest:
            QMessageBox.warning(
                self,
                "Missing Run Manifest",
                "This compact batch summary requires the sibling run_manifest.json for replay.\n\n"
                f"Expected next to:\n{path}",
            )
            return

        self.summary_rows = rows
        self.summary_manifest = manifest
        self.summary_workbook_path = path
        inferred_source_root = infer_original_source_root(rows, self.summary_manifest)
        self.summary_original_source_root_edit.setText(str(inferred_source_root or ""))
        self._update_summary_path_remap_hint()
        observed_radii = [
            parse_optional_float(row.get("observed_aperture_radius_um", ""))
            for row in rows
            if row.get("observed_aperture_radius_um", "").strip()
        ]
        max_observed_radius = max((value for value in observed_radii if value is not None), default=None)
        self.common_rho_axis_limit_um = None
        if max_observed_radius is not None:
            self.common_rho_axis_limit_um = float(int(np.ceil(max_observed_radius / 100.0) * 100.0))
        self._refresh_surf_options()
        self.details_output.setPlainText(
            f"Loaded {len(rows)} focus rows from:\n{path}\n\nRun manifest keys:\n"
            + "\n".join(sorted(str(key) for key in self.summary_manifest))
        )
        if self.current_selected_row() is None:
            self.preview_canvas.clear_message(
                "Summary loaded, but no matching AA/AP/PA/PP row is currently selectable."
            )
        else:
            self.plot_current_selection()

    def _summary_display_run_name(self) -> str:
        if self.summary_manifest.get("run_name"):
            return str(self.summary_manifest["run_name"])
        if self.summary_workbook_path is not None:
            return self.summary_workbook_path.stem
        return "summary_preview"

    @staticmethod
    def _has_actual_summary_remap(strategy: str) -> bool:
        return strategy not in {"", "workbook", "summary-relative"}

    @staticmethod
    def _split_preview_detail_text(text: str) -> tuple[dict[str, str], list[str]]:
        fields: dict[str, str] = {}
        top_coeff_lines: list[str] = []
        reading_top_coeffs = False
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line == "Top coeffs:":
                reading_top_coeffs = True
                continue
            if reading_top_coeffs:
                top_coeff_lines.append(raw_line.strip())
                continue
            if ": " in line:
                label, value = line.split(": ", maxsplit=1)
                fields[label] = value
        return fields, top_coeff_lines

    def _set_details_output_lines(self, lines: list[str]) -> None:
        bold_lines = {"Diagnostics and Locations", "Tilt Removal"}
        html_lines: list[str] = []
        for line in lines:
            rendered = escape(line)
            if line in bold_lines:
                rendered = f"<strong>{rendered}</strong>"
            html_lines.append(rendered)
        self.details_output.setHtml(
            "<div style='font-family: monospace; white-space: pre;'>" + "<br>".join(html_lines) + "</div>"
        )

    def _summary_folder_name(self) -> str:
        if self.summary_workbook_path is not None:
            return self.summary_workbook_path.parent.name
        return self._summary_display_run_name()

    def _summary_uses_center_weight(self, row: dict[str, str], fields: dict[str, str]) -> bool:
        mode = str(row.get("sphere_fit_mode") or self.summary_manifest.get("sphere_fit_mode") or "").strip().lower()
        mode = mode.replace("-", "_").replace(" ", "_")
        if mode:
            return mode == "center_weighted"
        return fields.get("Sphere fit mode", "").strip().lower().replace(" ", "_") == "center_weighted"

    def _summary_workbook_uses_zero_vertex_tilt(self, row: dict[str, str]) -> bool:
        return parse_boolish(self.summary_manifest.get("zero_vertex_tilt"), default=False) or parse_boolish(
            row.get("vertex_tilt_correction"), default=False
        )

    @staticmethod
    def _format_row_float(row: dict[str, str], key: str, *, precision: int) -> str | None:
        value = parse_optional_float(row.get(key, ""))
        if value is None:
            return None
        return f"{value:.{precision}f}"

    def _append_tilt_removal_detail_lines(self, detail_lines: list[str], row: dict[str, str]) -> None:
        detail_lines.extend(["", "Tilt Removal", "Vertex tilt correction: on"])
        fields = [
            ("original_center_slope_x_mrad", "Original center slope x (mrad)", 6),
            ("original_center_slope_y_mrad", "Original center slope y (mrad)", 6),
            ("original_center_slope_magnitude_mrad", "Original center slope magnitude (mrad)", 6),
            ("corrected_center_slope_x_mrad", "Corrected center slope x (mrad)", 6),
            ("corrected_center_slope_y_mrad", "Corrected center slope y (mrad)", 6),
            ("corrected_center_slope_magnitude_mrad", "Corrected center slope magnitude (mrad)", 6),
            ("delta_z2_um", "Applied Z2 correction (um)", 9),
            ("delta_z3_um", "Applied Z3 correction (um)", 9),
        ]
        appended = False
        for key, label, precision in fields:
            formatted = self._format_row_float(row, key, precision=precision)
            if formatted is None:
                continue
            detail_lines.append(f"{label}: {formatted}")
            appended = True
        if not appended:
            detail_lines.append("Per-file correction details: not stored in this workbook.")

    def _summary_selection_detail_lines(
        self,
        *,
        row: dict[str, str],
        resolution_details: dict[str, str],
        preview_text: str,
        preview_details: dict[str, str],
    ) -> list[str]:
        fields, top_coeff_lines = self._split_preview_detail_text(preview_text)
        source_remapped = self._has_actual_summary_remap(resolution_details["source_resolution_strategy"])
        coeff_remapped = self._has_actual_summary_remap(resolution_details["coeff_resolution_strategy"])

        detail_lines = [f"Folder: {self._summary_folder_name()}"]

        for label in (
            "File",
            "Surface",
            "Force",
            "Sphere fit mode",
            "Center weight",
            "Norm mode",
            "Fitted sphere radius",
            "Applied norm radius",
            "Observed aperture radius",
        ):
            if label in fields:
                if label == "Center weight" and not self._summary_uses_center_weight(row, fields):
                    continue
                detail_lines.append(f"{label}: {fields[label]}")

        for label in ("Sphere center", "Target vertex", "Reference vertex", "Vertex mismatch z"):
            if label in fields:
                detail_lines.append(f"{label}: {fields[label]}")

        if top_coeff_lines:
            detail_lines.extend(["Top Zernike coefficients:", *top_coeff_lines])

        detail_lines.extend(
            [
                "",
                "Diagnostics and Locations",
                f"Sphere RMS (um): {preview_details['sphere_rms_um']}",
                f"ZPBS residual RMS (um): {preview_details['zpbs_residual_rms_um']}",
                f"ZPBS residual cond: {preview_details['zpbs_residual_cond']}",
            ]
        )
        if "Coeff metadata radius" in fields:
            detail_lines.append(f"Coefficient metadata radius: {fields['Coeff metadata radius']}")
        if source_remapped:
            detail_lines.extend(
                [
                    f"Original source file: {resolution_details['original_source_file']}",
                    f"Resolved source file: {resolution_details['resolved_source_file']}",
                ]
            )
        if coeff_remapped:
            detail_lines.extend(
                [
                    f"Coefficient remap: {resolution_details['coeff_resolution_strategy']} | exists={resolution_details['coeff_exists']}",
                    f"Original coefficient file: {resolution_details['original_coeff_file']}",
                ]
            )
        if coeff_remapped:
            detail_lines.append(f"Coefficient file: {preview_details['coeff_file']}")
        if self._summary_workbook_uses_zero_vertex_tilt(row):
            self._append_tilt_removal_detail_lines(detail_lines, row)
        return detail_lines

    def _refresh_surf_options(self) -> None:
        available = {row["surf_id"] for row in self.summary_rows}
        current = self._selected_surf_id()
        target = (
            current
            if current in available
            else next((surf_id for surf_id in FOCUS_SURF_IDS if surf_id in available), None)
        )
        for surf_id, button in self.surf_id_buttons.items():
            button.blockSignals(True)
            button.setEnabled(surf_id in available)
            button.setChecked(surf_id == target)
            button.blockSignals(False)
        self._refresh_force_options()

    def _refresh_force_options(self) -> None:
        surf_id = self._selected_surf_id()
        matching = [row for row in self.summary_rows if row["surf_id"] == surf_id]
        force_ids = sorted({row["force_id"] for row in matching}, key=self._force_sort_key)
        current = self.force_id_combo.currentText()
        self.force_id_combo.blockSignals(True)
        self.force_id_combo.clear()
        self.force_id_combo.addItems(force_ids)
        if current in force_ids:
            self.force_id_combo.setCurrentText(current)
        elif force_ids:
            self.force_id_combo.setCurrentIndex(0)
        self.force_id_combo.blockSignals(False)
        has_force_selection = bool(force_ids)
        self.prev_force_button.setEnabled(has_force_selection)
        self.next_force_button.setEnabled(has_force_selection)
        self._refresh_surface_token_options()

    def _refresh_surface_token_options(self) -> None:
        surf_id = self._selected_surf_id()
        force_id = self.force_id_combo.currentText()
        matching = [row for row in self.summary_rows if row["surf_id"] == surf_id and row["force_id"] == force_id]
        available_suffixes = {row["surface_token"][-1].upper() for row in matching if row.get("surface_token")}
        current = self._selected_surface_suffix()
        target = (
            current
            if current in available_suffixes
            else next((suffix for suffix in ("I", "D") if suffix in available_suffixes), None)
        )
        for suffix, button in self.surface_suffix_buttons.items():
            button.blockSignals(True)
            button.setEnabled(suffix in available_suffixes)
            button.setChecked(suffix == target)
            button.blockSignals(False)
        self._update_force_navigation_buttons()
        self.plot_current_selection()

    def current_selected_row(self) -> dict[str, str] | None:
        surf_id = self._selected_surf_id()
        force_id = self.force_id_combo.currentText()
        surface_suffix = self._selected_surface_suffix()
        if not surf_id or not force_id or not surface_suffix:
            return None
        surface_token = f"{surf_id}{surface_suffix}"
        for row in self.summary_rows:
            if row["surf_id"] == surf_id and row["force_id"] == force_id and row["surface_token"] == surface_token:
                return row
        return None

    def plot_current_selection(self) -> None:
        row = self.current_selected_row()
        if row is None:
            self.preview_canvas.clear_message("Select a surface family, force, and state to preview a fit.")
            self.details_output.setPlainText("No matching summary row is currently selected.")
            return

        preview_row, resolution_details = prepare_summary_row_for_preview(
            row,
            summary_file=self.summary_workbook_path or Path(""),
            rows=self.summary_rows,
            manifest=self.summary_manifest,
            original_source_root_text=self.summary_original_source_root_edit.text(),
            local_source_root_text=self.summary_local_source_root_edit.text(),
        )
        rho_limit = self.common_rho_axis_limit_um
        try:
            text, details = self.preview_canvas.plot_selection(
                preview_row,
                self.summary_manifest,
                rho_axis_limit_um=rho_limit,
            )
        except ValueError as exc:
            self.preview_canvas.clear_message(str(exc))
            self.details_output.setPlainText(str(exc))
            return

        detail_lines = self._summary_selection_detail_lines(
            row=row,
            resolution_details=resolution_details,
            preview_text=text,
            preview_details=details,
        )
        self._set_details_output_lines(detail_lines)

    def _selected_surf_id(self) -> str:
        for surf_id, button in self.surf_id_buttons.items():
            if button.isChecked():
                return surf_id
        return ""

    def _selected_surface_suffix(self) -> str:
        for suffix, button in self.surface_suffix_buttons.items():
            if button.isChecked():
                return suffix
        return ""

    def _navigate_force(self, step: int) -> None:
        if self.force_id_combo.count() == 0:
            return
        current_index = self.force_id_combo.currentIndex()
        if current_index < 0:
            current_index = 0
        new_index = min(max(current_index + step, 0), self.force_id_combo.count() - 1)
        if new_index != current_index:
            self.force_id_combo.setCurrentIndex(new_index)
        self._update_force_navigation_buttons()

    def _update_force_navigation_buttons(self) -> None:
        count = self.force_id_combo.count()
        current_index = self.force_id_combo.currentIndex()
        has_items = count > 0 and current_index >= 0
        self.prev_force_button.setEnabled(has_items and current_index > 0)
        self.next_force_button.setEnabled(has_items and current_index < count - 1)

    @staticmethod
    def _force_sort_key(force_id: str) -> tuple[float, str]:
        if force_id.startswith("F") and force_id.endswith("mN"):
            try:
                return (float(force_id[1:-2]), force_id)
            except ValueError:
                pass
        return (float("inf"), force_id)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._single_file_thread is not None and self._single_file_thread.isRunning():
            QMessageBox.warning(
                self,
                "Single-File Analysis Running",
                "Wait for the current single-file analysis to finish before closing the application.",
            )
            event.ignore()
            return
        shutil.rmtree(self.single_file_temp_root, ignore_errors=True)
        super().closeEvent(event)
