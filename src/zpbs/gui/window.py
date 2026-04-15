from __future__ import annotations

import os
import shlex
import shutil
import sys
import tempfile
from datetime import datetime
from collections.abc import Callable
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str((Path(__file__).resolve().parents[3] / ".matplotlib").resolve()))

from PyQt5.QtCore import QEvent, QProcess, QThread, QTimer, Qt
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
    QVBoxLayout,
    QWidget,
)

from ..common import is_focus_surface_family, validate_sphere_reference_configuration
from ..io.xyz import collapse_identical_initial_inputs, parse_surface_metadata
from .canvases import FitPreviewCanvas, NumericTableWidgetItem, OverviewPlotCanvas, SubsetPlotCanvas
from .single_file import (
    SingleFileAnalysisRequest,
    SingleFileAnalysisResult,
    SingleFileAnalysisWorker,
    build_single_file_candidates,
    validate_single_file_source,
)
from .support import (
    NORMALIZATION_MODE_LABELS,
    ROC_MODE_LABELS,
    SPHERE_FIT_MODE_LABELS,
    SUBSET_KIND_LABELS,
    display_label,
    format_metric,
    parse_optional_float,
)
from ..io.remap import infer_original_run_dir, infer_original_source_root, prepare_summary_row_for_preview
from ..io.workbook import detect_subset_workbook_kind, parse_inline_xlsx_rows, parse_run_manifest

SCRIPT_DIR = Path(__file__).resolve().parents[3]
BATCH_SCRIPT = SCRIPT_DIR / "batch_fit_xyz.py"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "batch_outputs"
FOCUS_SURF_IDS = ("AA", "AP", "PA", "PP")
PATH_FIELD_MIN_WIDTH = 720
FIT_CONTROL_WIDTH = 250
BROWSE_BUTTON_WIDTH = 92
LARGE_BATCH_WARNING_THRESHOLD = 1000


class BatchFitWindow(QMainWindow):
    """Launcher, summary viewer, and subset omission inspector."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Batch Fit Launcher and Viewer")
        self.resize(1380, 940)
        self.process: QProcess | None = None
        self.summary_rows: list[dict[str, str]] = []
        self.summary_manifest: dict[str, object] = {}
        self.summary_workbook_path: Path | None = None
        self.common_rho_axis_limit_um: float | None = None
        self.subset_rows: list[dict[str, str]] = []
        self.subset_manifest: dict[str, object] = {}
        self.subset_workbook_path: Path | None = None
        self.subset_workbook_kind = ""
        self.subset_workbook_specs: list[dict[str, str]] = []
        self.subset_spec_paths: dict[str, Path] = {}
        self._subset_updating_spec_combo = False
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
        # Public release guard: subset inspection stays private.
        if False:
            tabs.addTab(self._build_subset_tab(), "Inspect Subsets")

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

        left.setMinimumWidth(650)
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
        left_layout.addWidget(self._build_single_file_status_group())
        left_layout.addWidget(self._build_single_file_coeff_table_group(), stretch=1)
        left_layout.addWidget(self._build_single_file_diagnostics_group(), stretch=1)
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
        splitter.setSizes([470, 930])

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
        self.single_file_round_radii_check.setChecked(True)
        self.single_file_round_coeffs_check.setChecked(True)

        for widget in (
            self.single_file_path_edit,
            self.single_file_folder_edit,
        ):
            widget.setMinimumWidth(PATH_FIELD_MIN_WIDTH)

        for control in (
            self.single_file_sphere_fit_mode_combo,
            self.single_file_center_weight_spin,
            self.single_file_n_modes_spin,
        ):
            control.setFixedWidth(190)

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
        selector_layout.addWidget(self.single_file_force_combo, stretch=1)
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
        options_layout.addStretch(1)

        layout.addWidget(file_row)
        layout.addWidget(selector_row)
        layout.addWidget(options_row)

        self.single_file_path_edit.editingFinished.connect(lambda: self._on_single_file_path_committed(show_warning=True))
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

        self._sync_single_file_controls()
        return group

    def _build_single_file_status_group(self) -> QGroupBox:
        group = QGroupBox("Actions", self)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(14, 16, 14, 14)
        layout.setSpacing(10)

        self.single_file_status_label = QLabel("Select a source file or preload a folder.")
        self.single_file_status_label.setWordWrap(True)
        self.single_file_status_label.setProperty("themeRole", "mutedText")
        self.single_file_status_label.setStyleSheet(f"color: {self._muted_text_color()};")

        button_row = QWidget(self)
        button_layout = QHBoxLayout(button_row)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(8)
        self.single_file_save_csv_button = QPushButton("Save CSV...", self)
        self.single_file_save_overview_button = QPushButton("Save Overview Plot...", self)
        self.single_file_save_bundle_button = QPushButton("Save Bundle...", self)
        self.single_file_save_csv_button.clicked.connect(self.save_single_file_csv)
        self.single_file_save_overview_button.clicked.connect(self.save_single_file_overview_plot)
        self.single_file_save_bundle_button.clicked.connect(self.save_single_file_bundle)
        for button in (
            self.single_file_save_csv_button,
            self.single_file_save_overview_button,
            self.single_file_save_bundle_button,
        ):
            button_layout.addWidget(button)
        button_layout.addStretch(1)

        layout.addWidget(self.single_file_status_label)
        layout.addWidget(button_row)
        return group

    def _build_single_file_coeff_table_group(self) -> QGroupBox:
        group = QGroupBox("Exported Coefficient CSV", self)
        layout = QVBoxLayout(group)
        self.single_file_coeff_table = QTableWidget(self)
        self.single_file_coeff_table.setColumnCount(2)
        self.single_file_coeff_table.setHorizontalHeaderLabels(["Field", "Value"])
        self.single_file_coeff_table.verticalHeader().setVisible(False)
        self.single_file_coeff_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.single_file_coeff_table.setSelectionMode(QTableWidget.NoSelection)
        self.single_file_coeff_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.single_file_coeff_table)
        return group

    def _build_single_file_diagnostics_group(self) -> QGroupBox:
        group = QGroupBox("Extended Diagnostics", self)
        layout = QVBoxLayout(group)
        self.single_file_diagnostics_table = QTableWidget(self)
        self.single_file_diagnostics_table.setColumnCount(2)
        self.single_file_diagnostics_table.setHorizontalHeaderLabels(["Field", "Value"])
        self.single_file_diagnostics_table.verticalHeader().setVisible(False)
        self.single_file_diagnostics_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.single_file_diagnostics_table.setSelectionMode(QTableWidget.NoSelection)
        self.single_file_diagnostics_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.single_file_diagnostics_table)
        return group

    def _set_single_file_status(self, text: str) -> None:
        self.single_file_status_label.setText(text)

    def _clear_single_file_result_views(self, message: str) -> None:
        self.single_file_result = None
        self.single_file_overview_canvas.clear_message(message)
        self._populate_name_value_table(self.single_file_coeff_table, [])
        self._populate_name_value_table(self.single_file_diagnostics_table, [])
        self._refresh_single_file_export_buttons()

    def _populate_name_value_table(self, table: QTableWidget, rows: list[tuple[str, str]]) -> None:
        table.setRowCount(len(rows))
        for row_index, (name, value) in enumerate(rows):
            table.setItem(row_index, 0, NumericTableWidgetItem(name, name))
            table.setItem(row_index, 1, NumericTableWidgetItem(value, value))
        table.resizeColumnsToContents()

    def _diagnostic_rows_for_result(self, result: SingleFileAnalysisResult) -> list[tuple[str, str]]:
        artifacts = result.artifacts
        return [
            ("Source file", str(artifacts.source_file)),
            ("Temp result directory", str(result.result_dir)),
            ("Points used", str(artifacts.points_used)),
            ("Method", artifacts.method),
            ("Sphere fit mode", artifacts.sphere_fit_mode),
            ("Center weight", f"{artifacts.center_weight:.2f}"),
            ("Normalization mode", artifacts.normalization_mode),
            ("Round radii", "on" if artifacts.round_radii_um else "off"),
            ("Round coefficients", str(artifacts.zernike_coeff_sigfigs or "off")),
            ("Applied reference radius (um)", format_metric(artifacts.applied_reference_radius_um, precision=4)),
            ("Observed aperture radius (um)", format_metric(artifacts.observed_aperture_radius_um, precision=4)),
            ("Normalization radius (um)", format_metric(artifacts.norm_radius_um, precision=4)),
            ("Sphere SSE", format_metric(artifacts.sphere_sse, precision=3)),
            ("Sphere RMS", format_metric(artifacts.sphere_rms, precision=3)),
            ("Surface RMS", format_metric(artifacts.surface_zernike_rms, precision=3)),
            ("Residual RMS", format_metric(artifacts.sphere_residual_zernike_rms, precision=3)),
            ("Vertex (um)", format_metric(artifacts.vertex_um, precision=6)),
            ("Vertex residual (um)", format_metric(artifacts.vertex_residual_um, precision=6)),
            ("Target vertex z (um)", format_metric(artifacts.target_vertex_z_um, precision=4)),
            ("Reference vertex z (um)", format_metric(artifacts.reference_vertex_z_um, precision=4)),
            ("Vertex mismatch z (um)", format_metric(artifacts.vertex_mismatch_z_um, precision=3)),
            ("Coefficient CSV", str(result.csv_path)),
            ("Overview plot", str(result.overview_plot_path)),
            ("Analysis JSON", str(result.analysis_json_path)),
        ]

    def _sync_single_file_controls(self) -> None:
        sphere_fit_mode = str(self.single_file_sphere_fit_mode_combo.currentData())
        self.single_file_center_weight_spin.setEnabled(sphere_fit_mode == "center_weighted")
        self._update_single_file_force_navigation_buttons()

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
        self.single_file_overview_canvas.plot_artifacts(result.artifacts)
        self._populate_name_value_table(self.single_file_coeff_table, result.coeff_rows)
        self._populate_name_value_table(self.single_file_diagnostics_table, self._diagnostic_rows_for_result(result))
        status_prefix = "Loaded cached analysis" if cached_result else "Analysis ready"
        self._set_single_file_status(
            f"{status_prefix}: {result.artifacts.source_file.name} | temp: {result.result_dir}"
        )
        self._refresh_single_file_export_buttons()

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
        surf_id = self._single_file_selected_surf_id()
        force_id = self.single_file_force_combo.currentText()
        suffix = self._single_file_selected_state_suffix()
        if not surf_id or not force_id or not suffix:
            return None
        for path in self.single_file_candidates:
            metadata = self._single_file_metadata(path)
            if metadata.surf_id == surf_id and metadata.force_id == force_id and metadata.surface_token.endswith(suffix):
                return path
        return None

    def _set_single_file_candidates(self, candidates: list[Path], preferred: Path | None = None) -> None:
        self.single_file_candidates = list(candidates)
        self._refresh_single_file_surf_options(preferred=preferred)

    def _refresh_single_file_surf_options(self, preferred: Path | None = None) -> None:
        available = {self._single_file_metadata(path).surf_id for path in self.single_file_candidates}
        preferred_surf_id = self._single_file_metadata(preferred).surf_id if preferred is not None and preferred.exists() else ""
        current = self._single_file_selected_surf_id()
        target = current if current in available else preferred_surf_id if preferred_surf_id in available else next(
            (surf_id for surf_id in FOCUS_SURF_IDS if surf_id in available),
            None,
        )
        for surf_id, button in self.single_file_surf_buttons.items():
            button.blockSignals(True)
            button.setEnabled(surf_id in available)
            button.setChecked(surf_id == target)
            button.blockSignals(False)
        self._refresh_single_file_force_options(preferred=preferred)

    def _refresh_single_file_force_options(self, preferred: Path | None = None) -> None:
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
            if self._single_file_metadata(path).surf_id == surf_id and self._single_file_metadata(path).force_id == force_id
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
        target = current if current in available_suffixes else preferred_suffix if preferred_suffix in available_suffixes else next(
            (suffix for suffix in ("I", "D") if suffix in available_suffixes),
            None,
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
            selected_path = preferred if preferred is not None and preferred.resolve() in {path.resolve() for path in candidates} else candidates[0]
            self._apply_single_file_source(selected_path, queue_analysis=True)
            self._set_single_file_status(f"Loaded {len(candidates)} maintained files from {folder}.")
        else:
            self._set_single_file_status(f"No maintained AA/AP/PA/PP files matched *_FVS_*.xyz in {folder}.")
            self._clear_single_file_result_views("No maintained AA/AP/PA/PP files were found in the selected folder.")

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

    def _copy_single_file_artifact(self, source_path: Path, *, caption: str, default_name: str, file_filter: str) -> None:
        selected, _ = QFileDialog.getSaveFileName(self, caption, str(source_path.with_name(default_name)), file_filter)
        if not selected:
            return
        shutil.copy2(source_path, selected)

    def save_single_file_csv(self) -> None:
        if self.single_file_result is None:
            return
        self._copy_single_file_artifact(
            self.single_file_result.csv_path,
            caption="Save Coefficient CSV",
            default_name=self.single_file_result.csv_path.name,
            file_filter="CSV Files (*.csv);;All Files (*)",
        )

    def save_single_file_overview_plot(self) -> None:
        if self.single_file_result is None:
            return
        self._copy_single_file_artifact(
            self.single_file_result.overview_plot_path,
            caption="Save Overview Plot",
            default_name=self.single_file_result.overview_plot_path.name,
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

    def _build_subset_tab(self) -> QWidget:
        tab = QWidget(self)
        layout = QVBoxLayout(tab)
        layout.addWidget(self._build_subset_loader_group())

        splitter = QSplitter(self)
        splitter.setChildrenCollapsible(False)

        left = QWidget(self)
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(self._build_subset_manifest_group())
        left_layout.addWidget(self._build_subset_selection_group())
        left_layout.addWidget(self._build_subset_table_group(), stretch=1)
        left_layout.addWidget(self._build_subset_details_group(), stretch=1)
        splitter.addWidget(left)

        right = QWidget(self)
        right_layout = QVBoxLayout(right)
        self.subset_canvas = SubsetPlotCanvas(right)
        self.subset_canvas.clear_message("Load a subset workbook emitted by zernike_subset_harness.py.")
        right_layout.addWidget(self.subset_canvas, stretch=1)
        splitter.addWidget(right)
        splitter.setSizes([470, 870])

        layout.addWidget(splitter, stretch=1)
        return tab

    def _build_header(self) -> QWidget:
        widget = QWidget(self)
        layout = QVBoxLayout(widget)
        title = QLabel("Batch XYZ Fitter")
        title.setStyleSheet("font-size: 20px; font-weight: 600;")
        subtitle = QLabel(
            "Wrapper for batch_fit_xyz.py. The runner only includes AA/AP/PA/PP files, and the viewer can load a summary workbook to inspect individual fits live."
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
        self.h5_path_edit = QLineEdit("ardea_real_data_validation.h5", self)
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
        self.round_radii_check.setChecked(True)
        self.round_coeffs_check.setChecked(True)
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

        self.roc_mode_combo.currentIndexChanged.connect(self._on_roc_mode_changed)
        self.sphere_fit_mode_combo.currentIndexChanged.connect(self._on_sphere_fit_mode_changed)
        self.normalization_mode_combo.currentIndexChanged.connect(self.refresh_command_preview)
        self.fixed_roc_spin.valueChanged.connect(self.refresh_command_preview)
        self.center_weight_spin.valueChanged.connect(self.refresh_command_preview)
        self.n_modes_spin.valueChanged.connect(self.refresh_command_preview)
        self.round_radii_check.stateChanged.connect(self.refresh_command_preview)
        self.round_coeffs_check.stateChanged.connect(self.refresh_command_preview)

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
        self.h5_enabled_check.setChecked(True)
        self.recursive_check.setToolTip("Search subdirectories under the input directory, not just the top level.")
        self.qa_report_check.setToolTip("Write an HTML gallery with thumbnail plots for quick visual inspection.")
        self.fail_fast_check.setToolTip("Stop the batch immediately on the first file error instead of continuing.")
        self.h5_enabled_check.setToolTip("Append the batch results into the shared HDF5 file for later analysis.")

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

        self.summary_original_source_root_edit.setPlaceholderText("Auto-filled from run_manifest.json or source_file paths")
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
        self.details_output = QPlainTextEdit(self)
        self.details_output.setReadOnly(True)
        self.details_output.setLineWrapMode(QPlainTextEdit.NoWrap)
        layout.addWidget(self.details_output)
        return group

    def _build_subset_loader_group(self) -> QGroupBox:
        group = QGroupBox("Subset Workbook", self)
        layout = QHBoxLayout(group)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(6)
        self.subset_file_edit = QLineEdit(self)
        self.subset_load_button = QPushButton("Load Workbook", self)
        self.subset_load_button.clicked.connect(self.load_subset_workbook)
        layout.addWidget(QLabel("Subset workbook(.xlsx)"), stretch=0)
        layout.addWidget(
            self._with_browse_button(
                self.subset_file_edit,
                directory=False,
                save=False,
                xlsx=True,
                on_selected=self.load_subset_workbook,
            ),
            stretch=1,
        )
        layout.addWidget(self.subset_load_button, stretch=0)
        return group

    def _build_subset_manifest_group(self) -> QGroupBox:
        group = QGroupBox("Subset Run", self)
        layout = QVBoxLayout(group)
        self.subset_manifest_label = QLabel("No subset workbook loaded.")
        self.subset_manifest_label.setWordWrap(True)
        self.subset_manifest_label.setProperty("themeRole", "mutedText")
        self.subset_manifest_label.setStyleSheet(f"color: {self._muted_text_color()};")
        layout.addWidget(self.subset_manifest_label)
        return group

    def _build_subset_selection_group(self) -> QGroupBox:
        group = QGroupBox("Subset Selection", self)
        layout = QVBoxLayout(group)

        workbook_row = QWidget(self)
        workbook_layout = QHBoxLayout(workbook_row)
        workbook_layout.setContentsMargins(0, 0, 0, 0)
        workbook_layout.addWidget(QLabel("Workbook Kind"))
        self.subset_spec_combo = QComboBox(self)
        self.subset_spec_combo.currentIndexChanged.connect(self._on_subset_spec_changed)
        workbook_layout.addWidget(self.subset_spec_combo, stretch=1)
        layout.addWidget(workbook_row)

        surf_widget = QWidget(self)
        surf_layout = QHBoxLayout(surf_widget)
        surf_layout.setContentsMargins(0, 0, 0, 0)
        surf_layout.addWidget(QLabel("Surface Family"))
        self.subset_surf_buttons: dict[str, QPushButton] = {}
        for surf_id in FOCUS_SURF_IDS:
            button = QPushButton(surf_id, self)
            button.setCheckable(True)
            button.setAutoExclusive(True)
            button.setEnabled(False)
            button.clicked.connect(self._refresh_subset_force_options)
            self.subset_surf_buttons[surf_id] = button
            surf_layout.addWidget(button)
        surf_layout.addStretch(1)
        layout.addWidget(surf_widget)

        force_widget = QWidget(self)
        force_layout = QHBoxLayout(force_widget)
        force_layout.setContentsMargins(0, 0, 0, 0)
        force_layout.addWidget(QLabel("Force"))
        self.subset_prev_force_button = QPushButton("Prev", self)
        self.subset_prev_force_button.setMaximumWidth(60)
        self.subset_force_combo = QComboBox(self)
        self.subset_force_combo.currentTextChanged.connect(self._refresh_subset_state_options)
        self.subset_next_force_button = QPushButton("Next", self)
        self.subset_next_force_button.setMaximumWidth(60)
        self.subset_prev_force_button.clicked.connect(lambda: self._navigate_subset_force(-1))
        self.subset_next_force_button.clicked.connect(lambda: self._navigate_subset_force(1))
        force_layout.addWidget(self.subset_prev_force_button)
        force_layout.addWidget(self.subset_force_combo, stretch=1)
        force_layout.addWidget(self.subset_next_force_button)
        layout.addWidget(force_widget)

        state_widget = QWidget(self)
        state_layout = QHBoxLayout(state_widget)
        state_layout.setContentsMargins(0, 0, 0, 0)
        state_layout.addWidget(QLabel("Surface State"))
        self.subset_state_buttons: dict[str, QPushButton] = {}
        for suffix, label in (("I", "Initial"), ("D", "Deformed")):
            button = QPushButton(label, self)
            button.setCheckable(True)
            button.setAutoExclusive(True)
            button.setEnabled(False)
            button.clicked.connect(self.refresh_subset_view)
            self.subset_state_buttons[suffix] = button
            state_layout.addWidget(button)
        state_layout.addStretch(1)
        layout.addWidget(state_widget)

        filter_widget = QWidget(self)
        filter_layout = QHBoxLayout(filter_widget)
        filter_layout.setContentsMargins(0, 0, 0, 0)
        filter_layout.addWidget(QLabel("Surf Filter"))
        self.subset_group_filter_combo = QComboBox(self)
        self.subset_group_filter_combo.currentTextChanged.connect(self.refresh_subset_view)
        filter_layout.addWidget(self.subset_group_filter_combo, stretch=1)
        filter_layout.addWidget(QLabel("Removed Count"))
        self.subset_removed_count_combo = QComboBox(self)
        self.subset_removed_count_combo.currentTextChanged.connect(self._render_subset_from_current_table_selection)
        filter_layout.addWidget(self.subset_removed_count_combo, stretch=1)
        layout.addWidget(filter_widget)
        return group

    def _build_subset_table_group(self) -> QGroupBox:
        group = QGroupBox("Subset Table", self)
        layout = QVBoxLayout(group)
        self.subset_table = QTableWidget(self)
        self.subset_table.setColumnCount(0)
        self.subset_table.setRowCount(0)
        self.subset_table.setSortingEnabled(True)
        self.subset_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.subset_table.setSelectionMode(QTableWidget.SingleSelection)
        self.subset_table.itemSelectionChanged.connect(self._render_subset_from_current_table_selection)
        layout.addWidget(self.subset_table)
        return group

    def _build_subset_details_group(self) -> QGroupBox:
        group = QGroupBox("Subset Details", self)
        layout = QVBoxLayout(group)
        self.subset_details_output = QPlainTextEdit(self)
        self.subset_details_output.setReadOnly(True)
        self.subset_details_output.setLineWrapMode(QPlainTextEdit.NoWrap)
        layout.addWidget(self.subset_details_output)
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
                selected = QFileDialog.getExistingDirectory(self, "Select Directory", line_edit.text() or str(SCRIPT_DIR))
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

    def _reset_run_name(self) -> None:
        self.run_name_edit.setText(self._default_run_name())

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
        args = [str(BATCH_SCRIPT), self.input_dir_edit.text().strip()]
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
        args.extend(["--roc-mode", roc_mode])
        if roc_mode == "fixed":
            args.extend(["--fixed-roc-um", f"{self.fixed_roc_spin.value():.6f}"])
        args.extend(["--sphere-fit-mode", sphere_fit_mode])
        args.extend(["--center-weight", f"{self.center_weight_spin.value():.2f}"])
        args.extend(["--normalization-mode", normalization_mode])
        args.extend(["--n-modes", str(self.n_modes_spin.value())])
        if not self.round_radii_check.isChecked():
            args.append("--no-round-radii-um")
        if not self.round_coeffs_check.isChecked():
            args.append("--no-round-zernike-coeffs")
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
        focus_files = [
            path
            for path in matched_files
            if is_focus_surface_family(parse_surface_metadata(path).surf_id)
        ]
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
        if not self._confirm_large_batch_run():
            return

        self.log_output.clear()
        self.refresh_command_preview()

        process = QProcess(self)
        process.setProgram(sys.executable)
        process.setArguments(self.build_command())
        process.setWorkingDirectory(str(SCRIPT_DIR))
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
            self.log_output.appendPlainText(chunk.rstrip("\n"))

    def _process_finished(self, exit_code: int, _exit_status: QProcess.ExitStatus) -> None:
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.log_output.appendPlainText(
            f"\nCompleted successfully with exit code {exit_code}.\n"
            if exit_code == 0
            else f"\nProcess exited with code {exit_code}.\n"
        )
        self.process = None

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

        self.summary_rows = rows
        self.summary_manifest = parse_run_manifest(path)
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
            self.preview_canvas.clear_message("Summary loaded, but no matching AA/AP/PA/PP row is currently selectable.")
        else:
            self.plot_current_selection()

    def _refresh_surf_options(self) -> None:
        available = {row["surf_id"] for row in self.summary_rows}
        current = self._selected_surf_id()
        target = current if current in available else next((surf_id for surf_id in FOCUS_SURF_IDS if surf_id in available), None)
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
        matching = [
            row
            for row in self.summary_rows
            if row["surf_id"] == surf_id and row["force_id"] == force_id
        ]
        available_suffixes = {row["surface_token"][-1].upper() for row in matching if row.get("surface_token")}
        current = self._selected_surface_suffix()
        target = current if current in available_suffixes else next((suffix for suffix in ("I", "D") if suffix in available_suffixes), None)
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
        text, details = self.preview_canvas.plot_selection(
            preview_row,
            self.summary_manifest,
            rho_axis_limit_um=rho_limit,
        )

        detail_lines = [
            f"Workbook: {self.summary_workbook_path}" if self.summary_workbook_path is not None else "Workbook:",
            f"Selected row: {row['run_name']} | {row['surf_id']} | {row['force_id']} | {row['surface_token']}",
            f"Source remap: {resolution_details['source_resolution_strategy']} | exists={resolution_details['source_exists']}",
            f"Original source file: {resolution_details['original_source_file']}",
            f"Coeff file: {details['coeff_file']}",
            f"Coeff remap: {resolution_details['coeff_resolution_strategy']} | exists={resolution_details['coeff_exists']}",
            f"Original coeff file: {resolution_details['original_coeff_file']}",
            f"Sphere SSE: {details['sphere_sse']}",
            f"Surface RMS: {details['surface_rms']}",
            f"Residual RMS: {details['residual_rms']}",
            f"Applied normalization radius (um): {details['applied_norm_radius_um']}",
            f"Observed aperture radius (um): {details['observed_aperture_radius_um']}",
            "",
            text,
        ]
        self.details_output.setPlainText("\n".join(detail_lines))

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

    def load_subset_workbook(self) -> None:
        # Public release guard: subset inspection is intentionally disabled.
        QMessageBox.information(
            self,
            "Subset Inspection Disabled",
            "Subset inspection is disabled in this public release.",
        )
        return

        path_text = self.subset_file_edit.text().strip()
        if not path_text:
            QMessageBox.warning(self, "Missing Workbook", "Choose a subset workbook.")
            return
        path = Path(path_text)
        if not path.exists():
            QMessageBox.warning(self, "Missing Workbook", f"Subset workbook not found:\n{path}")
            return

        rows = parse_inline_xlsx_rows(path)
        if not rows:
            QMessageBox.warning(self, "No Rows", "The selected subset workbook contains no rows.")
            return

        filtered_rows: list[dict[str, str]] = []
        for row in rows:
            surf_id = row.get("surf_id", "").strip()
            group_value = row.get("group_value", "").strip()
            group_type = row.get("group_type", "").strip()
            if surf_id and surf_id not in FOCUS_SURF_IDS:
                continue
            if group_type == "surf_id" and group_value and group_value not in FOCUS_SURF_IDS:
                continue
            filtered_rows.append(row)

        manifest = parse_run_manifest(path)
        self.subset_workbook_path = path
        self.subset_rows = filtered_rows
        self.subset_manifest = manifest
        self.subset_workbook_kind = detect_subset_workbook_kind(filtered_rows, path, manifest)
        self.subset_workbook_specs = list(manifest.get("gui_workbook_specs", []) or [])
        self._populate_subset_spec_combo()
        self._update_subset_manifest_label()
        self._configure_subset_controls()
        self.refresh_subset_view()

    def _populate_subset_spec_combo(self) -> None:
        self._subset_updating_spec_combo = True
        self.subset_spec_combo.blockSignals(True)
        self.subset_spec_combo.clear()
        self.subset_spec_paths = {}

        if self.subset_workbook_specs and self.subset_workbook_path is not None:
            run_dir = self.subset_workbook_path.parent
            for spec in self.subset_workbook_specs:
                file_name = str(spec.get("file", ""))
                kind = str(spec.get("kind", ""))
                description = str(spec.get("description", ""))
                display_kind = display_label(SUBSET_KIND_LABELS, kind)
                label = f"{display_kind} | {description}" if description else display_kind
                target_path = run_dir / file_name
                self.subset_spec_combo.addItem(label, str(target_path))
                self.subset_spec_paths[label] = target_path
            current_index = 0
            for index in range(self.subset_spec_combo.count()):
                current_path = Path(str(self.subset_spec_combo.itemData(index)))
                if current_path == self.subset_workbook_path:
                    current_index = index
                    break
            self.subset_spec_combo.setCurrentIndex(current_index)
            self.subset_spec_combo.setEnabled(True)
        elif self.subset_workbook_path is not None:
            display_kind = display_label(SUBSET_KIND_LABELS, self.subset_workbook_kind)
            self.subset_spec_combo.addItem(display_kind or self.subset_workbook_path.name, str(self.subset_workbook_path))
            self.subset_spec_combo.setCurrentIndex(0)
            self.subset_spec_combo.setEnabled(False)
        else:
            self.subset_spec_combo.setEnabled(False)

        self.subset_spec_combo.blockSignals(False)
        self._subset_updating_spec_combo = False

    def _on_subset_spec_changed(self) -> None:
        if self._subset_updating_spec_combo:
            return
        selected_path = self.subset_spec_combo.currentData()
        if not selected_path:
            return
        target_path = Path(str(selected_path))
        if self.subset_workbook_path is not None and target_path == self.subset_workbook_path:
            return
        self.subset_file_edit.setText(str(target_path))
        self.load_subset_workbook()

    def _update_subset_manifest_label(self) -> None:
        if not self.subset_rows:
            self.subset_manifest_label.setText("No subset workbook loaded.")
            return
        run_name = str(self.subset_manifest.get("run_name") or self.subset_rows[0].get("run_name", ""))
        normalization_mode = str(
            self.subset_manifest.get("normalization_mode") or self.subset_rows[0].get("normalization_mode", "")
        )
        targets = sorted({row.get("target", "") for row in self.subset_rows if row.get("target")})
        target = str(self.subset_manifest.get("target_pattern") or ", ".join(targets) or "n/a")
        source_count = self.subset_manifest.get("source_count")
        if source_count in (None, "", "null"):
            source_count = len({row.get("source_label", "") for row in self.subset_rows if row.get("source_label")})
        display_kind = display_label(SUBSET_KIND_LABELS, self.subset_workbook_kind)
        display_norm = display_label(NORMALIZATION_MODE_LABELS, normalization_mode) if normalization_mode else "n/a"
        self.subset_manifest_label.setText(
            f"Run: {run_name} | Kind: {display_kind} | "
            f"Normalization: {display_norm} | Target: {target} | "
            f"Source count: {source_count or 'n/a'}"
        )

    def _configure_subset_controls(self) -> None:
        per_source_kind = self.subset_workbook_kind in {
            "drop_importance",
            "subset_path_greedy",
            "subset_path_ranked",
            "global_consistent_subset",
        }
        surf_filter_kind = self.subset_workbook_kind == "mode_consistency_by_surf_id"
        removed_count_kind = self.subset_workbook_kind in {"global_mode_order", "global_consistent_subset_aggregate"}

        if per_source_kind:
            self._refresh_subset_surf_options()
        else:
            for button in self.subset_surf_buttons.values():
                button.setEnabled(False)
                button.setChecked(False)
            self.subset_force_combo.clear()
            for button in self.subset_state_buttons.values():
                button.setEnabled(False)
                button.setChecked(False)
            self._update_subset_force_navigation_buttons()

        self.subset_group_filter_combo.blockSignals(True)
        self.subset_group_filter_combo.clear()
        if surf_filter_kind:
            self.subset_group_filter_combo.addItems(["All", *FOCUS_SURF_IDS])
            self.subset_group_filter_combo.setEnabled(True)
        else:
            self.subset_group_filter_combo.addItem("All")
            self.subset_group_filter_combo.setEnabled(False)
        self.subset_group_filter_combo.blockSignals(False)

        self.subset_removed_count_combo.blockSignals(True)
        self.subset_removed_count_combo.clear()
        if removed_count_kind:
            if self.subset_workbook_kind == "global_mode_order":
                max_count = len(self.subset_rows)
                self.subset_removed_count_combo.addItems([str(count) for count in range(0, max_count + 1)])
            else:
                counts = sorted(
                    {
                        int(float(row.get("removed_mode_count", "0")))
                        for row in self.subset_rows
                        if row.get("removed_mode_count", "").strip()
                    }
                )
                self.subset_removed_count_combo.addItems([str(count) for count in counts])
            self.subset_removed_count_combo.setEnabled(True)
            if self.subset_removed_count_combo.count() > 0:
                self.subset_removed_count_combo.setCurrentIndex(0)
        else:
            self.subset_removed_count_combo.setEnabled(False)
        self.subset_removed_count_combo.blockSignals(False)

    def _refresh_subset_surf_options(self) -> None:
        available = {row["surf_id"] for row in self.subset_rows if row.get("surf_id") in FOCUS_SURF_IDS}
        current = self._selected_subset_surf_id()
        target = current if current in available else next((surf_id for surf_id in FOCUS_SURF_IDS if surf_id in available), None)
        for surf_id, button in self.subset_surf_buttons.items():
            button.blockSignals(True)
            button.setEnabled(surf_id in available)
            button.setChecked(surf_id == target)
            button.blockSignals(False)
        self._refresh_subset_force_options()

    def _refresh_subset_force_options(self) -> None:
        surf_id = self._selected_subset_surf_id()
        matching = [row for row in self.subset_rows if row.get("surf_id") == surf_id]
        force_ids = sorted({row["force_id"] for row in matching if row.get("force_id")}, key=self._force_sort_key)
        current = self.subset_force_combo.currentText()
        self.subset_force_combo.blockSignals(True)
        self.subset_force_combo.clear()
        self.subset_force_combo.addItems(force_ids)
        if current in force_ids:
            self.subset_force_combo.setCurrentText(current)
        elif force_ids:
            self.subset_force_combo.setCurrentIndex(0)
        self.subset_force_combo.blockSignals(False)
        self._update_subset_force_navigation_buttons()
        self._refresh_subset_state_options()

    def _refresh_subset_state_options(self) -> None:
        surf_id = self._selected_subset_surf_id()
        force_id = self.subset_force_combo.currentText()
        matching = [
            row
            for row in self.subset_rows
            if row.get("surf_id") == surf_id and row.get("force_id") == force_id
        ]
        available_suffixes = {
            row["surface_token"][-1].upper() for row in matching if row.get("surface_token")
        }
        current = self._selected_subset_state_suffix()
        target = current if current in available_suffixes else next((suffix for suffix in ("I", "D") if suffix in available_suffixes), None)
        for suffix, button in self.subset_state_buttons.items():
            button.blockSignals(True)
            button.setEnabled(suffix in available_suffixes)
            button.setChecked(suffix == target)
            button.blockSignals(False)
        self._update_subset_force_navigation_buttons()
        self.refresh_subset_view()

    def _selected_subset_surf_id(self) -> str:
        for surf_id, button in self.subset_surf_buttons.items():
            if button.isChecked():
                return surf_id
        return ""

    def _selected_subset_state_suffix(self) -> str:
        for suffix, button in self.subset_state_buttons.items():
            if button.isChecked():
                return suffix
        return ""

    def _navigate_subset_force(self, step: int) -> None:
        if self.subset_force_combo.count() == 0:
            return
        current_index = self.subset_force_combo.currentIndex()
        if current_index < 0:
            current_index = 0
        new_index = min(max(current_index + step, 0), self.subset_force_combo.count() - 1)
        if new_index != current_index:
            self.subset_force_combo.setCurrentIndex(new_index)
        self._update_subset_force_navigation_buttons()

    def _update_subset_force_navigation_buttons(self) -> None:
        count = self.subset_force_combo.count()
        current_index = self.subset_force_combo.currentIndex()
        has_items = count > 0 and current_index >= 0
        self.subset_prev_force_button.setEnabled(has_items and current_index > 0)
        self.subset_next_force_button.setEnabled(has_items and current_index < count - 1)

    def _current_subset_source_rows(self) -> list[dict[str, str]]:
        surf_id = self._selected_subset_surf_id()
        force_id = self.subset_force_combo.currentText()
        suffix = self._selected_subset_state_suffix()
        if not surf_id or not force_id or not suffix:
            return []
        token = f"{surf_id}{suffix}"
        return [
            row
            for row in self.subset_rows
            if row.get("surf_id") == surf_id
            and row.get("force_id") == force_id
            and row.get("surface_token") == token
        ]

    def _subset_display_rows(self) -> list[dict[str, str]]:
        if self.subset_workbook_kind in {
            "drop_importance",
            "subset_path_greedy",
            "subset_path_ranked",
            "global_consistent_subset",
        }:
            return self._current_subset_source_rows()
        if self.subset_workbook_kind == "mode_consistency_by_surf_id":
            selected_filter = self.subset_group_filter_combo.currentText()
            if selected_filter and selected_filter != "All":
                return [row for row in self.subset_rows if row.get("group_value") == selected_filter]
        return list(self.subset_rows)

    def _table_columns_for_subset_kind(self, kind: str) -> list[tuple[str, str]]:
        if kind == "drop_importance":
            return [
                ("impact_rank", "Rank"),
                ("removed_mode_noll", "Removed"),
                ("full_fit_abs_coefficient_um", "|Coeff|"),
                ("subset_rms", "Subset RMS"),
                ("delta_rms_vs_full", "dRMS"),
                ("delta_max_abs_residual_um_vs_full", "dMax Abs"),
            ]
        if kind in {"subset_path_greedy", "subset_path_ranked"}:
            return [
                ("step_index", "Step"),
                ("active_mode_count", "Active"),
                ("removed_mode_noll", "Removed"),
                ("rms", "Subset RMS"),
                ("delta_rms_vs_full", "dRMS"),
                ("delta_max_abs_residual_um_vs_full", "dMax Abs"),
            ]
        if kind in {"mode_consistency_overall", "mode_consistency_by_surf_id"}:
            return [
                ("mode_noll", "Mode"),
                ("sample_count", "N"),
                ("median_rank", "Median Rank"),
                ("top1_count", "Top1"),
                ("top3_count", "Top3"),
                ("top5_count", "Top5"),
                ("p95_delta_rms", "p95 dRMS"),
                ("max_delta_rms", "max dRMS"),
            ]
        if kind == "global_mode_order":
            return [
                ("global_order", "Order"),
                ("mode_noll", "Mode"),
                ("median_rank", "Median Rank"),
                ("top1_count", "Top1"),
                ("top3_count", "Top3"),
                ("p95_delta_rms", "p95 dRMS"),
                ("max_delta_rms", "max dRMS"),
            ]
        if kind == "global_consistent_subset":
            return [
                ("removed_mode_count", "Removed"),
                ("active_mode_count", "Active"),
                ("last_removed_mode_noll", "Last Removed"),
                ("rms", "Subset RMS"),
                ("delta_rms_vs_full", "dRMS"),
                ("delta_max_abs_residual_um_vs_full", "dMax Abs"),
            ]
        if kind == "global_consistent_subset_aggregate":
            return [
                ("removed_mode_count", "Removed"),
                ("active_mode_count", "Active"),
                ("median_delta_rms", "Median dRMS"),
                ("p95_delta_rms", "p95 dRMS"),
                ("max_delta_rms", "max dRMS"),
                ("worst_case_surface_token", "Worst Token"),
            ]
        return []

    def _populate_subset_table(self, rows: list[dict[str, str]]) -> dict[str, str] | None:
        columns = self._table_columns_for_subset_kind(self.subset_workbook_kind)
        self.subset_table.blockSignals(True)
        self.subset_table.setSortingEnabled(False)
        self.subset_table.clear()
        self.subset_table.setColumnCount(len(columns))
        self.subset_table.setHorizontalHeaderLabels([label for _field, label in columns])
        self.subset_table.setRowCount(len(rows))

        for row_index, row in enumerate(rows):
            for column_index, (field, _label) in enumerate(columns):
                raw_value = row.get(field, "")
                numeric = parse_optional_float(str(raw_value)) if str(raw_value).strip() else None
                display = format_metric(numeric if numeric is not None else raw_value)
                sort_value: float | int | str = numeric if numeric is not None else str(raw_value)
                item = NumericTableWidgetItem(display, sort_value)
                if column_index == 0:
                    item.setData(Qt.UserRole, row)
                self.subset_table.setItem(row_index, column_index, item)

        self.subset_table.resizeColumnsToContents()
        self.subset_table.setSortingEnabled(True)
        selected_row = rows[0] if rows else None
        if rows:
            self.subset_table.selectRow(0)
        self.subset_table.blockSignals(False)
        return selected_row

    def _current_subset_table_row(self) -> dict[str, str] | None:
        selected_items = self.subset_table.selectedItems()
        if not selected_items:
            return None
        return selected_items[0].data(Qt.UserRole)

    def refresh_subset_view(self) -> None:
        if not self.subset_rows:
            self.subset_canvas.clear_message("Load a subset workbook to inspect omission results.")
            self.subset_details_output.setPlainText("No subset workbook loaded.")
            return

        display_rows = self._subset_display_rows()
        if not display_rows:
            self.subset_canvas.clear_message("No rows match the current subset filters.")
            self.subset_details_output.setPlainText("No subset rows match the current selection.")
            self.subset_table.blockSignals(True)
            self.subset_table.clear()
            self.subset_table.setRowCount(0)
            self.subset_table.setColumnCount(0)
            self.subset_table.blockSignals(False)
            return

        self._populate_subset_table(display_rows)
        self._render_subset_from_current_table_selection()

    def _render_subset_from_current_table_selection(self) -> None:
        if not self.subset_rows:
            return
        display_rows = self._subset_display_rows()
        if not display_rows:
            return

        selected_row = self._current_subset_table_row()
        if selected_row is None:
            selected_row = display_rows[0]

        detail = ""
        if self.subset_workbook_kind == "drop_importance":
            detail = self.subset_canvas.plot_drop_importance(display_rows, selected_row)
        elif self.subset_workbook_kind in {"subset_path_greedy", "subset_path_ranked"}:
            title = "Subset Path: Greedy Refit" if self.subset_workbook_kind == "subset_path_greedy" else "Subset Path: Ranked Refit"
            detail = self.subset_canvas.plot_subset_path(display_rows, selected_row, title=title)
        elif self.subset_workbook_kind in {"mode_consistency_overall", "mode_consistency_by_surf_id"}:
            title = (
                "Mode Consistency by Surface Family"
                if self.subset_workbook_kind == "mode_consistency_by_surf_id"
                else "Mode Consistency Overall"
            )
            detail = self.subset_canvas.plot_mode_consistency(display_rows, selected_row, title=title)
        elif self.subset_workbook_kind == "global_mode_order":
            prefix_text = self.subset_removed_count_combo.currentText().strip()
            prefix_count = int(prefix_text) if prefix_text else 0
            detail = self.subset_canvas.plot_global_mode_order(display_rows, prefix_count=prefix_count)
            if selected_row is not None:
                detail = (
                    f"{detail}\nSelected row: order={selected_row.get('global_order', '')}, "
                    f"mode=Z{selected_row.get('mode_noll', '')}, "
                    f"p95_delta_rms={format_metric(parse_optional_float(selected_row.get('p95_delta_rms', '')) or 0.0)}"
                )
        elif self.subset_workbook_kind == "global_consistent_subset_aggregate":
            selected_text = self.subset_removed_count_combo.currentText().strip()
            selected_count = int(selected_text) if selected_text else None
            detail = self.subset_canvas.plot_global_subset_aggregate(display_rows, selected_count=selected_count)
            if selected_row is not None:
                detail = (
                    f"{detail}\nSelected row: removed={selected_row.get('removed_mode_count', '')}, "
                    f"median={format_metric(parse_optional_float(selected_row.get('median_delta_rms', '')) or 0.0)}, "
                    f"p95={format_metric(parse_optional_float(selected_row.get('p95_delta_rms', '')) or 0.0)}"
                )
        elif self.subset_workbook_kind == "global_consistent_subset":
            detail = self.subset_canvas.plot_global_subset_source(display_rows, selected_row)
        else:
            self.subset_canvas.clear_message(f"Unsupported subset workbook kind: {self.subset_workbook_kind}")
            detail = f"Unsupported subset workbook kind: {self.subset_workbook_kind}"

        source_label = selected_row.get("source_label", "") if selected_row is not None else ""
        self.subset_details_output.setPlainText(
            "\n".join(
                [
                    f"Workbook: {self.subset_workbook_path}" if self.subset_workbook_path is not None else "Workbook:",
                    f"Kind: {self.subset_workbook_kind}",
                    f"Selected source: {source_label or 'aggregate view'}",
                    "",
                    detail,
                ]
            )
        )

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
