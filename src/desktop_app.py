"""Windows-friendly Tkinter desktop interface for ai-image-reviewer."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Callable
import webbrowser

from .config import ConfigError
from .gui_controller import (
    CROP_MODE_DESCRIPTIONS,
    CROP_MODE_LABELS,
    REASONING_EFFORTS,
    REVIEW_MODE_DESCRIPTIONS,
    REVIEW_MODE_LABELS,
    BackendConnection,
    ConfigStore,
    DesktopSettings,
    ReviewEngine,
    ReviewProgress,
    ReviewSummary,
    check_backend_connection,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class _CallbackLogHandler(logging.Handler):
    def __init__(self, callback: Callable[[str], None]) -> None:
        super().__init__()
        self.callback = callback

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.callback(self.format(record))
        except Exception:
            self.handleError(record)


class DesktopApp:
    """Tk widgets and queue-based coordination with :class:`ReviewEngine`."""

    POLL_INTERVAL_MS = 100

    def __init__(self, root: tk.Tk, *, store: ConfigStore | None = None) -> None:
        self.root = root
        self.store = store or ConfigStore(PROJECT_ROOT)
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.engine: ReviewEngine | None = None
        self.worker: threading.Thread | None = None
        self.session_logger: logging.Logger | None = None
        self.closing = False
        self.last_report_path: Path | None = None

        self.root.title("AI Image Reviewer")
        self.root.geometry("1180x820")
        self.root.minsize(980, 680)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._configure_style()
        self._create_variables()
        self._build_layout()

        startup_error: str | None = None
        try:
            settings = self.store.load_settings()
        except (ConfigError, OSError, ValueError) as exc:
            startup_error = str(exc)
            settings = DesktopSettings(
                input_paths=(PROJECT_ROOT / "samples" / "incoming",),
                output_path=PROJECT_ROOT / "output",
            )
        self._apply_settings(settings)
        self.root.after(self.POLL_INTERVAL_MS, self._drain_events)
        if startup_error:
            self.root.after(
                50,
                lambda: messagebox.showwarning(
                    "設定を読み込めませんでした",
                    f"既定値で起動しました。\n\n{startup_error}",
                    parent=self.root,
                ),
            )

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Header.TLabel", font=("Yu Gothic UI", 18, "bold"))
        style.configure("Subheader.TLabel", font=("Yu Gothic UI", 9), foreground="#4b5563")
        style.configure("Status.TLabel", font=("Yu Gothic UI", 10, "bold"))
        style.configure("Accent.TButton", font=("Yu Gothic UI", 10, "bold"), padding=(14, 7))
        style.configure("Treeview", rowheight=25)

    def _create_variables(self) -> None:
        self.backend_var = tk.StringVar(value="codex_cli")
        self.codex_model_var = tk.StringVar(value="gpt-5.6-luna")
        self.reasoning_var = tk.StringVar(value="low")
        self.review_mode_var = tk.StringVar(value=REVIEW_MODE_LABELS["standard"])
        self.review_mode_help_var = tk.StringVar(value=REVIEW_MODE_DESCRIPTIONS["standard"])
        self.crop_recheck_enabled_var = tk.BooleanVar(value=False)
        self.crop_mode_var = tk.StringVar(value=CROP_MODE_LABELS["balanced"])
        self.crop_mode_help_var = tk.StringVar(value=CROP_MODE_DESCRIPTIONS["balanced"])
        self.keep_crop_files_var = tk.BooleanVar(value=True)
        self.lmstudio_url_var = tk.StringVar(value="http://127.0.0.1:1234/v1")
        self.lmstudio_model_var = tk.StringVar(value="qwen3-vl-8b")
        self.output_var = tk.StringVar(value=os.fspath(PROJECT_ROOT / "output"))
        self.operation_var = tk.StringVar(value="copy")
        self.recursive_var = tk.BooleanVar(value=True)
        self.preserve_relative_var = tk.BooleanVar(value=True)
        self.force_var = tk.BooleanVar(value=False)
        self.manual_path_var = tk.StringVar()
        self.status_var = tk.StringVar(value="設定を確認して開始してください")
        self.connection_var = tk.StringVar(value="未確認")
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_text_var = tk.StringVar(value="待機中")
        self.counts_var = tk.StringVar(value="PASS 0   REVIEW 0   FAIL 0   ERROR 0")

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, padding=(18, 14))
        outer.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(outer)
        header.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(header, text="AI Image Reviewer", style="Header.TLabel").pack(side=tk.LEFT)
        ttk.Label(
            header,
            text="生成画像を自動判定し、PASS / REVIEW / FAILへ安全に仕分け",
            style="Subheader.TLabel",
        ).pack(side=tk.LEFT, padx=(16, 0), pady=(8, 0))
        ttk.Label(header, textvariable=self.status_var, style="Status.TLabel").pack(side=tk.RIGHT, pady=(6, 0))

        settings_row = ttk.Frame(outer)
        settings_row.pack(fill=tk.X)
        settings_row.columnconfigure(0, weight=3)
        settings_row.columnconfigure(1, weight=2)

        paths_frame = ttk.LabelFrame(settings_row, text="1. 入出力フォルダ", padding=10)
        paths_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        paths_frame.columnconfigure(0, weight=1)

        list_frame = ttk.Frame(paths_frame)
        list_frame.grid(row=0, column=0, columnspan=3, sticky="nsew")
        list_frame.columnconfigure(0, weight=1)
        self.input_list = tk.Listbox(
            list_frame,
            height=4,
            selectmode=tk.EXTENDED,
            font=("Yu Gothic UI", 9),
            activestyle="none",
        )
        self.input_list.grid(row=0, column=0, sticky="nsew")
        input_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.input_list.yview)
        input_scroll.grid(row=0, column=1, sticky="ns")
        self.input_list.configure(yscrollcommand=input_scroll.set)
        path_buttons = ttk.Frame(list_frame)
        path_buttons.grid(row=0, column=2, sticky="ns", padx=(8, 0))
        ttk.Button(path_buttons, text="フォルダ追加", command=self._browse_input).pack(fill=tk.X)
        ttk.Button(path_buttons, text="選択を削除", command=self._remove_inputs).pack(fill=tk.X, pady=(6, 0))

        ttk.Entry(paths_frame, textvariable=self.manual_path_var).grid(
            row=1, column=0, sticky="ew", pady=(8, 0)
        )
        ttk.Button(paths_frame, text="パスを追加", command=self._add_manual_input).grid(
            row=1, column=1, sticky="ew", padx=(6, 0), pady=(8, 0)
        )
        ttk.Label(paths_frame, text="UNCパスも直接入力できます", style="Subheader.TLabel").grid(
            row=1, column=2, sticky="w", padx=(8, 0), pady=(8, 0)
        )

        ttk.Label(paths_frame, text="出力先").grid(row=2, column=0, sticky="w", pady=(10, 0))
        output_row = ttk.Frame(paths_frame)
        output_row.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(3, 0))
        output_row.columnconfigure(0, weight=1)
        ttk.Entry(output_row, textvariable=self.output_var).grid(row=0, column=0, sticky="ew")
        ttk.Button(output_row, text="参照", command=self._browse_output).grid(row=0, column=1, padx=(6, 0))

        option_row = ttk.Frame(paths_frame)
        option_row.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        ttk.Checkbutton(option_row, text="サブフォルダーを含める", variable=self.recursive_var).pack(side=tk.LEFT)
        ttk.Checkbutton(option_row, text="フォルダー構造を保持", variable=self.preserve_relative_var).pack(
            side=tk.LEFT, padx=(14, 0)
        )
        ttk.Label(option_row, text="処理:").pack(side=tk.LEFT, padx=(18, 4))
        ttk.Combobox(
            option_row,
            textvariable=self.operation_var,
            values=("copy", "move"),
            state="readonly",
            width=7,
        ).pack(side=tk.LEFT)

        backend_frame = ttk.LabelFrame(settings_row, text="2. 判定バックエンド", padding=10)
        backend_frame.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        backend_frame.columnconfigure(0, weight=1)
        self.backend_tabs = ttk.Notebook(backend_frame)
        self.backend_tabs.grid(row=0, column=0, sticky="nsew")
        self.backend_tabs.bind("<<NotebookTabChanged>>", self._on_backend_tab_changed)

        codex_tab = ttk.Frame(self.backend_tabs, padding=10)
        codex_tab.columnconfigure(1, weight=1)
        self.backend_tabs.add(codex_tab, text="Codex CLI / ChatGPT")
        ttk.Label(codex_tab, text="モデル").grid(row=0, column=0, sticky="w")
        ttk.Entry(codex_tab, textvariable=self.codex_model_var).grid(
            row=0, column=1, sticky="ew", padx=(8, 0)
        )
        ttk.Label(codex_tab, text="推論設定").grid(row=1, column=0, sticky="w", pady=(9, 0))
        ttk.Combobox(
            codex_tab,
            textvariable=self.reasoning_var,
            values=REASONING_EFFORTS,
            state="readonly",
        ).grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(9, 0))
        ttk.Label(
            codex_tab,
            text="画像はOpenAIへ送信されます。ChatGPT認証だけを許可し、APIキー経路は停止します。",
            style="Subheader.TLabel",
            wraplength=390,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(10, 0))

        lmstudio_tab = ttk.Frame(self.backend_tabs, padding=10)
        lmstudio_tab.columnconfigure(1, weight=1)
        self.backend_tabs.add(lmstudio_tab, text="LM Studio / ローカル")
        ttk.Label(lmstudio_tab, text="API URL").grid(row=0, column=0, sticky="w")
        ttk.Entry(lmstudio_tab, textvariable=self.lmstudio_url_var).grid(
            row=0, column=1, columnspan=2, sticky="ew", padx=(8, 0)
        )
        ttk.Label(lmstudio_tab, text="モデル").grid(row=1, column=0, sticky="w", pady=(9, 0))
        self.lmstudio_model_combo = ttk.Combobox(
            lmstudio_tab,
            textvariable=self.lmstudio_model_var,
            values=(),
        )
        self.lmstudio_model_combo.grid(row=1, column=1, sticky="ew", padx=(8, 6), pady=(9, 0))
        ttk.Button(lmstudio_tab, text="一覧取得", command=self._test_connection).grid(
            row=1, column=2, pady=(9, 0)
        )
        ttk.Label(
            lmstudio_tab,
            text="LM StudioのOpenAI互換APIを利用します。画像入力対応モデルを選択してください。",
            style="Subheader.TLabel",
            wraplength=390,
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(10, 0))

        connection_row = ttk.Frame(backend_frame)
        connection_row.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        self.connection_button = ttk.Button(connection_row, text="接続確認", command=self._test_connection)
        self.connection_button.pack(side=tk.LEFT)
        ttk.Label(connection_row, textvariable=self.connection_var).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Label(connection_row, text="判定基準:").pack(side=tk.LEFT, padx=(18, 4))
        review_mode_combo = ttk.Combobox(
            connection_row,
            textvariable=self.review_mode_var,
            values=tuple(REVIEW_MODE_LABELS.values()),
            state="readonly",
            width=14,
        )
        review_mode_combo.pack(side=tk.LEFT)
        review_mode_combo.bind("<<ComboboxSelected>>", self._on_review_mode_changed)
        ttk.Label(
            backend_frame,
            textvariable=self.review_mode_help_var,
            style="Subheader.TLabel",
            wraplength=500,
        ).grid(row=2, column=0, sticky="w", pady=(7, 0))

        crop_frame = ttk.LabelFrame(backend_frame, text="追加確認（クロップ再判定）", padding=7)
        crop_frame.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        crop_frame.columnconfigure(2, weight=1)
        ttk.Checkbutton(
            crop_frame,
            text="クロップ再判定を有効にする",
            variable=self.crop_recheck_enabled_var,
        ).grid(row=0, column=0, columnspan=3, sticky="w")
        ttk.Label(crop_frame, text="確認モード:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        crop_mode_combo = ttk.Combobox(
            crop_frame,
            textvariable=self.crop_mode_var,
            values=tuple(CROP_MODE_LABELS.values()),
            state="readonly",
            width=17,
        )
        crop_mode_combo.grid(row=1, column=1, sticky="w", padx=(8, 0), pady=(6, 0))
        crop_mode_combo.bind("<<ComboboxSelected>>", self._on_crop_mode_changed)
        ttk.Checkbutton(
            crop_frame,
            text="クロップ画像を保持",
            variable=self.keep_crop_files_var,
        ).grid(row=1, column=2, sticky="w", padx=(12, 0), pady=(6, 0))
        ttk.Label(
            crop_frame,
            textvariable=self.crop_mode_help_var,
            style="Subheader.TLabel",
            wraplength=500,
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(6, 0))

        actions = ttk.Frame(outer)
        actions.pack(fill=tk.X, pady=(12, 8))
        self.save_button = ttk.Button(actions, text="設定を保存", command=self._save_settings)
        self.save_button.pack(side=tk.LEFT)
        self.scan_button = ttk.Button(actions, text="一括スキャン開始", style="Accent.TButton", command=self._start_scan)
        self.scan_button.pack(side=tk.LEFT, padx=(8, 0))
        self.watch_button = ttk.Button(actions, text="フォルダ監視開始", command=self._start_watch)
        self.watch_button.pack(side=tk.LEFT, padx=(8, 0))
        self.stop_button = ttk.Button(actions, text="停止", command=self._stop, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(8, 0))
        ttk.Checkbutton(actions, text="処理済み画像も再判定", variable=self.force_var).pack(
            side=tk.LEFT, padx=(16, 0)
        )
        ttk.Button(actions, text="出力フォルダを開く", command=self._open_output).pack(side=tk.RIGHT)
        ttk.Button(actions, text="レポートを開く", command=self._open_report).pack(side=tk.RIGHT, padx=(0, 8))

        progress_frame = ttk.Frame(outer)
        progress_frame.pack(fill=tk.X, pady=(0, 8))
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(progress_frame, textvariable=self.progress_text_var, width=36).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Label(progress_frame, textvariable=self.counts_var, width=43).pack(side=tk.RIGHT)

        panes = ttk.Panedwindow(outer, orient=tk.VERTICAL)
        panes.pack(fill=tk.BOTH, expand=True)
        results_frame = ttk.LabelFrame(panes, text="判定結果", padding=6)
        log_frame = ttk.LabelFrame(panes, text="実行ログ", padding=6)
        panes.add(results_frame, weight=3)
        panes.add(log_frame, weight=1)

        columns = (
            "model_result",
            "result",
            "decision_source",
            "confidence",
            "file",
            "problems",
            "destination",
        )
        self.results = ttk.Treeview(results_frame, columns=columns, show="headings")
        headings = {
            "model_result": "モデル判定",
            "result": "最終判定",
            "decision_source": "判定源",
            "confidence": "確信度",
            "file": "ファイル",
            "problems": "問題点",
            "destination": "出力先",
        }
        widths = {
            "model_result": 85,
            "result": 85,
            "decision_source": 95,
            "confidence": 75,
            "file": 210,
            "problems": 320,
            "destination": 340,
        }
        for name in columns:
            self.results.heading(name, text=headings[name])
            self.results.column(name, width=widths[name], minwidth=60, anchor=tk.W)
        self.results.column("model_result", anchor=tk.CENTER, stretch=False)
        self.results.column("result", anchor=tk.CENTER, stretch=False)
        self.results.column("decision_source", anchor=tk.CENTER, stretch=False)
        self.results.column("confidence", anchor=tk.CENTER, stretch=False)
        result_y = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results.yview)
        result_x = ttk.Scrollbar(results_frame, orient=tk.HORIZONTAL, command=self.results.xview)
        self.results.configure(yscrollcommand=result_y.set, xscrollcommand=result_x.set)
        self.results.grid(row=0, column=0, sticky="nsew")
        result_y.grid(row=0, column=1, sticky="ns")
        result_x.grid(row=1, column=0, sticky="ew")
        results_frame.rowconfigure(0, weight=1)
        results_frame.columnconfigure(0, weight=1)

        self.log_text = tk.Text(
            log_frame,
            height=7,
            wrap=tk.WORD,
            state=tk.DISABLED,
            font=("Consolas", 9),
            background="#111827",
            foreground="#e5e7eb",
            insertbackground="#e5e7eb",
        )
        log_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        log_scroll.grid(row=0, column=1, sticky="ns")
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)

    def _apply_settings(self, settings: DesktopSettings) -> None:
        self.input_list.delete(0, tk.END)
        for path in settings.input_paths:
            self.input_list.insert(tk.END, os.fspath(path))
        self.output_var.set(os.fspath(settings.output_path))
        self.backend_var.set(settings.backend)
        self.codex_model_var.set(settings.codex_model)
        self.reasoning_var.set(settings.reasoning_effort)
        self.review_mode_var.set(REVIEW_MODE_LABELS[settings.review_mode])
        self.review_mode_help_var.set(REVIEW_MODE_DESCRIPTIONS[settings.review_mode])
        self.crop_recheck_enabled_var.set(settings.crop_recheck_enabled)
        self.crop_mode_var.set(CROP_MODE_LABELS[settings.crop_mode])
        self.crop_mode_help_var.set(CROP_MODE_DESCRIPTIONS[settings.crop_mode])
        self.keep_crop_files_var.set(settings.keep_crop_files)
        self.lmstudio_url_var.set(settings.lmstudio_url)
        self.lmstudio_model_var.set(settings.lmstudio_model)
        self.operation_var.set(settings.operation)
        self.recursive_var.set(settings.recursive)
        self.preserve_relative_var.set(settings.preserve_relative_paths)
        self.backend_tabs.select(0 if settings.backend == "codex_cli" else 1)
        try:
            self.last_report_path = self.store.load_config().report_path
        except (ConfigError, OSError, ValueError):
            self.last_report_path = PROJECT_ROOT / "review.html"

    def _on_backend_tab_changed(self, _event: object | None = None) -> None:
        if not hasattr(self, "backend_tabs"):
            return
        self.backend_var.set("codex_cli" if self.backend_tabs.index("current") == 0 else "lmstudio")
        self.connection_var.set("未確認")

    def _on_review_mode_changed(self, _event: object | None = None) -> None:
        mode = next(
            (key for key, label in REVIEW_MODE_LABELS.items() if label == self.review_mode_var.get()),
            "standard",
        )
        self.review_mode_help_var.set(REVIEW_MODE_DESCRIPTIONS[mode])

    def _on_crop_mode_changed(self, _event: object | None = None) -> None:
        mode = next(
            (key for key, label in CROP_MODE_LABELS.items() if label == self.crop_mode_var.get()),
            "balanced",
        )
        self.crop_mode_help_var.set(CROP_MODE_DESCRIPTIONS[mode])

    def _browse_input(self) -> None:
        selected = filedialog.askdirectory(parent=self.root, title="入力フォルダを選択")
        if selected:
            self._insert_input(selected)

    def _add_manual_input(self) -> None:
        value = self.manual_path_var.get().strip().strip('"')
        if value:
            self._insert_input(value)
            self.manual_path_var.set("")

    def _insert_input(self, value: str) -> None:
        normalized = os.fspath(Path(value).expanduser().resolve())
        existing = {str(self.input_list.get(index)) for index in range(self.input_list.size())}
        if normalized not in existing:
            self.input_list.insert(tk.END, normalized)

    def _remove_inputs(self) -> None:
        for index in reversed(self.input_list.curselection()):
            self.input_list.delete(index)

    def _browse_output(self) -> None:
        selected = filedialog.askdirectory(parent=self.root, title="出力フォルダを選択")
        if selected:
            self.output_var.set(os.fspath(Path(selected).resolve()))

    def _settings_from_ui(self) -> DesktopSettings:
        paths = tuple(Path(str(self.input_list.get(index))) for index in range(self.input_list.size()))
        review_mode = next(
            (key for key, label in REVIEW_MODE_LABELS.items() if label == self.review_mode_var.get()),
            "standard",
        )
        crop_mode = next(
            (key for key, label in CROP_MODE_LABELS.items() if label == self.crop_mode_var.get()),
            "balanced",
        )
        return DesktopSettings(
            input_paths=paths,
            output_path=Path(self.output_var.get().strip().strip('"')),
            backend=self.backend_var.get(),
            codex_model=self.codex_model_var.get(),
            reasoning_effort=self.reasoning_var.get(),
            review_mode=review_mode,
            crop_recheck_enabled=self.crop_recheck_enabled_var.get(),
            crop_mode=crop_mode,
            keep_crop_files=self.keep_crop_files_var.get(),
            lmstudio_url=self.lmstudio_url_var.get(),
            lmstudio_model=self.lmstudio_model_var.get(),
            operation=self.operation_var.get(),
            recursive=self.recursive_var.get(),
            preserve_relative_paths=self.preserve_relative_var.get(),
        )

    def _save_settings(self, *, show_confirmation: bool = True):
        try:
            config = self.store.save(self._settings_from_ui())
        except (ConfigError, OSError, ValueError) as exc:
            messagebox.showerror("設定エラー", str(exc), parent=self.root)
            return None
        self.last_report_path = config.report_path
        self.status_var.set(f"設定保存済み: {self.store.user_path.name}")
        if show_confirmation:
            self._append_log(f"設定を保存しました: {self.store.user_path}")
        return config

    def _test_connection(self) -> None:
        if self._is_busy():
            return
        config = self._save_settings(show_confirmation=False)
        if config is None:
            return
        self.connection_var.set("確認中...")
        self.connection_button.configure(state=tk.DISABLED)

        def work() -> None:
            try:
                result = check_backend_connection(config)
                self.events.put(("connection", result))
            except Exception as exc:
                self.events.put(("connection_error", exc))

        threading.Thread(target=work, name="backend-check", daemon=True).start()

    def _start_scan(self) -> None:
        self._start(monitor=False)

    def _start_watch(self) -> None:
        self._start(monitor=True)

    def _start(self, *, monitor: bool) -> None:
        if self._is_busy():
            return
        config = self._save_settings(show_confirmation=False)
        if config is None:
            return
        if config.output.operation == "move":
            confirmed = messagebox.askyesno(
                "move処理の確認",
                "moveでは元画像が出力先へ移動します。\nコピーを残さず実行してよいですか？",
                icon=messagebox.WARNING,
                parent=self.root,
            )
            if not confirmed:
                return

        for item in self.results.get_children():
            self.results.delete(item)
        self.progress_var.set(0)
        self.counts_var.set("PASS 0   REVIEW 0   FAIL 0   ERROR 0")
        self.progress_text_var.set("準備中...")
        self._set_busy(True)
        logger = self._create_session_logger(config.application_log_path)
        engine = ReviewEngine(config, logger=logger)
        self.engine = engine
        force = self.force_var.get()

        def work() -> None:
            try:
                summary = engine.run(
                    force=force,
                    monitor=monitor,
                    on_progress=lambda progress: self.events.put(("progress", progress)),
                )
                self.events.put(("done", summary))
            except Exception as exc:
                self.events.put(("run_error", exc))

        self.worker = threading.Thread(target=work, name="review-worker", daemon=True)
        self.worker.start()
        self.status_var.set("フォルダ監視を準備中" if monitor else "スキャン実行中")

    def _create_session_logger(self, log_path: Path) -> logging.Logger:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger(f"ai_image_reviewer.gui.{id(self)}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        for handler in logger.handlers:
            handler.close()
        logger.handlers.clear()

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        callback_handler = _CallbackLogHandler(lambda line: self.events.put(("log", line)))
        callback_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(file_handler)
        logger.addHandler(callback_handler)
        self.session_logger = logger
        return logger

    def _close_session_logger(self) -> None:
        if self.session_logger is None:
            return
        for handler in self.session_logger.handlers:
            handler.close()
        self.session_logger.handlers.clear()
        self.session_logger = None

    def _stop(self) -> None:
        if self.engine is None:
            return
        self.engine.stop()
        if self.session_logger is not None:
            self.session_logger.info("停止要求を受け付けました。現在の判定要求が完了次第、残りの処理を停止します")
        self.stop_button.configure(state=tk.DISABLED)
        self.status_var.set("停止要求済み")
        self.progress_text_var.set("現在の画像の判定終了後に停止します")

    def _is_busy(self) -> bool:
        return self.worker is not None and self.worker.is_alive()

    def _set_busy(self, busy: bool) -> None:
        normal = tk.DISABLED if busy else tk.NORMAL
        self.save_button.configure(state=normal)
        self.scan_button.configure(state=normal)
        self.watch_button.configure(state=normal)
        self.connection_button.configure(state=normal)
        self.stop_button.configure(state=tk.NORMAL if busy else tk.DISABLED)

    def _drain_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "log":
                    self._append_log(str(payload))
                elif kind == "progress":
                    self._show_progress(payload)  # type: ignore[arg-type]
                elif kind == "connection":
                    self._show_connection(payload)  # type: ignore[arg-type]
                elif kind == "connection_error":
                    self.connection_button.configure(state=tk.NORMAL)
                    self.connection_var.set("接続失敗")
                    messagebox.showerror("接続確認に失敗しました", str(payload), parent=self.root)
                elif kind == "done":
                    self._show_done(payload)  # type: ignore[arg-type]
                elif kind == "run_error":
                    self._show_run_error(payload)  # type: ignore[arg-type]
        except queue.Empty:
            pass
        try:
            if self.root.winfo_exists():
                self.root.after(self.POLL_INTERVAL_MS, self._drain_events)
        except tk.TclError:
            pass

    def _show_connection(self, result: BackendConnection) -> None:
        self.connection_button.configure(state=tk.NORMAL)
        self.connection_var.set(("OK: " if result.ok else "要確認: ") + result.message)
        self._append_log(result.message)
        if result.models:
            self.lmstudio_model_combo.configure(values=result.models)
            if not result.ok and result.backend == "lmstudio":
                self.backend_tabs.select(1)

    def _show_progress(self, progress: ReviewProgress) -> None:
        if progress.monitoring:
            if str(self.progress_bar.cget("mode")) != "indeterminate":
                self.progress_bar.configure(mode="indeterminate")
                self.progress_bar.start(12)
            self.status_var.set("フォルダ監視中")
            self.progress_text_var.set(
                f"監視中: {progress.current_path.name}" if progress.current_path else "新しい画像を待機中"
            )
        else:
            if str(self.progress_bar.cget("mode")) != "determinate":
                self.progress_bar.stop()
                self.progress_bar.configure(mode="determinate")
            percent = (progress.completed / progress.total * 100) if progress.total else 0
            self.progress_var.set(percent)
            current = progress.current_path.name if progress.current_path else "準備中"
            self.progress_text_var.set(f"{progress.completed}/{progress.total}  {current}")
        counts = progress.counts
        self.counts_var.set(
            f"PASS {counts.get('PASS', 0)}   REVIEW {counts.get('REVIEW', 0)}   "
            f"FAIL {counts.get('FAIL', 0)}   ERROR {counts.get('error', 0)}"
        )
        if progress.record is not None:
            record = progress.record
            confidence = f"{record.confidence:.0%}"
            problems = ", ".join(record.problems) if record.problems else record.summary
            self.results.insert(
                "",
                0,
                values=(
                    record.model_result or "—",
                    record.result,
                    record.decision_source,
                    confidence,
                    record.source_path,
                    problems,
                    record.destination_path or "",
                ),
            )

    def _show_done(self, summary: ReviewSummary) -> None:
        self.progress_bar.stop()
        self.progress_bar.configure(mode="determinate")
        self.engine = None
        self.worker = None
        self.last_report_path = summary.report_path
        self._set_busy(False)
        if summary.cancelled:
            self.status_var.set("停止しました")
            self.progress_text_var.set("完了済みの結果は保存されています")
            if self.session_logger is not None:
                self.session_logger.info("停止完了: この実行で処理した画像 %d件", len(summary.records))
        elif summary.monitoring:
            self.status_var.set("フォルダ監視を終了しました")
            self.progress_text_var.set("監視終了")
            if self.session_logger is not None:
                self.session_logger.info("フォルダ監視を終了しました")
        else:
            self.status_var.set("スキャン完了")
            self.progress_var.set(100)
            self.progress_text_var.set(f"判定結果 {len(summary.records)}件")
            if self.session_logger is not None:
                self.session_logger.info("スキャン完了: この実行で処理した画像 %d件", len(summary.records))
        self._close_session_logger()
        if self.closing:
            self.root.destroy()

    def _show_run_error(self, error: Exception) -> None:
        self.progress_bar.stop()
        self.progress_bar.configure(mode="determinate")
        self.engine = None
        self.worker = None
        self._set_busy(False)
        self._close_session_logger()
        self.status_var.set("エラーで停止しました")
        self._append_log(f"ERROR: {error}")
        if self.closing:
            self.root.destroy()
            return
        messagebox.showerror("処理に失敗しました", str(error), parent=self.root)

    def _append_log(self, text: str) -> None:
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.insert(tk.END, text.rstrip() + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _open_output(self) -> None:
        try:
            path = Path(self.output_var.get()).expanduser().resolve()
            path.mkdir(parents=True, exist_ok=True)
            self._open_path(path)
        except OSError as exc:
            messagebox.showerror("フォルダを開けません", str(exc), parent=self.root)

    def _open_report(self) -> None:
        path = self.last_report_path or (PROJECT_ROOT / "review.html")
        if not path.is_file():
            messagebox.showinfo("レポート未作成", "スキャン後にレポートが作成されます。", parent=self.root)
            return
        webbrowser.open(path.resolve().as_uri())

    @staticmethod
    def _open_path(path: Path) -> None:
        if hasattr(os, "startfile"):
            os.startfile(path)  # type: ignore[attr-defined]
        else:
            webbrowser.open(path.resolve().as_uri())

    def _on_close(self) -> None:
        if not self._is_busy():
            self._close_session_logger()
            self.root.destroy()
            return
        if not messagebox.askyesno(
            "処理を停止しますか？",
            "現在の画像の判定が終わってから停止し、ウィンドウを閉じます。",
            parent=self.root,
        ):
            return
        self.closing = True
        self._stop()
        self.status_var.set("停止後に終了します")


def main() -> int:
    root = tk.Tk()
    DesktopApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
