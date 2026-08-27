"""Command-line entry point for ai-image-reviewer."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Sequence

from src.classifier import ImageClassifier, LocalRulesConfig
from src.codex_cli_client import CodexCLIClient, CodexCLIError
from src.config import AppConfig, ConfigError, load_config
from src.file_watcher import ImageWatcher
from src.lmstudio_client import LMStudioClient, LMStudioError
from src.logger import setup_logging
from src.report_builder import ReportBuilder
from src.scanner import ImageScanner, ScanRecord
from src.sorter import ImageSorter


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-image-reviewer",
        description="LM StudioまたはCodex CLIで生成画像をPASS / REVIEW / FAILに仕分けします。",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="設定ファイル（既定: プロジェクト直下のconfig.yaml）",
    )
    parser.add_argument("--verbose", action="store_true", help="詳細ログを表示します")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan = subparsers.add_parser("scan", help="既存画像を一括処理します")
    scan.add_argument(
        "--path",
        action="append",
        dest="paths",
        help="処理対象フォルダまたは画像。複数回指定可。省略時はwatch.pathsを使用します",
    )
    scan.add_argument(
        "--force",
        action="store_true",
        help="処理済みハッシュを無視して再判定します",
    )

    watch = subparsers.add_parser("watch", help="新規・更新画像を継続監視します")
    watch.add_argument(
        "--path",
        action="append",
        dest="paths",
        help="監視対象を一時的に上書きします。複数回指定可",
    )

    subparsers.add_parser("rescan-review", help="output/review内を強制再判定します")
    subparsers.add_parser("build-report", help="保存済みログからCSVとreview.htmlを再生成します")
    subparsers.add_parser("test-lmstudio", help="LM Studio APIと設定モデルを確認します")
    subparsers.add_parser("test-codex", help="Codex CLIとChatGPT認証を確認します（モデル呼び出しなし）")
    return parser


def _paths(values: Sequence[str] | None, config: AppConfig) -> tuple[Path, ...]:
    if not values:
        return config.watch.paths
    return tuple(Path(value).expanduser().resolve() for value in values)


def _validate_inputs(paths: Sequence[Path], *, allow_files: bool = True) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        formatted = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"入力パスが見つかりません: {formatted}")
    if not allow_files:
        files = [path for path in paths if not path.is_dir()]
        if files:
            formatted = ", ".join(str(path) for path in files)
            raise NotADirectoryError(f"監視対象はフォルダを指定してください: {formatted}")


AnalysisClient = LMStudioClient | CodexCLIClient


def _make_client(config: AppConfig) -> AnalysisClient:
    if config.classifier.backend == "codex_cli":
        client = CodexCLIClient(config)
        # Enforce the ChatGPT-auth guard before processing any image. This is
        # intentionally fail-fast so an API-key session cannot incur API fees.
        client.get_status()
        return client
    return LMStudioClient(config)


def _make_pipeline(
    config: AppConfig,
    roots: Sequence[Path],
    logger: logging.Logger,
) -> tuple[AnalysisClient, ImageScanner, ReportBuilder]:
    client = _make_client(config)
    if config.classifier.backend == "codex_cli":
        logger.info(
            "判定バックエンド: Codex CLI / model=%s / reasoning=%s / mode=%s / ChatGPT認証必須=%s",
            config.codex_cli.model,
            config.codex_cli.reasoning_effort,
            config.rules.mode,
            config.codex_cli.require_chatgpt_login,
        )
    else:
        logger.info(
            "判定バックエンド: LM Studio / model=%s / mode=%s / url=%s",
            config.lmstudio.model,
            config.rules.mode,
            config.lmstudio.base_url,
        )
    classifier = ImageClassifier(client, LocalRulesConfig.from_settings(config.rules),
                                 crop_config=config.crop_recheck, logger=logger)
    logger.info("クロップ再判定: enabled=%s / mode=%s / detector=%s",
                config.crop_recheck.enabled, config.crop_recheck.mode, config.crop_recheck.detectors.provider)
    sorter = ImageSorter(config=config, source_roots=roots)
    report = ReportBuilder(config=config)
    scanner = ImageScanner(
        roots=roots,
        classifier=classifier,
        sorter=sorter,
        config=config,
        report_builder=report,
        logger=logger,
    )
    return client, scanner, report


def _print_summary(records: Sequence[ScanRecord], logger: logging.Logger) -> None:
    counts = {"PASS": 0, "REVIEW": 0, "FAIL": 0, "error": 0}
    for record in records:
        if record.status == "error":
            counts["error"] += 1
        if record.result in counts:
            counts[record.result] += 1
    logger.info(
        "処理 %d件: PASS=%d REVIEW=%d FAIL=%d error=%d",
        len(records),
        counts["PASS"],
        counts["REVIEW"],
        counts["FAIL"],
        counts["error"],
    )


def command_scan(
    config: AppConfig,
    logger: logging.Logger,
    values: Sequence[str] | None,
    *,
    force: bool,
) -> int:
    roots = _paths(values, config)
    _validate_inputs(roots)
    client, scanner, report = _make_pipeline(config, roots, logger)
    try:
        records = scanner.scan(force=force)
        generated = report.build()
    finally:
        client.close()
    _print_summary(records, logger)
    logger.info("レポート: %s", generated["html"])
    return 1 if any(record.status == "error" for record in records) else 0


def command_rescan_review(config: AppConfig, logger: logging.Logger) -> int:
    review_dir = config.output.directory / "review"
    _validate_inputs((review_dir,), allow_files=False)
    client, scanner, report = _make_pipeline(config, (review_dir,), logger)
    try:
        records = scanner.rescan_review(review_dir)
        generated = report.build()
    finally:
        client.close()
    _print_summary(records, logger)
    logger.info("レポート: %s", generated["html"])
    return 1 if any(record.status == "error" for record in records) else 0


def command_watch(
    config: AppConfig,
    logger: logging.Logger,
    values: Sequence[str] | None,
) -> int:
    roots = _paths(values, config)
    _validate_inputs(roots, allow_files=False)
    client, scanner, report = _make_pipeline(config, roots, logger)

    def process(path: Path, event: str = "created") -> None:
        record = scanner.process_file(path)
        if record is not None:
            logger.info("%s -> %s (%s)", path, record.result, event)
            report.build()

    watcher = ImageWatcher(
        roots=roots,
        callback=process,
        config=config,
        logger=logger,
        # An explicit initial scan below handles both watcher backends equally.
        include_existing=False,
    )
    try:
        initial = scanner.scan()
        report.build()
        _print_summary(initial, logger)
        logger.info(
            "監視開始: backend=%s paths=%s（終了: Ctrl+C）",
            watcher.active_backend,
            ", ".join(str(path) for path in roots),
        )
        watcher.run_forever()
    finally:
        watcher.stop()
        client.close()
    logger.info("監視を終了しました")
    return 0


def command_build_report(config: AppConfig, logger: logging.Logger) -> int:
    generated = ReportBuilder(config=config).build()
    logger.info("JSONL: %s", generated["jsonl"])
    logger.info("CSV: %s", generated["csv"])
    logger.info("HTML: %s", generated["html"])
    return 0


def command_test_lmstudio(config: AppConfig, logger: logging.Logger) -> int:
    client = LMStudioClient(config)
    try:
        models = client.get_models()
    except LMStudioError as exc:
        logger.error("LM Studioへ接続できません: %s", exc)
        return 1
    finally:
        client.close()

    model_ids = [str(item.get("id", "")) for item in models if isinstance(item, dict)]
    logger.info("LM Studio APIへ接続できました（モデル %d件）", len(model_ids))
    if model_ids:
        logger.info("利用可能モデル: %s", ", ".join(model_ids))
    if config.lmstudio.model not in model_ids:
        logger.error("設定モデル '%s' がモデル一覧にありません", config.lmstudio.model)
        return 2
    logger.info("設定モデル '%s' を確認しました", config.lmstudio.model)
    return 0


def command_test_codex(config: AppConfig, logger: logging.Logger) -> int:
    client = CodexCLIClient(config)
    try:
        status = client.get_status(refresh=True)
    except CodexCLIError as exc:
        logger.error("Codex CLIを利用できません: %s", exc)
        return 1
    logger.info("Codex CLI: %s", status["version"])
    logger.info("認証方式: %s", status["authentication"])
    logger.info("設定モデル: %s", status["model"])
    if status["authentication"] == "chatgpt":
        logger.info("ChatGPTサブスクリプション認証を確認しました（APIキー課金経路ではありません）")
    else:
        logger.warning("ChatGPT認証ではありません。設定によりAPI料金が発生する可能性があります")
    return 0


def run(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        config.ensure_directories()
        logger = setup_logging(config.application_log_path, args.verbose)
        if args.command == "scan":
            return command_scan(config, logger, args.paths, force=args.force)
        if args.command == "watch":
            return command_watch(config, logger, args.paths)
        if args.command == "rescan-review":
            return command_rescan_review(config, logger)
        if args.command == "build-report":
            return command_build_report(config, logger)
        if args.command == "test-lmstudio":
            return command_test_lmstudio(config, logger)
        if args.command == "test-codex":
            return command_test_codex(config, logger)
        raise AssertionError(f"unknown command: {args.command}")
    except (CodexCLIError, ConfigError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n中断しました。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(run())
