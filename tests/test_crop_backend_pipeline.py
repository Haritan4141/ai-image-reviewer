"""End-to-end local pipeline tests; only the external transports are faked."""

import json
from pathlib import Path
import subprocess

import pytest
from PIL import Image

from src.classifier import ImageClassifier
from src.codex_cli_client import CodexCLIClient
from src.config import load_config
from src.lmstudio_client import LMStudioClient
from src.models import ClassificationResult, ScoreSet
from src.report_builder import ReportBuilder
from src.scanner import ImageScanner
from src.sorter import ImageSorter
from src.utils import sha256_file


@pytest.mark.parametrize("backend", ["lmstudio", "codex_cli"])
def test_both_real_clients_support_full_detection_crops_sort_and_report(tmp_path: Path, backend: str):
    source_dir = tmp_path / "incoming"
    source_dir.mkdir()
    source = source_dir / "image.png"
    Image.new("RGB", (1000, 1000), "green").save(source)
    digest = sha256_file(source)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "watch:\n  paths: [incoming]\n  file_stable_seconds: 0\n"
        f"classifier:\n  backend: {backend}\n"
        "crop_recheck:\n  enabled: true\n  mode: strict\n"
        "lmstudio:\n  retries: 0\ncodex_cli:\n  retries: 0\n", encoding="utf-8",
    )
    cfg = load_config(config_path)
    good = {"result": "PASS", "confidence": .97, "scores": ScoreSet(9, 9, 9, 9, 9).to_dict(),
            "problems": [], "summary": "A person is visible."}
    review = {**good, "result": "REVIEW", "problems": ["possible fused finger"], "summary": "Check visible fingers."}
    detection = {
        "person_present": True, "confidence": .97, "not_visible": [], "summary": "Visible face, hand and foot.",
        "regions": [
            {"kind": "face", "box": [.1, .1, .3, .3], "confidence": .95},
            {"kind": "hand", "box": [.5, .3, .7, .5], "confidence": .95},
            {"kind": "foot", "box": [.4, .8, .6, 1], "confidence": .95},
        ],
    }
    responses = [good, detection, good, review, good]
    calls = []

    class Response:
        status_code = 200
        def __init__(self, content):
            self.content = content
        def json(self):
            return {"choices": [{"message": {"content": json.dumps(self.content)}}]}

    class Session:
        def post(self, url, **kwargs):
            calls.append(kwargs["json"]["messages"])
            return Response(responses.pop(0))

    def runner(args, **kwargs):
        if list(args[1:]) == ["--version"]:
            return subprocess.CompletedProcess(args, 0, "codex-cli test", "")
        if list(args[1:]) == ["login", "status"]:
            return subprocess.CompletedProcess(args, 0, "Logged in using ChatGPT", "")
        assert args[1] == "exec"
        schema = json.loads(Path(args[args.index("--output-schema") + 1]).read_text(encoding="utf-8"))
        assert ("regions" in schema["properties"]) == (len(calls) == 1)
        calls.append(kwargs["input"])
        return subprocess.CompletedProcess(args, 0, json.dumps(responses.pop(0)), "")

    client = LMStudioClient(cfg, session=Session()) if backend == "lmstudio" else CodexCLIClient(cfg, runner=runner, executable_path="fake-codex")
    classifier = ImageClassifier(client, crop_config=cfg.crop_recheck)
    report = ReportBuilder(config=cfg)
    scanner = ImageScanner(source_dir, classifier, ImageSorter(config=cfg, source_roots=[source_dir]),
                           config=cfg, report_builder=report, stable_seconds=0)
    records = scanner.scan()
    report.build()
    assert len(calls) == 5
    assert not responses
    assert len(records) == 1
    record = records[0]
    assert record.result == "REVIEW"
    assert record.model_result == "PASS"
    assert record.full_result_before_merge == "PASS"
    assert [check["result"] for check in record.crop_checks] == ["PASS", "REVIEW", "PASS"]
    assert Path(record.destination_path).parent.name == "review"
    assert sha256_file(source) == digest == sha256_file(record.destination_path)
    restored = ClassificationResult.from_mapping(json.loads(cfg.results_jsonl_path.read_text(encoding="utf-8").splitlines()[0]))
    assert len(restored.crop_checks) == 3
    assert restored.result.value == "REVIEW"
    html = cfg.report_path.read_text(encoding="utf-8")
    assert "[Hand 0]" in html
    assert "hand-0.png" in html
    assert "possible fused finger" in html
