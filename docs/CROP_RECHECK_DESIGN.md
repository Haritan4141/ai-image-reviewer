# 顔・手・足クロップ再判定 実装仕様

この文書は、現在の `ai-image-reviewer` に実装されている **CROP_RECHECK** の仕様です。全体画像（full）で見落としやすい顔・手・足を、必要な場合だけ拡大して再判定します。元画像を削除せず、不確実な結果は `REVIEW` に倒すことを優先します。

## 適用範囲と安全方針

- 全体判定は従来の `PASS / REVIEW / FAIL` と 5 項目のスコア（`anatomy`、`hands`、`face`、`artifacts`、`composition`）を使います。crop も同じ JSON 契約を使い、対象だけをプロンプトで限定します。
- full の `result` は local rules 適用 **後** の値です。モデルが最初に返した値は `model_result` に残します。
- crop の `PASS` だけで full `REVIEW` を `PASS` に戻しません。
- 検出失敗、未知、低信頼、crop 生成失敗、JSON 不正、途中キャンセルは `REVIEW` 寄りに扱います。
- 原画像の自動削除はありません。`move` を選んだ場合は既存の仕分け仕様に従って原画像を出力先へ移動します。`keep_crop_files: false` の場合だけ、原画像とは別に、この実行の`CropWorkspace`が作った一時cropファイルと空のrun/hashディレクトリをcleanupします。既存ファイルは削除しません。
- モデル応答の取込みは `ClassificationResult.from_model_mapping()` が5フィールド（`result`、`confidence`、`scores`、`problems`、`summary`）だけを受け付けます。モデルが混入させた`model_result`、`crop_checks`、pipeline metadata等は監査情報として信頼せず、保存済み記録の復元時だけ`from_mapping()`で読み込みます。

既存動作との互換性のため `crop_recheck.enabled` の既定値は `false` です。常用時は `balanced` を推奨します。

## 実装構成

```text
image
  └─ full VLM判定（local rules後の full result）
       ├─ full FAIL       → FAILを維持し、cropを省略
       ├─ crop不要        → full_onlyとして保存
       └─ crop対象        → VLMによる領域検出
                              ├─ face / hand / foot の候補を計画
                              ├─ CropWorkspaceで拡大画像を生成
                              ├─ target-aware VLM判定
                              └─ conservative merge → 最終 result
```

責務は次のファイルに分けています。

| ファイル | 現在の責務 |
|---|---|
| `src/models.py` | `RegionKind`、正規化・境界クランプする`CropBox`、`RegionCheckResult`、`ClassificationResult.crop_checks`とpipeline metadata |
| `src/prompts.py` | full／face／hand／foot／localizationのtarget-awareプロンプトと5スコアJSON契約 |
| `src/lmstudio_client.py` | LM Studioへのfull・crop・領域検出リクエスト |
| `src/codex_cli_client.py` | Codex CLI / GPT-5.6 Lunaへのfull・crop・領域検出リクエスト |
| `src/region_detection.py` | `RegionDetector`抽象、VLM detector、利用不能時の明示的なunknown結果 |
| `src/crop_utils.py` | EXIF補正、余白、最小サイズ、最大辺、hash/run単位のcrop workspaceと所有ファイルのcleanup |
| `src/crop_pipeline.py` | trigger、region plan、crop inspection、mergeとキャンセル状態 |
| `src/classifier.py` | full判定、既存local rules、crop pipeline呼び出し |
| `src/scanner.py` | image hash、仕分け、cache/crop配下の再帰取り込み防止、記録 |
| `src/report_builder.py` | JSONL、CSV、`review.html`へのfull／crop監査情報出力 |
| `src/config.py` | cropモード、planner、detector、targetの検証と絶対パス解決 |
| `config.yaml`、`main.py` | 無効既定値の設定例、CLIからのpipeline組み立てと進捗ログ |
| `src/gui_controller.py`、`src/desktop_app.py` | 独立したcrop有効化・モード・保持設定、部位間停止、進捗ログ |
| `.gitignore` | 標準の`cache/crops/`をGit対象外にする |

追加テストは`tests/test_crop_models.py`、`tests/test_crop_utils.py`、`tests/test_region_detection.py`、`tests/test_crop_pipeline.py`、`tests/test_crop_backend_pipeline.py`です。既存の`test_prompts.py`、`test_lmstudio_client.py`、`test_codex_cli_client.py`、`test_config.py`、`test_gui_controller.py`、`test_scanner.py`、`test_report_builder.py`も拡張しています。説明文書は本書に加えて`README.md`、`docs/SETUP.md`、`docs/COMMANDS.md`、`docs/ROADMAP.md`を更新しています。

## 検出器とバックエンド

`detectors.provider: auto`（既定）または `vlm` では、選択中の同じVLMの `locate_regions()` を使います。LM Studio用の別モデルや専用検出器を暗黙に追加しません。Codex CLIとLM Studioは同じ `classify_image(..., target="full"|"face"|"hand"|"foot", region_index=...)` 契約を使います。

cropの応答もfullと同じ5スコアJSON契約ですが、対象外のスコアはプロンプトで中立値`10`を指示するだけでなく、受信後のコードでも中立値`10`へ強制します。これにより、face／hand／footの局所確認が無関係な全体構図や別部位のスコアを誤って下げることを防ぎます。

`provider: none`、または選択中VLMが領域検出に対応しない場合も、アプリケーション自体は停止しません。領域が必要なのに検出できなかった事実を記録し、最終結果を `REVIEW` にします。固定グリッド、中央crop、未検出部位の推測boxは使用しません。

検出結果は次を区別します。

- 高信頼の `not_visible`: 画面外・遮蔽などで見えない部位。欠損とは扱わず、cropを要求しません。
- `unknown`、検出器不在、低信頼、異常なJSON: 本当に見えないと確認できないため、必要な対象が未確認として `REVIEW` にします。

## モードの実際の挙動

`crop_recheck.mode`（確認量）と `rules.mode`（全体・crop判定の厳しさ）は別設定です。

### Fast

追加コストを抑え、full-onlyに近づけます。full が `FAIL` ならcropを省略します。次のいずれかでcrop再判定を起動します。

- full が `REVIEW`
- full confidence が `0.90` 未満
- `trigger_low_scores` に指定したスコアがレビュー閾値未満
- `review_problem_keywords` に指定した問題語がある
- summaryに小さい顔・手、遠い人物などのヒントがある

`planner.run_on_review_only: true` を併用すると、full `REVIEW` だけに限定できます。

### Balanced（推奨）

full が `FAIL` でない画像では、人物がいる可能性がある場合に有効な face と hand を優先します。検出された人物が高信頼で「いない」場合は、人物部位のcropを要求しません。foot は次の条件でだけ追加します。

- full `anatomy` がレビュー閾値未満
- fullの問題語に `foot`、`feet`、`toe` などがある
- 検出されたfoot領域が `large_foot_area`（既定 `0.04`）以上

### Strict

人物がいる可能性がある場合、enabledなface・hand・footの可視候補を広く再確認します。必要な部位が未知・低信頼・未検出なら `REVIEW` にします。ただし検出器が高信頼で `not_visible` と返した部位は、遮蔽／画面外を欠陥とは扱いません。

対象設定の `upper_body` と `lower_body` は将来用の予約キーです。型と設定欄はありますが、現時点で有効化すると設定エラーになります。

## crop生成の境界

- EXIF回転を正規化し、透明画像は白背景へ合成してから座標を適用します。
- `CropBox` は正規化座標 `[x1, y1, x2, y2]` で、画像境界へクランプします。負の幅・高さは推測で修復せず無効にします。
- 既定のpaddingは`0.15`、最小幅・高さは`96px`、最大辺は`2048px`です。小さすぎる候補はcrop推論せず、根拠を記録します。
- 同じkindの候補はIoU `0.50`以上を重複としてまとめ、顔2・手4・足4までに制限します。手・足は左右固定名ではなく`kind + index`です。
- cropは`cache/crops/<content-hash>/run-<一意値>/`に生成し、元画像を変更しません。`keep_crop_files: true`なら保持します。`false`なら現在のworkspaceが作ったファイルと空のrun/hashディレクトリだけをcleanupし、既存ファイルは削除しません。
- 1画像あたりの推論回数上限は、再試行を除き **full 1 + localization 1 + face 2 + hand 4 + foot 4 = 最大12回** です。Fastや検出候補が少ない場合はこれより少なくなります。

## 統合（merge）ルール

1. full `FAIL` は基本的にそのまま `FAIL` とし、追加cropを行いません。
2. full `PASS` + crop `REVIEW`／`FAIL` は、少なくとも `REVIEW` へ引き上げます。
3. crop `FAIL` を全体 `FAIL`へ確定するには、crop confidenceとdetector confidenceが閾値以上で、fail keywordがあり、`score_fail_below`未満のスコアが`fail_score_count`以上必要です。条件が一つでも欠けるcrop FAILは `REVIEW` に戻します。
4. full `REVIEW` + crop `PASS` は `REVIEW` のままです。
5. 検出失敗、低信頼、JSON不正、crop生成失敗、未確認候補、キャンセルは `REVIEW` を維持・引き上げます。

`full_result_before_merge`、`crop_mode`、`pipeline_stage`、`pipeline_version`、`decision_source`を統合結果へ残します。各cropの`RegionCheckResult`にはkind、index、box、result、confidence、score/scores、問題、detector名・信頼度、保持したcrop pathを残します。診断用`raw`は通常のシリアライズ対象外です。

## 設定

`config.yaml`からの相対パスは設定ファイルのディレクトリを基準に解決されます。既定ではcrop再判定を無効にし、以下が有効化時の設定ツリーです。

```yaml
crop_recheck:
  enabled: false
  mode: balanced
  keep_crop_files: true
  crop_cache_dir: cache/crops
  planner:
    run_on_review_only: false
    min_confidence_for_skip: 0.90
    trigger_low_scores: [hands, face, anatomy]
    review_problem_keywords: [finger, hand, face, eye, foot, toe]
    max_hand_crops: 4
    max_face_crops: 2
    max_foot_crops: 4
    min_crop_size: 96
    crop_padding_ratio: 0.15
    dedup_iou: 0.50
    min_detector_confidence: 0.80
    large_foot_area: 0.04
  detectors:
    provider: auto       # auto / vlm / none
    allow_fallback: true
    detector_failure_policy: review
    person_required_for_balanced: true
  targets:
    face: {enabled: true}
    hand: {enabled: true}
    foot: {enabled: true}
    upper_body: {enabled: false}  # 予約。trueは設定エラー
    lower_body: {enabled: false}  # 予約。trueは設定エラー
```

## ログとレポート

`results.jsonl`は1画像1レコードで、既存のfull結果に次を追加します。

- `model_result`、`final_result`、`decision_source`
- `pipeline_stage`、`pipeline_version`、`crop_mode`、`full_result_before_merge`
- `crop_checks[]`（kind、index、box、result、confidence、score/scores、problems、summary、detector情報、crop path）

`latest_summary.csv`は`crop_checks`をJSON文字列としてUTF-8 BOM付きで保存します。`review.html`は元画像カードにfull／final、モード、pipeline version、`[Face 0]`／`[Hand 0]`／`[Foot 0]`ごとの結果・信頼度・理由・detectorを表示します。保持され、かつ実在するcropだけサムネイルを表示し、`keep_crop_files: false`などで消えたpathには壊れた画像リンクを作りません。元画像・crop pathはローカル監査用の情報なので、HTMLやログを公開場所へ置かないでください。

## 再評価・停止・再開

処理済みcache（`cache/processed.json`）は画像hashだけを保持し、設定・モデル・cropごとの推論結果はキャッシュしません。そのため、判定基準、crop mode、プロンプト、モデルを変更した場合はGUIの「処理済み画像の再判定」を有効にするか、次を実行します。

```powershell
python main.py --config config.local.yaml scan --path 'D:\images\incoming' --force
python main.py --config config.local.yaml rescan-review
```

`--force`が無視するのは画像hash抑止だけです。現在の実装にはcrop単位のresume／inference cacheはありません。GUI停止や途中キャンセルでは、処理済み画像のpipeline記録（`pipeline_stage: cancelled`、未確認理由を含む）が残ることがあります。その画像を再評価する場合もGUI forceまたは`scan --force`を使ってください。未処理の画像は通常のscanで追いつけます。

## 既知の制約と検証状況

- VLMによる座標は近似です。専用face/hand/foot detectorはまだ導入しておらず、追加依存なしで動く構成を優先しています。
- 実画像での検出precision/recallや最終判定の校正値は未取得です。現時点の自動テストはstub／mock中心で、live VLMの品質を保証するテストではありません。実運用前に`test-codex`または`test-lmstudio`と少数画像の目視確認を行ってください。
- 大量画像、UNC、低VRAM、タイムアウト、推論枠の制約では処理時間と失敗率が増えます。上限、retry、`keep_crop_files`、ログ容量を事前に確認してください。
- 上位／下位bodyは未実装です。自動削除、crop単位の再開、専用detectorの選択は今後の候補です。

詳細な初回設定と安全なコマンド例は [SETUP.md](SETUP.md) と [COMMANDS.md](COMMANDS.md) を参照してください。
