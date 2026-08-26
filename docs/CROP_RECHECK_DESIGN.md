# 顔・手クロップ再判定 設計メモ

この文書は、全体画像の判定後に顔や手の領域を拡大して再確認するための設計方針です。初版ではフル画像だけでも動作することを優先し、クロップ処理を必須依存にしません。

## 目的

全体画像では小さく見える手・指・目・口などの破綻を拡大し、見逃しを減らします。一方、検出器が誤った領域を切り出すと誤検知が増えるため、クロップ結果だけで PASS を確定しません。

基本方針:

- 全体画像を一次判定（`pipeline_full_image`）
- 必要な場合だけ顔／手クロップを追加判定
- クロップが失敗・空・不確実な場合は `REVIEW` へ倒す
- クロップ判定で重大な破綻が出たら全体結果を `REVIEW` または `FAIL` へ引き上げる
- 元画像、クロップ画像、判定、モデル、設定を追跡可能にする

## 推奨パイプライン

```text
image
  │
  ├─ pipeline_full_image ── full_result
  │                            │
  │                            ├─ 明確な FAIL → FAIL
  │                            ├─ 低信頼／人物あり／手や顔が小さい
  │                            │       └─ crop planner
  │                            └─ 高信頼 PASS（条件を満たす）→ PASS候補
  │
  └─ crop planner
       ├─ face detector → pipeline_face_crop → face_result
       └─ hand detector → pipeline_hand_crop → hand_result（左右・複数）
                                      │
                         result merger / safety rules
                                      │
                           最終 PASS / REVIEW / FAIL
```

クロップ判定の入口は、実装上次のように分割できる構造を想定します。

```python
def pipeline_full_image(image_path: Path, context: RunContext) -> Verdict: ...
def pipeline_face_crop(crop: Crop, context: RunContext) -> Verdict: ...
def pipeline_hand_crop(crop: Crop, context: RunContext) -> Verdict: ...
def merge_crop_verdicts(full: Verdict, crops: list[Verdict], policy: CropPolicy) -> Verdict: ...
```

実際の型名・関数名は実装に合わせてよく、重要なのは「全体」「顔」「手」「統合」を別責務にすることです。

## クロップ生成

### 検出器

候補の順序は次のとおりです。

1. 既存の画像プロセッサ／軽量な顔・手検出器
2. VLM に全体画像を見せて概略 bounding box を返させる（低信頼扱い）
3. 固定グリッド／中心クロップによるフォールバック（検出器なし）

検出器の追加は大きなモデル依存を招くため、初版の必須依存にはしません。座標は常に元画像の幅・高さで正規化し、画像境界へクランプします。

### 品質条件

- 最小幅・高さ未満のクロップは破棄
- 余白（例: 10〜30%）を加え、関節・手首・顔周辺を切り落とさない
- アスペクト比と最大ピクセル数を制限し、API の payload を抑える
- 同じ領域の重複クロップを IoU で統合
- EXIF 回転を正規化してから座標を適用
- 一時クロップは `cache/crops/<content-hash>/` などへ置き、元画像を上書きしない

## 追加判定の条件

毎回すべてを再判定すると時間とコストが増えるため、次のようなトリガーを設定化します。

- `full_result == REVIEW`
- full の `confidence` が閾値未満
- `hands` または `face` スコアが低い
- 問題語に `hand`、`finger`、`face`、`eye` が含まれる
- 人物検出あり、かつ対象領域が小さい
- ユーザーが `rescan-review --crops` を明示した

明確な full `FAIL` は通常クロップを省略できます。`FAIL` の根拠を詳しくしたい運用ではオプションで実行します。

## 統合ルール（安全側）

例として、次の優先順位を推奨します。

1. full または crop に明確な重大破綻 → `FAIL`
2. crop が `REVIEW`、低信頼、検出失敗、JSON 不正 → `REVIEW`
3. full の `PASS` でも、クロップの anatomy/hands/face が閾値未満 → `REVIEW`
4. full と全クロップが `PASS` かつ最低信頼度が閾値以上 → `PASS`

クロップの一部だけを見て `PASS` に引き上げてはいけません。手が検出されなかったことは「手に問題がない」証拠ではないため、検出失敗は `REVIEW` とします。

### 重大破綻の例

- 明らかな extra fingers / missing fingers
- fused hand/body parts
- extra limb、欠落した四肢、関節の大きな崩れ
- 顔の複数目・目の位置異常・顔の融合
- クロップ間で同一人物の構造が大きく矛盾

問題語は英語・日本語・モデル固有の表記揺れを正規化して評価します。文字列一致だけに依存せず、`scores` と `confidence` も併用します。

## API とログの拡張

既存のフル画像用 JSON 契約を拡張し、クロップ情報を追加します。

```json
{
  "result": "REVIEW",
  "confidence": 0.78,
  "scores": {"anatomy": 7, "hands": 5, "face": 8, "artifacts": 9, "composition": 8},
  "problems": ["possible fused fingers in right hand"],
  "summary": "Full image is usable, but the right hand needs review.",
  "crop_checks": [
    {"kind": "hand", "index": 0, "box": [0.62, 0.40, 0.83, 0.76], "result": "REVIEW", "confidence": 0.58}
  ]
}
```

実装では `results.jsonl` の 1 レコード内に次を残します。

- 元画像の絶対／相対パスと content hash（可能なら）
- クロップの種類、座標、生成時刻、検出器名・バージョン
- 各クロップの API モデル、プロンプトバージョン、判定 JSON
- 統合ポリシーのバージョンと最終判定
- 失敗理由（検出なし、API タイムアウト、JSON 不正など）

`review.html` では元画像の横にクロップを表示し、どの領域が REVIEW の原因かを確認できるようにします。クロップ画像に個人情報や不要な一時ファイルが含まれないよう、保持期間を設定可能にします。

## 性能と安定性

- まず full 判定、必要な画像だけ crop 判定とする
- 3090 24 GB では VLM の量子化・画像解像度・コンテキスト長を実測する
- API の並列数を低く開始し、VRAM 不足・タイムアウト・429 相当の応答を監視
- 同じ元画像・同じ crop 設定・同じモデルの結果をキャッシュできるようキーを設計
- UNC ではクロップをローカル一時領域へコピーしてから処理し、共有切断による半端なファイルを避ける
- 途中停止から再開できるよう、crop ごとの完了状態を保存する

## 段階導入

1. クロップなしの full 判定を安定稼働させる
2. 検出器なしの固定手動クロップを開発用で検証する
3. 顔／手検出器を 1 種類ずつ導入し、誤検知率を測る
4. REVIEW 画像だけに crop を適用する
5. 人手ラベルで統合閾値を調整し、モデル／プロンプトのバージョンをログに残す

自動削除はこの設計の対象外です。クロップ判定が不確実な場合は、元画像を保持したまま `REVIEW` に送ることを原則とします。
