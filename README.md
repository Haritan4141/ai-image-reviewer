# ai-image-reviewer

Stable Diffusion WebUI や ComfyUI で大量生成した画像を、ChatGPT Proで認証したCodex CLIまたはローカルのLM Studioで検品し、`PASS` / `REVIEW` / `FAIL` に整理するWindows 11向けツールです。

目的は「自動削除」ではありません。問題がなさそうな画像を `PASS` に集め、怪しい画像を `REVIEW` に残して人が確認できるようにします。`FAIL` も仕分け先へコピーまたは移動するだけで、元画像を直接削除しません。初回運用では `copy` モードと広めの `REVIEW` 判定を推奨します。

## 主な機能

- ローカルパスと UNC パス（`\\server\share\images`）の一括スキャン・監視
- `.png` / `.jpg` / `.jpeg` / `.webp`、サブフォルダ、ポーリング監視
- ChatGPT Proの利用枠でGPT-5.6 Lunaを呼び出すCodex CLIバックエンド
- LM StudioのOpenAI互換APIを使う完全ローカルバックエンド
- VLM の JSON 判定に加えたローカル側の安全ルール
- `output/pass`、`output/review`、`output/fail` へのコピーまたは移動
- `logs/results.jsonl`、`logs/latest_summary.csv` への結果保存
- サムネイル付き静的 `review.html`（結果・信頼度・問題点を表示、結果で絞り込み）
- `review`だけの再判定、各バックエンドの接続テスト、レポート再生成
- 処理済み判定による重複処理の抑止

## 動作要件

- Windows 11（Windows PowerShell 5.1 または PowerShell 7）
- Python 3.11（64-bit 推奨）
- 既定構成: Codex CLIとChatGPT Plus/Pro認証（モデルは`gpt-5.6-luna`）
- 代替構成: LM Studioと画像入力に対応するVLM（Qwen3-VL系など）
- VLM を GPU で動かす場合は VRAM 24 GB 級（例: RTX 3090）を推奨
- 監視元が別 PC の場合は、検品 PC から UNC 共有へ読み書きできること

セットアップの詳細は [docs/SETUP.md](docs/SETUP.md) を参照してください。

## クイックスタート

PowerShell でプロジェクトディレクトリを開きます。

```powershell
Set-Location 'C:\Users\<ユーザー名>\Documents\ai-image-reviewer'
py -3.11 -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

Codex CLIをChatGPTアカウントで認証し、`config.yaml`の監視対象と出力先を確認します。`config.yaml`はプロジェクトルートから相対パスを解決するサンプルを兼ねています。既存の設定を保全したい場合は、別名にコピーして`--config`で指定します。

```powershell
Copy-Item config.yaml config.local.yaml
notepad config.local.yaml
codex login status
python main.py --config config.local.yaml test-codex
python main.py --config config.local.yaml scan --path 'D:\images\batch1'
python main.py --config config.local.yaml build-report
```

結果は通常、`output\pass`、`output\review`、`output\fail`、`logs\` に作成されます。`review.html` をブラウザで開いて確認してください。

```powershell
Start-Process .\review.html
```

継続監視は次のコマンドです。停止は `Ctrl+C` です。

```powershell
python main.py watch
```

コマンドのオプションと例は [docs/COMMANDS.md](docs/COMMANDS.md) を参照してください。

## Codex CLIバックエンド（既定）

`classifier.backend: codex_cli`では、Pythonが画像ごとに`codex exec`を非対話実行します。画像を`--image`で添付し、`--output-schema`で判定JSONを固定します。OpenAI互換HTTPプロキシやAPIキーは不要です。

```powershell
codex --version
codex login
codex login status
python main.py --config config.yaml test-codex
```

`codex login status`が`Logged in using ChatGPT`になっていることを確認してください。既定の`require_chatgpt_login: true`は、APIキー認証を検出した時点で画像送信前に停止します。これによりOpenAI PlatformのAPI料金経路へ意図せず切り替わることを防ぎます。ChatGPT Pro自体の利用上限と、購入済みChatGPTクレジットの消費は別途適用されます。

実行は`read-only`サンドボックス、エフェメラルセッション、ユーザー設定を読み込まない構成です。モデルには画像だけを分析し、ファイル探索やコマンド実行を行わないよう指示します。判定画像はOpenAIへ送信されるため、完全ローカル処理が必要な画像にはLM Studioバックエンドを使用してください。

公式仕様: [Codex non-interactive mode](https://learn.chatgpt.com/docs/non-interactive-mode)、[Codex authentication](https://learn.chatgpt.com/docs/auth)、[ChatGPT/Codex pricing](https://learn.chatgpt.com/docs/pricing)

## LM Studioバックエンド（ローカル代替）

1. LM Studio を検品 PC にインストールして起動します。
2. 画像を受け付ける VLM をダウンロードし、モデル名（LM Studio の表示名または API が返す ID）を確認します。
3. LM Studio の Developer / Local Server 画面で API サーバーを有効化します。
4. 既定の API ベース URL（例: `http://127.0.0.1:1234/v1`）と、設定ファイルのモデル名を一致させます。
5. `classifier.backend: lmstudio`へ切り替え、`python main.py --config config.yaml test-lmstudio`で確認します。

LM Studio の画面名やポートはバージョンで変わるため、実際に表示される Local Server / Developer 設定を確認してください。API を LAN に公開する必要は通常ありません。別 PC から LM Studio を呼ぶ構成にする場合は、ファイアウォール・認証・アクセス元を限定し、WAN には公開しないでください。

`Qwen3-VL-8B` 系を使う場合でも、量子化形式・コンテキスト長・画像入力対応はモデルごとに異なります。まず 1 枚で接続テストと小規模 `scan` を実行し、JSON として受け付けられることを確認してください。

## 設定ファイル

設定の実フィールドは次のとおりです。未指定項目には実装の既定値が使われます。

```yaml
watch:
  paths:
    - 'D:\images\incoming'
    # - '\\DESKTOP-EXAMPLE\generated\images'
  recursive: true
  mode: polling                    # polling または watchdog
  polling_interval_seconds: 5
  file_stable_seconds: 2

output:
  directory: output
  operation: copy                  # 初回は copy。move は検証後
  preserve_relative_paths: true

classifier:
  backend: codex_cli              # codex_cli または lmstudio

codex_cli:
  executable: codex
  model: gpt-5.6-luna
  reasoning_effort: low
  timeout_seconds: 180
  retries: 1
  retry_delay_seconds: 2
  max_image_dimension: 2048
  jpeg_quality: 90
  working_directory: cache/codex-cli
  require_chatgpt_login: true     # APIキー認証なら送信前に停止
  ignore_user_config: true
  ephemeral: true

# classifier.backend: lmstudio の場合に使用
lmstudio:
  base_url: http://127.0.0.1:1234/v1
  model: qwen3-vl-8b
  timeout_seconds: 120
  retries: 2
  retry_delay_seconds: 2
  max_image_dimension: 2048
  jpeg_quality: 90

processing:
  parallel_workers: 1
  extensions: [.png, .jpg, .jpeg, .webp]

rules:
  threshold_pass: 0.85
  threshold_review: 0.50
  score_review_below: 6
  score_fail_below: 3
  fail_score_count: 2
  review_problem_keywords: [extra finger, missing finger, fused finger, deformed hand]
  fail_problem_keywords: [severe deformation, major anatomy failure]

logs:
  directory: logs
  results_jsonl: results.jsonl
  summary_csv: latest_summary.csv
  application_log: app.log

cache:
  directory: cache
  processed_file: processed.json
  use_content_hash: true

report:
  filename: review.html
  thumbnail_width: 320
  thumbnail_height: 320
```

相対パスは `config.yaml` のあるディレクトリ基準で解決されます。Windows パスは YAML の単一引用符で囲むのが安全です。`threshold_review` は `threshold_pass` 以下、スコアの閾値は 1〜10 の範囲で指定します。

## 判定と安全策

VLM には、手・指・顔・四肢・人体・余分なパーツ・融合・生成ノイズなどを厳しめに検査し、JSON のみ返すよう要求します。期待する結果の形は次のとおりです。

```json
{
  "result": "PASS",
  "confidence": 0.93,
  "scores": {
    "anatomy": 9,
    "hands": 8,
    "face": 9,
    "artifacts": 9,
    "composition": 8
  },
  "problems": [],
  "summary": "No obvious issues found."
}
```

`result` は `PASS`、`REVIEW`、`FAIL` のいずれか、`confidence` は 0〜1、`scores` は 1〜10 の整数を想定します。JSON 以外の応答は補正・再試行の対象になります。JSON が欠落または不正な場合は、安全側に `REVIEW` として扱います。

ローカルルールにより、VLM の `FAIL` は `FAIL` のまま、`REVIEW` は `REVIEW` のまま扱います。`PASS` でも信頼度が閾値未満、重大な問題語、低スコアがある場合は `REVIEW` 以上へ引き上げます。しきい値は `rules` で調整します。

## UNC パスの注意

- 検品を実行する Windows ユーザーが、対象共有の読み取り権限と、仕分け先の書き込み権限を持つことを確認してください。
- エクスプローラーで接続済みでも、タスクスケジューラや別ユーザーの PowerShell からは見えない場合があります。実際にツールを起動するユーザーで `Test-Path` と `Get-ChildItem` を確認してください。
- UNC を YAML に書くときは単一引用符で囲みます。末尾の `\` は付けず、共有名以降の相対フォルダを明示してください。
- 生成中の一時ファイルに備え、`watch.file_stable_seconds` を設定します。ネットワーク切断・スリープ時はログを確認してから再開してください。
- `move` モードでは共有元からの移動権限と再試行失敗時の扱いを先に確認します。最初は `copy` で実データを保全してください。
- SMB は LAN 内に限定し、TCP 445 や SSH を WAN に公開しないでください。

## よくあるトラブル

### `test-codex`が失敗する

`codex --version`と`codex login status`を同じWindowsユーザーで確認します。未ログインなら`codex login`を実行してChatGPT認証を選びます。APIキー認証では、既定の料金保護ガードによりスキャンを開始しません。

```powershell
codex login
codex login status
python main.py --config config.yaml test-codex
```

タスクスケジューラや別ユーザーから実行すると、対話ユーザーのCodex認証キャッシュを利用できない場合があります。まず通常のPowerShellから確認してください。

### `test-lmstudio` が接続できない

LM Studio が起動中か、Local Server が有効か、`base_url` のポートと `/v1` の有無が合っているかを確認します。

```powershell
Test-NetConnection 127.0.0.1 -Port 1234
Invoke-RestMethod 'http://127.0.0.1:1234/v1/models'
```

モデル ID が API の一覧と異なる場合は `lmstudio.model` を修正します。画像入力非対応モデルでは接続できても検品に失敗します。

### 画像が処理されない／ずっと `REVIEW` になる

拡張子、読み取り権限、対象パス、`logs/app.log` の API 応答を確認します。まず `parallel_workers: 1`、`retries: 0`、小さなフォルダで再現させます。VLM が JSON 以外の説明を返す場合は画像対応版にし、JSON 補正ログを確認してください。

### UNC が見えない

`Test-Path '\\server\share\folder'`、`Get-ChildItem` を同じユーザーで実行します。共有・NTFS の両方の権限を確認し、可能ならネットワークドライブ文字ではなく UNC を直接指定してください。

### 仕分け先へコピー／移動できない

出力先が入力フォルダの配下になっていないか、ファイルが別プロセスでロックされていないか、同名ファイルが存在しないかを確認します。`copy` で書き込み権限を確認してから `move` に切り替えます。

### `review.html` の画像が表示されない

レポートをプロジェクトルートで生成し、`review.html` をその場所から開いてください。相対パスのため HTML だけを別フォルダへコピーすると画像が見えない場合があります。必要なら簡易 HTTP サーバーを使います。

```powershell
python -m http.server 8000
Start-Process 'http://127.0.0.1:8000/review.html'
```

### 監視中に同じ画像が重複処理される

生成アプリの一時ファイルや複数イベントで重複することがあります。`file_stable_seconds`、ポーリング間隔、`cache.use_content_hash` を確認し、`watch` を二重起動しないでください。

## 出力とバックアップ

```text
output/
  pass/
  review/
  fail/
logs/
  results.jsonl
  latest_summary.csv
  app.log
cache/
  processed.json
review.html
```

JSONL には元画像のパス、相対パス、判定、信頼度、スコア、問題点、要約、処理時刻などを 1 行 1 件で保存します。ログと元画像は定期的にバックアップしてください。`move` を使う場合は特に、バックアップなしでの一括実行を避けます。

## 開発・テスト

```powershell
& .\.venv\Scripts\Activate.ps1
python -m pytest -q
```

自動テストはCodex、LM Studio、実ファイル共有へ接続せず、設定検証・認証ガード・コマンド組み立て・JSON正規化・仕分け・レポートを確認します。実バックエンドの確認は`test-codex`または`test-lmstudio`と、少数画像の手動スキャンで行ってください。

顔・手などのクロップ再判定の方針は [docs/CROP_RECHECK_DESIGN.md](docs/CROP_RECHECK_DESIGN.md)、優先順位は [docs/ROADMAP.md](docs/ROADMAP.md) を参照してください。
