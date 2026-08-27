# 初回セットアップ手順（Windows 11）

検品専用 PC に `ai-image-reviewer` を初めて導入するときの手順です。最初は元画像を保持する `copy` と少数ファイルで確認し、動作を確認してから監視や `move` に進みます。

## 1. 事前確認

- Windows 11 と Python 3.11（64-bit）がインストールされている
- Codex CLIをChatGPT Plus/Proアカウントで利用できる（既定）
- またはLM Studioで画像入力対応VLMを実行できる（ローカル代替）
- 対象画像フォルダへ読み取り、出力先へ書き込みできる
- UNC を使う場合、検品 PC から共有へ到達できる

```powershell
py -3.11 --version
Test-Path 'D:\images\incoming'
Test-Path '\\DESKTOP-EXAMPLE\generated\images'
```

UNC の `Test-Path` が失敗する場合は、共有名・ネットワーク接続・共有権限・NTFS 権限を直します。ツールの設定を先に変えても解決しません。

## 2. プロジェクトと仮想環境

```powershell
Set-Location 'C:\Users\<ユーザー名>\Documents\ai-image-reviewer'
py -3.11 -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

本番の最小環境では `requirements.txt`、テスト込みでは `requirements-dev.txt` を使用します。複数の Python にインストールして「モジュールがない」状態にならないよう、`python -m pip` を使って同じインタープリターへインストールしてください。

PowerShell の実行ポリシーで `Activate.ps1` が拒否された場合は、組織のポリシーを確認してください。許可される環境でユーザー単位の設定を使う例は次のとおりです。

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## 3. Codex CLIの準備（既定）

Codex CLIをインストールし、このツールを動かすWindowsユーザーでChatGPT認証します。APIキー認証は使用しません。

```powershell
codex --version
codex login
codex login status
```

`Logged in using ChatGPT`と表示されることを確認します。`config.yaml`の`codex_cli.require_chatgpt_login: true`により、APIキー認証や不明な認証状態では画像送信前に停止します。

## 4. LM Studioの準備（ローカル代替）

1. LM Studio を検品 PC で起動します。
2. 画像入力に対応する VLM をダウンロードします。既定想定は Qwen3-VL-8B 系ですが、実際のモデル ID は API のモデル一覧と一致させます。
3. Developer / Local Server 画面でサーバーを起動します。
4. API ベース URL とポートを確認します。一般的な例は `http://127.0.0.1:1234/v1` です。
5. モデルをロードし、1 枚の画像を扱えることを確認します。

```powershell
Test-NetConnection 127.0.0.1 -Port 1234
Invoke-RestMethod 'http://127.0.0.1:1234/v1/models'
```

API を LAN へ公開する設定を使う場合、アクセス元を必要な LAN サブネットに限定します。TCP 1234 を WAN に公開しないでください。可能なら LM Studio とツールを同じ PC で動かし、`127.0.0.1` を使います。

## 5. デスクトップGUIの起動

セットアップ後は、エクスプローラーから`start-gui.cmd`をダブルクリックします。PowerShellでは次でも起動できます。

```powershell
python gui.py
```

初回は次の順序で操作します。

1. 入力フォルダと出力フォルダを選択する
2. `Codex CLI / ChatGPT`または`LM Studio / ローカル`タブを選ぶ
3. Codexではモデルと推論設定、LM StudioではAPI URLとモデルを指定する
4. 判定基準は最初に`標準（推奨）`を選ぶ
5. 必要なら「追加確認（クロップ再判定）」を有効にし、通常は`Balanced（推奨）`を選ぶ
6. `接続確認`を実行する
7. `copy`のまま、少数画像で`一括スキャン開始`を実行する
8. モデル判定・最終判定・判定源、crop原因と`review.html`を目視確認する
9. 問題がなければ`フォルダ監視開始`へ進む

GUIの設定は`config.local.yaml`へ保存されます。`標準`では意図的なアニメ調・誇張・重なりを許容し、裏付けのないモデルFAILをREVIEWへ戻します。`厳格`は小さな不確実性まで拾うため、FAIL過多になる場合があります。クロップ再判定の`Fast / Balanced / Strict`は確認量の設定で、判定基準の`緩め / 標準 / 厳格`とは別です。CodexではChatGPT認証以外を拒否する設定が常に維持されます。

停止ボタンは処理中の画像を破損させないため、その1枚の判定終了後に停止します。クロップ中に停止した画像は未確認理由付きで`REVIEW`として記録される場合があります。設定変更後やキャンセル済み画像の再評価には「処理済み画像も再判定」を有効にしてください。`move`を選ぶと元画像が移動するため、実行前に確認ダイアログが表示されます。

## 6. `config.yaml` の編集

`config.yaml` はプロジェクトルートから相対パスを解決します。既存設定を保全する場合は別名にコピーし、各コマンドに `--config` を付けます。

```powershell
Copy-Item config.yaml config.local.yaml
notepad config.local.yaml
```

| 項目 | 内容 | 例 |
|---|---|---|
| `watch.paths` | 入力画像フォルダの一覧 | `D:\images\incoming` / `\\PC\share\images` |
| `watch.recursive` | サブフォルダも対象にするか | `true` |
| `watch.mode` | `polling` または `watchdog` | UNC は `polling` 推奨 |
| `watch.polling_interval_seconds` | ポーリング間隔 | `5` |
| `watch.file_stable_seconds` | 書き込み完了待ち | `2` |
| `output.directory` | 仕分け先ルート | `output` |
| `output.operation` | `copy` または `move` | 初回は `copy` |
| `output.preserve_relative_paths` | 元フォルダ構造を保持 | `true` |
| `classifier.backend` | 判定バックエンド | `codex_cli`または`lmstudio` |
| `codex_cli.model` | Codexモデル | `gpt-5.6-luna` |
| `codex_cli.reasoning_effort` | 推論量 | 分類は`low`から開始 |
| `codex_cli.require_chatgpt_login` | APIキー料金経路を拒否 | `true` |
| `codex_cli.max_image_dimension` | 送信前の最大辺 | `2048` |
| `lmstudio.base_url` | OpenAI 互換 API | `http://127.0.0.1:1234/v1` |
| `lmstudio.model` | API のモデル ID | `qwen3-vl-8b` |
| `lmstudio.timeout_seconds` | 1 枚の API タイムアウト | `120` |
| `lmstudio.retries` | API 失敗時の再試行回数 | `2` |
| `processing.parallel_workers` | 並列数 | 初回は `1` |
| `rules.*` | confidence／スコア／問題語の安全ルール | 設定例を参照 |
| `crop_recheck.enabled` | 顔・手・足の追加確認 | 初期値`false`。有効化を推奨 |
| `crop_recheck.mode` | crop確認量 | `fast` / `balanced` / `strict`。`balanced`推奨 |
| `crop_recheck.keep_crop_files` | cropサムネイルを保持 | `true` / `false` |
| `crop_recheck.crop_cache_dir` | 一時crop保存先 | `cache/crops` |
| `crop_recheck.planner.*` | trigger、候補数、padding、最小サイズ等 | 既定値は実装設計を参照 |
| `crop_recheck.detectors.*` | VLM領域検出と失敗時方針 | `auto` / `vlm` / `none` |
| `crop_recheck.targets.*` | face / hand / footの対象 | upper/lower bodyは予約 |
| `cache.use_content_hash` | ハッシュによる重複抑止 | `true` |
| `report.*` | HTML とサムネイル設定 | 設定例を参照 |

Windows のパスは YAML の単一引用符で囲みます。

```yaml
watch:
  paths:
    - 'D:\images\incoming'
    - '\\DESKTOP-EXAMPLE\generated\images'
output:
  directory: output
  operation: copy
```

入力と同じ共有下へ出力すると、コピー／移動権限・競合・切断時の復旧が複雑になります。可能なら検品 PC のローカル SSD に出力し、完了後に手動で共有へコピーします。

## 7. 接続テストとサンプル実行

仮想環境を有効にした状態で、選択中バックエンドを確認します。既定構成では次を実行します。

```powershell
python main.py --config config.local.yaml test-codex
```

LM Studioへ切り替えた場合は、代わりに次を実行します。

```powershell
python main.py --config config.local.yaml test-lmstudio
```

次に数枚だけを含む検証フォルダをスキャンします。

```powershell
python main.py --config config.local.yaml scan --path 'D:\images\smoke-test'
python main.py --config config.local.yaml build-report
Start-Process .\review.html
```

`results.jsonl`、`latest_summary.csv`、仕分け先ファイル、`review.html` が作成されることを確認します。期待どおりであることを確認するまで、元画像フォルダ全体の監視や `move` へ切り替えないでください。

### クロップ再判定を有効にする場合

配布用`config.yaml`では後方互換性のため無効です。GUIで「追加確認（クロップ再判定）」を有効にするか、設定へ次を追加します。

```yaml
crop_recheck:
  enabled: true
  mode: balanced
  keep_crop_files: true
  crop_cache_dir: cache/crops
  detectors:
    provider: auto
    allow_fallback: true
    detector_failure_policy: review
```

`auto` / `vlm`は選択中のLM StudioまたはCodex CLIのVLMに領域検出を依頼します。別の検出器モデルや固定グリッドは使用しません。`none`、利用不能、JSON不正、候補の低信頼はアプリを停止させず、必要な部位を`REVIEW`へ倒します。1画像あたりの最大呼び出しは、再試行を除きfull 1回＋領域検出1回＋face 2回＋hand 4回＋foot 4回です。

full `PASS`にcrop `REVIEW`／`FAIL`があれば最終結果は少なくとも`REVIEW`です。crop `PASS`だけでfull `REVIEW`を`PASS`へ戻しません。保持したcropは`review.html`で確認できますが、実画像のprecision/recallは未校正であり、現時点の自動テストはstub中心です。

## 8. 監視の開始

設定ファイルの対象を確認後、監視を開始します。

```powershell
python main.py --config config.local.yaml watch
```

`Ctrl+C` で停止します。生成アプリが書き込み中の画像を監視する場合は `file_stable_seconds` を十分な値にし、未完成ファイルを VLM に送らないようにします。ネットワーク共有では `polling` が安定することがあります。

停止後に同じ画像を設定変更後の条件でやり直すには、GUIの「処理済み画像も再判定」を使うか、CLIで`--force`を付けます。`cache/processed.json`は画像hashだけを管理し、crop単位のresume／推論結果cacheはありません。

## 9. タスクスケジューラ（任意）

常駐させる場合は、`.venv\Scripts\python.exe` と `main.py` の絶対パス、プロジェクトルートを作業フォルダに指定します。「ユーザーがログオンしているかどうかにかかわらず実行」では UNC の資格情報が別になる場合があるため、対話型 PowerShell と同じユーザー／資格情報でまず検証してください。失敗時に無限再試行しないこと、ログを定期確認することも必要です。

## 10. 更新とバックアップ

```powershell
& .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt --upgrade
python -m pytest -q
python main.py --config config.local.yaml test-codex
```

更新前に `config*.yaml`、`logs/`、`cache/`（保持cropを含む）、仕分け済み画像をバックアップします。依存関係更新後は少数画像で再確認してください。実VLMの品質測定は自動テストに含まれないため、バックエンド接続テストと目視確認を別途行います。

## トラブルシューティング早見表

| 症状 | 確認 |
|---|---|
| Codex認証エラー | 同じWindowsユーザーで`codex login status`、ChatGPT認証か |
| API 接続不可 | LM Studio、URL/ポート、モデルロード、ローカル FW |
| モデルが画像を受け付けない | VLM の画像対応、API のモデル ID、画像 MIME |
| UNC が見えない | 同じユーザーで `Test-Path`、共有／NTFS 権限、資格情報 |
| 結果が全て REVIEW | JSON 応答、confidence/スコア、`rules` の閾値と問題語 |
| cropが全てREVIEW | `detectors.provider`、領域検出JSON、検出信頼度、`review.html`のpipeline理由 |
| 設定変更が反映されない | hash cacheが残っているためGUI forceまたは`scan --force`。`rescan-review`は`output/review`を対象 |
| コピー失敗 | 出力先権限、同名ファイル、ファイルロック、容量 |
| 監視が重複する | 二重起動、安定化待ち、ハッシュ、ポーリング間隔 |
