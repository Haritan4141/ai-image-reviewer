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

## 5. `config.yaml` の編集

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

## 6. 接続テストとサンプル実行

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

## 7. 監視の開始

設定ファイルの対象を確認後、監視を開始します。

```powershell
python main.py --config config.local.yaml watch
```

`Ctrl+C` で停止します。生成アプリが書き込み中の画像を監視する場合は `file_stable_seconds` を十分な値にし、未完成ファイルを VLM に送らないようにします。ネットワーク共有では `polling` が安定することがあります。

## 8. タスクスケジューラ（任意）

常駐させる場合は、`.venv\Scripts\python.exe` と `main.py` の絶対パス、プロジェクトルートを作業フォルダに指定します。「ユーザーがログオンしているかどうかにかかわらず実行」では UNC の資格情報が別になる場合があるため、対話型 PowerShell と同じユーザー／資格情報でまず検証してください。失敗時に無限再試行しないこと、ログを定期確認することも必要です。

## 9. 更新とバックアップ

```powershell
& .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt --upgrade
python -m pytest -q
python main.py --config config.local.yaml test-codex
```

更新前に `config*.yaml`、`logs/`、`cache/`、仕分け済み画像をバックアップします。依存関係更新後は少数画像で再確認してください。

## トラブルシューティング早見表

| 症状 | 確認 |
|---|---|
| Codex認証エラー | 同じWindowsユーザーで`codex login status`、ChatGPT認証か |
| API 接続不可 | LM Studio、URL/ポート、モデルロード、ローカル FW |
| モデルが画像を受け付けない | VLM の画像対応、API のモデル ID、画像 MIME |
| UNC が見えない | 同じユーザーで `Test-Path`、共有／NTFS 権限、資格情報 |
| 結果が全て REVIEW | JSON 応答、confidence/スコア、`rules` の閾値と問題語 |
| コピー失敗 | 出力先権限、同名ファイル、ファイルロック、容量 |
| 監視が重複する | 二重起動、安定化待ち、ハッシュ、ポーリング間隔 |
