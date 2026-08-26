# コマンド集

すべてのコマンドはプロジェクトルートで、仮想環境を有効にして実行します。

```powershell
Set-Location 'C:\Users\<ユーザー名>\Documents\ai-image-reviewer'
& .\.venv\Scripts\Activate.ps1
```

## 共通形式

設定ファイルは `--config` で明示できます。省略時はプロジェクトルートの `config.yaml` を使用します。

```powershell
python main.py --config config.local.yaml <command>
python main.py --help
```

## `test-codex`

Codex CLIの存在、バージョン、認証方式、設定モデルを確認します。モデル推論は実行しないため、この確認自体はPro利用枠を消費しません。

```powershell
codex login status
python main.py --config config.yaml test-codex
```

既定の`codex_cli.require_chatgpt_login: true`では、認証方式がChatGPTでなければ失敗します。APIキー認証のままスキャンしてOpenAI Platform API料金が発生することを防ぐためのガードです。

## `test-lmstudio`

LM Studio の OpenAI 互換 API に接続できるかを確認します。

```powershell
python main.py --config config.yaml test-lmstudio
```

確認するもの:

- `lmstudio.base_url` が正しい
- Local Server が起動している
- `lmstudio.model` が API のモデル ID と一致する
- 画像入力対応モデルがロードされている

モデル一覧だけを確認する場合:

```powershell
Invoke-RestMethod 'http://127.0.0.1:1234/v1/models'
```

## `scan`

既存フォルダを一括スキャンします。`--path` は設定ファイルの `watch.paths` を上書きする一時指定です。

```powershell
python main.py --config config.yaml scan --path 'D:\images\batch1'
python main.py --config config.yaml scan --path '\\DESKTOP-7600\share\generated'
```

サブフォルダ、対応拡張子、出力先、コピー／移動、閾値は `config.yaml` に従います。まず 5〜10 枚で動作を確認してから大量処理します。利用可能な引数は次で確認できます。

```powershell
python main.py scan --help
```

## `watch`

設定された入力フォルダを監視し、新しい画像を自動処理します。

```powershell
python main.py --config config.yaml watch
```

終了は `Ctrl+C` です。`watch.mode`、ポーリング間隔、サブフォルダ、ファイル安定化、同時実行数は設定ファイルに従います。ネットワーク共有を監視する場合は `polling` が安定することがあります。

常駐前に次を確認します。

- 対象フォルダが実行ユーザーから見える
- 出力先が対象入力の配下ではない（自己再検出防止）
- LM Studio が起動しモデルがロードされている
- `copy` で十分な容量がある
- 同じ `watch` プロセスを二重起動していない

## `rescan-review`

既存の `output/review` を対象に再判定します。閾値やプロンプトを調整した後に使用します。

```powershell
python main.py --config config.yaml rescan-review
```

再判定後に `REVIEW` から `PASS` または `FAIL` へ移ることがあります。再判定履歴を残す実装では、元の判定と再判定時刻を区別します。初回は `copy` を推奨します。

## `build-report`

保存済み結果から静的レビュー画面を再生成します。VLM への問い合わせは行いません。

```powershell
python main.py --config config.yaml build-report
Start-Process .\review.html
```

通常はプロジェクト直下の `review.html` を生成します。HTML と参照画像の相対パス関係を保ったまま、同じフォルダから開いてください。表示できない場合:

```powershell
python -m http.server 8000
Start-Process 'http://127.0.0.1:8000/review.html'
```

## ログ確認

```powershell
Get-Content .\logs\results.jsonl -Tail 5
Import-Csv .\logs\latest_summary.csv | Select-Object -First 10
Get-Content .\logs\app.log -Tail 50
```

JSONL は 1 行 1 件の結果、CSV は一覧、`app.log` はアプリケーションログです。処理済みキャッシュを使う版では `cache/processed.json` もログと一緒に保全します。

## 停止・再開

`watch` は `Ctrl+C` で停止し、同じコマンドで再開します。停止中に生成された画像は `scan` で追いつけます。

```powershell
python main.py --config config.yaml scan --path 'D:\images\incoming'
python main.py --config config.yaml watch
```

バックエンド障害中に大量の再試行を発生させないため、`test-codex`または`test-lmstudio`を通してから再開してください。失敗がログに記録される場合は、未処理画像を少数ずつ再試行します。

## 安全な運用順序

1. 選択中バックエンドに合わせて`test-codex`または`test-lmstudio`
2. `scan --path` で少数画像
3. `build-report` と元画像の目視比較
4. `copy` のまま通常バッチ
5. ログと重複抑止を確認
6. 必要性を検討してから `move`

自動削除は提供しません。VLM の結果が不明・JSON 不正・タイムアウトの場合は安全側の `REVIEW` として扱います。
