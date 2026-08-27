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

## デスクトップGUI

エクスプローラーで`start-gui.cmd`をダブルクリックするか、次のコマンドで起動します。

```powershell
python gui.py
```

GUIは設定を`config.local.yaml`へ保存します。バックエンド／モデル／推論設定／判定基準、入力・出力、`copy` / `move`、一括スキャン、継続監視、接続確認、レポート表示を画面から操作できます。判定基準は`緩め`、`標準（推奨）`、`厳格`から、crop確認量は`Fast`、`Balanced（推奨）`、`Strict`から選択できます。高度な閾値やplanner／detector設定は`config.local.yaml`を直接編集できます。

停止は現在処理中の1枚が完了した時点で反映されます。Codex CLIの実行中プロセスやLM Studioへの進行中HTTPリクエストを強制終了しません。

Windows GUIから実行するCodex CLI子プロセスにはコンソール非表示フラグを指定しています。判定ごとに黒い画面が表示される場合は、古いGUIプロセスが残っていないことを確認してGUIを再起動してください。

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

既存フォルダを一括スキャンします。`--path` は設定ファイルの `watch.paths` を上書きする一時指定です。処理済みhashを無視して設定変更後に再評価する場合は`--force`を付けます。

```powershell
python main.py --config config.yaml scan --path 'D:\images\batch1'
python main.py --config config.yaml scan --path '\\DESKTOP-7600\share\generated'
python main.py --config config.yaml scan --path 'D:\images\batch1' --force
```

サブフォルダ、対応拡張子、出力先、コピー／移動、閾値、crop再判定は `config.yaml` に従います。まず 5〜10 枚で動作を確認してから大量処理します。利用可能な引数は次で確認できます。

```powershell
python main.py scan --help
```

## crop再判定

`crop_recheck.enabled: true`の場合、full判定の後に選択中のVLMで領域を検出し、face／hand／条件付きfootを追加判定します。常用は`balanced`推奨です。`fast`はfullのREVIEW、低信頼・低スコア・部位問題語・小さい部位の要約などをtriggerにし、`strict`は人物がいる可能性のある画像の可視部位を広く確認します。`rules.mode`（判定の厳しさ）とは別設定です。

```powershell
python main.py --config config.local.yaml scan --path 'D:\images\smoke-test' --force
```

`auto` / `vlm`は選択中バックエンドの`locate_regions()`を使い、`none`や利用不能時は必要な領域を`REVIEW`として記録します。固定グリッドや未検出部位の推測はありません。1画像の上限は再試行を除きfull 1回＋localization 1回＋face 2回＋hand 4回＋foot 4回（最大12回）です。full `REVIEW`をcrop `PASS`だけで`PASS`へ上げず、crop失敗・JSON不正・低信頼は安全側へ倒します。

生成cropは`cache/crops/<hash>/run-...`へ置き、`keep_crop_files: false`ではこの実行が作ったファイルだけを削除します。`results.jsonl`／CSV／`review.html`にはbox、判定、信頼度、detector、pipeline metadataが残ります。保持cropやログにはローカルパスが含まれるため、共有・公開しないでください。

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
- `crop_recheck`を有効にする場合は、crop cache容量と選択中VLMの画像入力を確認する

## `rescan-review`

既存の `output/review` を対象に強制再判定します。閾値、プロンプト、crop設定を調整した後に使用します。対象は`output/review`だけです。

```powershell
python main.py --config config.yaml rescan-review
```

再判定後に `REVIEW` から `PASS` または `FAIL` へ移ることがあります。`cache/processed.json`は画像hashだけを保持し、crop単位のresume／推論cacheはありません。GUIで全入力を再評価する場合は「処理済み画像も再判定」を有効にし、CLIでは`scan --force`を使います。初回は `copy` を推奨します。

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

JSONLは1行1件の結果、CSVは一覧、`app.log`はアプリケーションログです。JSONL/CSVにはモデル判定、最終判定、判定源、ルール根拠、crop checks、pipeline metadataも保存されます。`review.html`では実在する保持cropだけをサムネイル表示します。処理済みキャッシュ`cache/processed.json`は画像hashだけなので、ログと一緒に保全してください。

## 停止・再開

GUIの「停止」ボタンはコントローラのキャンセルcallbackを設定し、現在の画像が安全な境界まで進んだあと停止します。crop確認中なら、残りの部位が未確認であることを含むキャンセル記録（`REVIEW`、`pipeline_stage: cancelled`）が保存されることがあります。

CLIの`watch`／`scan`で`Ctrl+C`を押した場合はプロセス割り込みです。完了済みの記録は保持されますが、割り込み時点で推論中の画像は最終レコードがJSONLへ追加されない場合があります。未処理画像は同じ入力に通常の`scan`を再実行して追いつけます。すでに処理済みまたはキャンセル済みの画像を設定変更後にやり直すときは、GUIのforce、または画像hash抑止を無視する`scan --force`を使います。

```powershell
python main.py --config config.yaml scan --path 'D:\images\incoming'
python main.py --config config.yaml scan --path 'D:\images\incoming' --force
python main.py --config config.yaml watch
```

バックエンド障害中に大量の再試行を発生させないため、`test-codex`または`test-lmstudio`を通してから再開してください。自動テストは実VLMのprecision/recallを測定しないため、失敗がログに記録される場合は、未処理画像を少数ずつ再試行し、`review.html`を目視確認します。

## 安全な運用順序

1. 選択中バックエンドに合わせて`test-codex`または`test-lmstudio`
2. `scan --path` で少数画像
3. `build-report` と元画像の目視比較
4. `copy` のまま通常バッチ
5. ログと重複抑止を確認
6. 必要性を検討してから `move`

自動削除は提供しません。VLM の結果が不明・JSON 不正・タイムアウトの場合は安全側の `REVIEW` として扱います。
