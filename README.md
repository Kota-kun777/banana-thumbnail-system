# Banana Thumbnail System

Streamlit でサムネイルを生成するアプリです。

## モデル比較

- 既定値は従来どおり Gemini 3 Pro / GPT Image 2 の高品質設定です。
- Gemini は 3.1 Flash（低コスト・バランス）と 3.1 Flash Lite（最安・1K）を選べます。
- OpenAI は GPT Image 2 の high / medium / low を比較できます。旧廉価モデルの GPT Image 1 mini も比較用に選べますが、公式に非推奨のため継続運用には使わないでください。
- 生成履歴には、使用したモデル名とOpenAIの品質・サイズが記録されます。

## 保存の仕組み

- プロンプト履歴
  - GitHub Gist をデバイス横断の正本として使用します。
  - 保存時にGistの最新版を再取得してからマージするため、先に開いていたPC画面がスマホ側の新しい履歴を上書きしません。
  - 同じブラウザの `localStorage` とサーバー上の `past_prompts.json` も予備として併用します。
- 生成画像
  - `replica_output/gallery_manifest.json` に現在の作業ギャラリーを記録し、ブラウザを閉じた後や別ブラウザから開いた場合も復元します。
  - 同じプロンプトの作業ギャラリーは最大200枚です。プロンプトを変えると新しい作業ギャラリーを開始し、以前の画像はプロンプト付きの生成履歴へ移ります。
  - 生成履歴は標準7日、最大100回・合計500枚まで保存します。各回を画面で再表示し、ZIPでまとめて保存できます。
  - Streamlit Community Cloud の実行ファイルはサーバー再起動時に消える場合があります。残したい画像はサイドバーの「全画像のZIPを準備」からダウンロードしてください。

## Streamlit Secrets

デバイス横断のプロンプト同期には、Streamlit Community Cloud の App settings → Secrets に次を登録します。値をリポジトリへコミットしないでください。

```toml
github_token = "Gistを書き込めるGitHubトークン"

# 任意。省略時は banana_past_prompts.json を含む既存Gistを探し、
# 見つからなければ初回保存時に非公開（secret）Gistを自動作成します。
# gist_id = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# 任意（1〜30日。省略時は7日）
GALLERY_RETENTION_DAYS = 7
```

既存の `GITHUB_TOKEN` / `GIST_ID` という大文字名にも対応しています。

## 検証

```powershell
python -m py_compile thumbnail_app.py gallery_persistence.py
python -m unittest discover -s tests -v
```
