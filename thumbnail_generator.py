"""
Banana Replica - AI Studio 完全レプリカ
プロンプトを一言一句変えず、そのままAPIに流し込むだけの純粋なループ装置
"""

import os
import sys
from datetime import datetime
from pathlib import Path

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("エラー: google-genai が必要です")
    print("  pip install google-genai")
    exit(1)


def load_image(image_path: Path) -> types.Part:
    """画像をそのまま読み込む"""
    with open(image_path, "rb") as f:
        image_data = f.read()
    return types.Part.from_bytes(data=image_data, mime_type="image/png")


def main():
    # APIキー
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("エラー: GEMINI_API_KEY を設定してください")
        return

    client = genai.Client(api_key=api_key)

    # 参照画像
    illustration_path = Path("illustration.png")
    image_part = None
    if illustration_path.exists():
        image_part = load_image(illustration_path)
        print(f"参照画像: {illustration_path.absolute()}")
    else:
        print("参照画像: なし（テキストのみで生成）")

    # 出力フォルダ
    output_dir = Path("replica_output")
    output_dir.mkdir(exist_ok=True)

    # ヘッダー
    print("\n" + "=" * 60)
    print("Banana Replica - AI Studio 完全レプリカ")
    print("=" * 60)
    print("モデル: gemini-3-pro-image-preview")
    print("加工: なし（プロンプトをそのまま使用）")
    print("=" * 60)

    # プロンプト入力
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    else:
        print("\nAI Studioのプロンプトをそのまま貼り付けてください:")
        print("（入力後 Enter を2回押して確定）\n")

        lines = []
        while True:
            try:
                line = input()
                if line == "" and lines and lines[-1] == "":
                    lines.pop()
                    break
                lines.append(line)
            except EOFError:
                break
        prompt = "\n".join(lines).strip()

    if not prompt:
        print("エラー: プロンプトが空です")
        return

    print(f"\n【プロンプト（そのまま使用）】")
    print("-" * 60)
    print(prompt[:500] + ("..." if len(prompt) > 500 else ""))
    print("-" * 60)

    # 生成回数
    num_images = 10
    print(f"\n生成開始: {num_images}枚")
    print()

    success = 0
    fail = 0

    # タイムスタンプを取得 (スクリプト実行単位で同一のタイムスタンプを使用)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i in range(1, num_images + 1):
        filename = f"replica_{timestamp}_{i:02d}.png"
        filepath = output_dir / filename

        print(f"[{i}/{num_images}] ", end="", flush=True)

        try:
            # コンテンツ構築（プロンプト + 画像があれば画像）
            if image_part:
                contents = [prompt, image_part]
            else:
                contents = prompt

            # API呼び出し（そのまま、加工なし）
            response = client.models.generate_content(
                model="gemini-3-pro-image-preview",
                contents=contents,
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE", "TEXT"],
                ),
            )

            # 画像抽出
            saved = False
            for part in response.candidates[0].content.parts:
                if part.inline_data is not None:
                    with open(filepath, "wb") as f:
                        f.write(part.inline_data.data)
                    print(f"OK -> {filename}")
                    saved = True
                    success += 1
                    break

            if not saved:
                print("SKIP (画像なし)")
                fail += 1

        except Exception as e:
            print(f"ERROR: {str(e)[:50]}")
            fail += 1

    # 結果
    print("\n" + "=" * 60)
    print(f"完了: 成功 {success} / 失敗 {fail}")
    print(f"保存先: {output_dir.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
