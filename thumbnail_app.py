import os
import json
import streamlit as st
from datetime import datetime
from pathlib import Path
from PIL import Image

try:
    from google import genai
    from google.genai import types
except ImportError:
    st.error("エラー: google-genai が必要です。 pip install google-genai を実行してください。")
    st.stop()

# ページ設定
st.set_page_config(page_title="Banana Replica UI", page_icon="🍌", layout="wide")


# ==============================================================
# パスワード保護（Streamlit Cloud デプロイ時のセキュリティ）
# ==============================================================
def check_password():
    """アプリ起動時にパスワード認証を行う。secrets にパスワード未設定の場合はスキップ。"""
    # secrets に APP_PASSWORD が設定されていない場合（ローカル開発など）は認証スキップ
    try:
        app_password = st.secrets["APP_PASSWORD"]
    except (FileNotFoundError, KeyError):
        return True

    if not app_password:
        return True

    if st.session_state.get("authenticated", False):
        return True

    st.title("🍌 Banana Replica UI")
    st.markdown("### 🔒 パスワードを入力してください")
    pwd = st.text_input("パスワード", type="password", key="login_password")
    if st.button("ログイン", use_container_width=True, type="primary"):
        if pwd == app_password:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("パスワードが違います。")
    return False

if not check_password():
    st.stop()


st.title("🍌 Banana Replica UI")
st.markdown("AI StudioのレプリカをWebブラウザ上で操作できる対話型ツールです。")

# ==== セッション状態の初期化 ====
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# APIキーの取得優先順位: Secrets → 環境変数 → サイドバー入力
def get_api_key():
    """Streamlit Secrets > 環境変数 > 手動入力 の順にAPIキーを取得"""
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
        if key:
            return key
    except (FileNotFoundError, KeyError):
        pass
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""

if "api_key" not in st.session_state:
    st.session_state.api_key = get_api_key()

# ギャラリー蓄積用（ボタンを押すたびに5枚ずつ追加、最大20枚）
if "gallery_images" not in st.session_state:
    st.session_state.gallery_images = []  # 現在のギャラリーの画像パスリスト

# 過去のプロンプト初期化
past_prompts_file = Path(__file__).parent / "past_prompts.json"
if "past_prompts" not in st.session_state:
    if past_prompts_file.exists():
        with open(past_prompts_file, "r", encoding="utf-8") as f:
            st.session_state.past_prompts = json.load(f)
    else:
        st.session_state.past_prompts = []

def save_prompt(new_prompt):
    if new_prompt and new_prompt not in st.session_state.past_prompts:
        st.session_state.past_prompts.insert(0, new_prompt)
        st.session_state.past_prompts = st.session_state.past_prompts[:20]
        with open(past_prompts_file, "w", encoding="utf-8") as f:
            json.dump(st.session_state.past_prompts, f, ensure_ascii=False, indent=2)

# 出力ディレクトリ
output_dir = Path(__file__).parent / "replica_output"
output_dir.mkdir(exist_ok=True, parents=True)


# ==============================================================
# ギャラリー表示コンポーネント（◀ ▶ で画像を切り替え＋サムネ一覧）
# ==============================================================
def show_gallery(images, gallery_key):
    """◀ ▶ ボタンで大きな画像を切り替えて比較できるギャラリー"""
    valid_images = [p for p in images if p.exists()]
    if not valid_images:
        return

    idx_key = f"gidx_{gallery_key}"
    if idx_key not in st.session_state:
        st.session_state[idx_key] = 0

    total = len(valid_images)
    idx = st.session_state[idx_key] % total

    # --- ナビゲーションバー: ◀前へ  [3 / 10]  次へ▶ ---
    nav_left, nav_center, nav_right = st.columns([1, 3, 1])
    with nav_left:
        if st.button("◀ 前へ", key=f"prev_{gallery_key}", use_container_width=True, disabled=(total <= 1)):
            st.session_state[idx_key] = (idx - 1) % total
            st.rerun()
    with nav_center:
        st.markdown(
            f"<p style='text-align:center; font-size:1.3rem; font-weight:bold; margin:0.3rem 0;'>"
            f"{idx + 1} / {total}</p>",
            unsafe_allow_html=True,
        )
    with nav_right:
        if st.button("次へ ▶", key=f"next_{gallery_key}", use_container_width=True, disabled=(total <= 1)):
            st.session_state[idx_key] = (idx + 1) % total
            st.rerun()

    # --- メイン画像（大きく表示） ---
    _, center_col, _ = st.columns([1, 6, 1])
    with center_col:
        st.image(
            str(valid_images[idx]),
            caption=valid_images[idx].name,
            use_container_width=True,
        )

    # --- サムネイルストリップ（1行5枚、現在選択中にマーク表示） ---
    if total > 1:
        for row_start in range(0, total, 5):
            row_imgs = valid_images[row_start:row_start + 5]
            thumb_cols = st.columns(5)
            for j, thumb_path in enumerate(row_imgs):
                actual_idx = row_start + j
                with thumb_cols[j]:
                    label = f"▲ {actual_idx + 1}" if actual_idx == idx else f"{actual_idx + 1}"
                    st.image(str(thumb_path), caption=label, use_container_width=True)


# ==== サイドバー ====
with st.sidebar:
    st.header("⚙️ 設定")
    # Secrets にAPIキーが設定済みならサイドバーの入力欄を非表示にする
    api_from_secrets = False
    try:
        if st.secrets.get("GEMINI_API_KEY", ""):
            api_from_secrets = True
    except (FileNotFoundError, KeyError):
        pass

    if api_from_secrets:
        st.success("✅ APIキー設定済み")
    else:
        api_key_input = st.text_input("Gemini API Key", value=st.session_state.api_key, type="password")
        if api_key_input != st.session_state.api_key:
            st.session_state.api_key = api_key_input
            st.rerun()

    st.header("👤 キャラクター設定")
    use_illustration = st.checkbox("すあし社長のイラストを含める", value=True)
    if use_illustration:
        ill_path = Path(__file__).parent / "illustration.png"
        if not ill_path.exists():
            st.warning("illustration.png が見つかりません。")

    st.header("🖼️ 追加の参考画像 (任意)")
    st.markdown("他にも参考にしたい画像があればアップロードしてください")
    uploaded_files = st.file_uploader("画像を選択", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    if uploaded_files:
        st.success(f"{len(uploaded_files)} 枚の追加画像をセットしました")
        for file in uploaded_files:
            st.image(file, caption=file.name, use_container_width=True)

    # ギャラリー状況の表示
    st.markdown("---")
    gallery_count = len(st.session_state.gallery_images)
    st.markdown(f"**📊 現在のギャラリー: {gallery_count} / 20 枚**")
    if gallery_count > 0:
        if st.button("🔄 ギャラリーをリセット（新しく始める）", use_container_width=True):
            st.session_state.gallery_images = []
            st.rerun()


# ==== メインエリア ====

# --- 現在のギャラリー（蓄積された画像を ◀ ▶ で閲覧） ---
if st.session_state.gallery_images:
    gallery_count = len(st.session_state.gallery_images)
    st.subheader(f"🖼️ 生成ギャラリー（{gallery_count} 枚）")
    show_gallery(st.session_state.gallery_images, "main_gallery")
    st.markdown("---")

# --- 過去の会話履歴 ---
for msg_idx, message in enumerate(st.session_state.chat_history):
    with st.chat_message(message["role"]):
        st.markdown(message["text"])
        if "images" in message and message["images"]:
            show_gallery(message["images"], f"hist_{msg_idx}")


# ==== 入力エリア：プロンプト送信と画像生成 ====

# プロンプト入力用のセッション状態を初期化
if "current_prompt" not in st.session_state:
    st.session_state.current_prompt = ""

def set_prompt(text):
    st.session_state.current_prompt = text

st.markdown("---")
# 過去のプロンプトをワンクリックで入力欄にセット
if st.session_state.past_prompts:
    st.markdown("**💡 過去のプロンプトをクリックして入力欄にセット:**")
    cols = st.columns(3)
    for idx, past_prompt in enumerate(st.session_state.past_prompts[:6]):
        with cols[idx % 3]:
            btn_label = past_prompt if len(past_prompt) <= 18 else past_prompt[:18] + "..."
            st.button(btn_label, key=f"past_btn_{idx}", help=past_prompt, on_click=set_prompt, args=(past_prompt,), use_container_width=True)

# ボタンラベルの決定
gallery_count = len(st.session_state.gallery_images)
remaining = 20 - gallery_count
is_max = remaining <= 0

if is_max:
    btn_label = "🚫 最大20枚に達しました（リセットしてください）"
elif gallery_count == 0:
    btn_label = "✨ 画像を生成する（5枚）"
else:
    btn_label = f"✨ さらに5枚追加生成する（現在 {gallery_count} 枚 → {gallery_count + 5} 枚）"

# 入力フォーム
with st.form(key="prompt_form"):
    prompt = st.text_area(
        "プロンプトまたは修正指示を入力してください... (例: オフィス背景で明るく)",
        key="current_prompt",
        height=100,
    )
    submit_button = st.form_submit_button(
        label=btn_label,
        use_container_width=True,
        type="primary",
        disabled=is_max,
    )

if submit_button and prompt and not is_max:
    save_prompt(prompt)

    if not st.session_state.api_key:
        st.error("左のサイドバーから Gemini API Key を設定してください。")
        st.stop()

    client = genai.Client(api_key=st.session_state.api_key)

    # 今回生成する枚数（常に5枚、ただし上限20枚を超えない）
    num_to_generate = min(5, 20 - len(st.session_state.gallery_images))

    # 生成処理
    with st.spinner(f"5枚追加生成中... もうしばらくお待ちください！"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # APIへの送信コンテンツ構築
        contents = [prompt]

        if use_illustration:
            ill_path = Path(__file__).parent / "illustration.png"
            if ill_path.exists():
                with open(ill_path, "rb") as f:
                    ill_data = f.read()
                ill_part = types.Part.from_bytes(data=ill_data, mime_type="image/png")
                contents.append(ill_part)

        if uploaded_files:
            for file in uploaded_files:
                image_part = types.Part.from_bytes(data=file.getvalue(), mime_type=file.type)
                contents.append(image_part)

        success = 0

        # プログレスバーとステータス
        progress_bar = st.progress(0)
        status_text = st.empty()

        # 通し番号は既存の枚数 + 1 から開始
        start_num = len(st.session_state.gallery_images) + 1

        for i in range(num_to_generate):
            img_num = start_num + i
            filename = f"replica_{timestamp}_{img_num:02d}.png"
            filepath = output_dir / filename

            status_text.text(f"生成中 [{i + 1}/{num_to_generate}] ... （合計 {img_num} 枚目）")

            try:
                response = client.models.generate_content(
                    model="gemini-3-pro-image-preview",
                    contents=contents,
                    config=types.GenerateContentConfig(
                        response_modalities=["IMAGE", "TEXT"],
                    ),
                )

                for part in response.candidates[0].content.parts:
                    if part.inline_data is not None:
                        with open(filepath, "wb") as f:
                            f.write(part.inline_data.data)
                        # ギャラリーに追加（蓄積）
                        st.session_state.gallery_images.append(filepath)
                        success += 1
                        break

            except Exception as e:
                status_text.text(f"⚠️ 画像 {img_num} でエラー: {str(e)[:80]}")

            progress_bar.progress((i + 1) / num_to_generate)

        total_count = len(st.session_state.gallery_images)
        status_text.text(f"✅ {success} 枚追加完了！ ギャラリー合計: {total_count} 枚")

    # UIを更新してギャラリーを表示
    st.rerun()
