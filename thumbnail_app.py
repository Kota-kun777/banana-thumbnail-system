import os
import json
import time
import threading
import uuid
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
# バックグラウンド画像生成システム
# ==============================================================
class GenerationState:
    """スレッドとUI間でデータを共有するためのスレッドセーフな状態クラス"""
    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        self.stop_requested = False
        self.images = []          # 生成済み画像パス (str)
        self.errors = []          # エラーメッセージ
        self.total = 0            # 生成予定枚数
        self.completed = 0        # 完了した試行数
        self.success_count = 0    # 成功数
        self.status = ""          # 現在の状態テキスト
        self.finished = False     # スレッド完了フラグ


_gen_states = {}
_gen_states_lock = threading.Lock()


def get_gen_state(session_id):
    with _gen_states_lock:
        if session_id not in _gen_states:
            _gen_states[session_id] = GenerationState()
        return _gen_states[session_id]


def generation_worker(session_id, api_key, contents, num_to_generate,
                      output_dir, timestamp, start_num):
    """バックグラウンドスレッドで画像を生成するワーカー関数"""
    state = get_gen_state(session_id)
    MAX_RETRIES = 3

    try:
        client = genai.Client(api_key=api_key)

        for i in range(num_to_generate):
            if state.stop_requested:
                with state.lock:
                    state.status = f"⏹️ ユーザーにより停止（{state.success_count}枚生成済み）"
                break

            img_num = start_num + i
            filename = f"replica_{timestamp}_{img_num:02d}.png"
            filepath = output_dir / filename

            for attempt in range(MAX_RETRIES):
                if state.stop_requested:
                    break

                retry_label = f"（リトライ {attempt + 1}/{MAX_RETRIES}）" if attempt > 0 else ""
                with state.lock:
                    state.status = f"生成中 [{i + 1}/{num_to_generate}] ...（合計 {img_num} 枚目）{retry_label}"

                try:
                    response = client.models.generate_content(
                        model="gemini-3-pro-image-preview",
                        contents=contents,
                        config=types.GenerateContentConfig(
                            response_modalities=["IMAGE", "TEXT"],
                        ),
                    )

                    # candidatesが空 → ブロックされた場合リトライ
                    if not response.candidates:
                        block_reason = ""
                        if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                            block_reason = str(getattr(response.prompt_feedback, 'block_reason', ''))
                        if attempt < MAX_RETRIES - 1:
                            time.sleep(2 * (attempt + 1))
                            continue
                        else:
                            with state.lock:
                                state.errors.append(f"画像 {img_num}: {MAX_RETRIES}回リトライ後も生成失敗（{block_reason}）")
                            break

                    # 画像データを探す
                    image_found = False
                    text_response = ""
                    for part in response.candidates[0].content.parts:
                        if part.inline_data is not None:
                            with open(filepath, "wb") as f:
                                f.write(part.inline_data.data)
                            with state.lock:
                                state.images.append(str(filepath))
                                state.success_count += 1
                            image_found = True
                            break
                        elif part.text:
                            text_response = part.text

                    if not image_found:
                        if attempt < MAX_RETRIES - 1:
                            time.sleep(2 * (attempt + 1))
                            continue
                        err_detail = f"画像 {img_num}: 画像データなし"
                        if text_response:
                            err_detail += f"（API応答: {text_response[:100]}）"
                        with state.lock:
                            state.errors.append(err_detail)
                    break  # 成功 or 最終リトライ後 → 次の画像へ

                except Exception as e:
                    if attempt < MAX_RETRIES - 1 and ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)):
                        with state.lock:
                            state.status = f"⏳ レート制限のため待機中... [{i + 1}/{num_to_generate}]（{attempt + 1}回目）"
                        time.sleep(5 * (attempt + 1))
                        continue
                    with state.lock:
                        state.errors.append(f"画像 {img_num}: {str(e)[:150]}")
                    break

            with state.lock:
                state.completed = i + 1

    finally:
        with state.lock:
            state.running = False
            state.finished = True


# ==============================================================
# パスワード保護（Streamlit Cloud デプロイ時のセキュリティ）
# ==============================================================
def check_password():
    """アプリ起動時にパスワード認証を行う。secrets にパスワード未設定の場合はスキップ。"""
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

# セッションID（バックグラウンドスレッドとの通信用）
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

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

# ギャラリー蓄積用（ボタンを押すたびに追加、最大50枚）
if "gallery_images" not in st.session_state:
    st.session_state.gallery_images = []

# 生成結果メッセージ（rerun後も表示するため）
if "last_gen_errors" not in st.session_state:
    st.session_state.last_gen_errors = []
if "last_gen_success" not in st.session_state:
    st.session_state.last_gen_success = None

# 生成中フラグ
if "generating" not in st.session_state:
    st.session_state.generating = False

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
        st.session_state.past_prompts = st.session_state.past_prompts[:50]
        with open(past_prompts_file, "w", encoding="utf-8") as f:
            json.dump(st.session_state.past_prompts, f, ensure_ascii=False, indent=2)

# 出力ディレクトリ
output_dir = Path(__file__).parent / "replica_output"
output_dir.mkdir(exist_ok=True, parents=True)

# プロンプト入力用のセッション状態を初期化
if "current_prompt" not in st.session_state:
    st.session_state.current_prompt = ""

def set_prompt(text):
    st.session_state.current_prompt = text

# 生成枚数の選択用コールバック
if "gen_count" not in st.session_state:
    st.session_state.gen_count = 5

def set_gen_count(count):
    st.session_state.gen_count = count


# ==============================================================
# ギャラリー表示コンポーネント（◀ ▶ で画像を切り替え＋サムネ一覧）
# ==============================================================
def _go_prev(idx_key, total):
    st.session_state[idx_key] = (st.session_state[idx_key] - 1) % total

def _go_next(idx_key, total):
    st.session_state[idx_key] = (st.session_state[idx_key] + 1) % total

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
        st.button("◀ 前へ", key=f"prev_{gallery_key}", use_container_width=True,
                  disabled=(total <= 1), on_click=_go_prev, args=(idx_key, total))
    with nav_center:
        st.markdown(
            f"<p style='text-align:center; font-size:1.3rem; font-weight:bold; margin:0.3rem 0;'>"
            f"{idx + 1} / {total}</p>",
            unsafe_allow_html=True,
        )
    with nav_right:
        st.button("次へ ▶", key=f"next_{gallery_key}", use_container_width=True,
                  disabled=(total <= 1), on_click=_go_next, args=(idx_key, total))

    # --- メイン画像（大きく表示） ---
    _, center_col, _ = st.columns([1, 6, 1])
    with center_col:
        st.image(
            str(valid_images[idx]),
            caption=valid_images[idx].name,
            use_container_width=True,
        )
        # --- ダウンロードボタン ---
        with open(valid_images[idx], "rb") as img_file:
            st.download_button(
                label=f"📥 この画像をダウンロード（{valid_images[idx].name}）",
                data=img_file,
                file_name=valid_images[idx].name,
                mime="image/png",
                key=f"dl_{gallery_key}_{idx}",
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


# ==============================================================
# 生成モニターフラグメント（プログレッシブ表示 + 停止ボタン）
# ==============================================================
@st.fragment(run_every=2)
def generation_monitor():
    """2秒ごとにバックグラウンドスレッドの進捗を確認し、画像を表示する"""
    sid = st.session_state.session_id
    state = get_gen_state(sid)

    if not state.running and not state.finished:
        return

    # --- スレッド完了検出: session_stateに同期してフルリラン ---
    if state.finished:
        with state.lock:
            new_images = [Path(p) for p in state.images]
            new_errors = list(state.errors)
            success_count = state.success_count
            # 状態リセット
            state.finished = False
            state.images = []
            state.errors = []
            state.completed = 0
            state.total = 0
            state.success_count = 0
            state.status = ""
            state.stop_requested = False

        # session_stateにギャラリー画像を追加
        old_count = len(st.session_state.gallery_images)
        for img in new_images:
            if img not in st.session_state.gallery_images:
                st.session_state.gallery_images.append(img)

        # ギャラリーインデックスを新しい画像の先頭に移動
        if len(st.session_state.gallery_images) > old_count:
            st.session_state["gidx_main_gallery"] = old_count

        st.session_state.last_gen_errors = new_errors
        st.session_state.last_gen_success = success_count
        st.session_state.generating = False
        st.rerun()
        return

    # --- 生成中: リアルタイム進捗表示 ---
    with state.lock:
        current_images = [Path(p) for p in state.images]
        progress = state.completed / state.total if state.total > 0 else 0
        status = state.status
        total = state.total
        stop_already = state.stop_requested

    st.progress(progress)

    col_status, col_stop = st.columns([4, 1])
    with col_status:
        st.text(status)
    with col_stop:
        if stop_already:
            st.button("⏹️ 停止中...", key="stop_generation", disabled=True, use_container_width=True)
        elif st.button("⏹️ 停止", key="stop_generation", type="secondary", use_container_width=True):
            state.stop_requested = True

    # --- 生成済み画像をグリッド表示（ダウンロードボタン付き） ---
    if current_images:
        st.markdown(f"**生成済み: {len(current_images)} / {total} 枚**")
        cols_per_row = 3
        for row_start in range(0, len(current_images), cols_per_row):
            row_imgs = current_images[row_start:row_start + cols_per_row]
            cols = st.columns(cols_per_row)
            for j, img_path in enumerate(row_imgs):
                with cols[j]:
                    if img_path.exists():
                        st.image(str(img_path), caption=img_path.name, use_container_width=True)
                        with open(img_path, "rb") as f:
                            st.download_button(
                                "📥 ダウンロード",
                                data=f.read(),
                                file_name=img_path.name,
                                mime="image/png",
                                key=f"dl_gen_{img_path.name}",
                                use_container_width=True,
                            )


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

    # 過去のプロンプト履歴（7件目以降をサイドバーに表示）
    if len(st.session_state.past_prompts) > 6:
        st.markdown("---")
        st.header("📝 プロンプト履歴")
        for idx, past_prompt in enumerate(st.session_state.past_prompts[6:]):
            btn_label = past_prompt if len(past_prompt) <= 30 else past_prompt[:30] + "..."
            st.button(
                btn_label,
                key=f"sidebar_past_{idx}",
                help=past_prompt,
                on_click=set_prompt,
                args=(past_prompt,),
                use_container_width=True,
            )

    # ギャラリー状況の表示
    st.markdown("---")
    gallery_count = len(st.session_state.gallery_images)
    st.markdown(f"**📊 現在のギャラリー: {gallery_count} / 50 枚**")
    if gallery_count > 0:
        if st.button("🔄 ギャラリーをリセット（新しく始める）", use_container_width=True):
            st.session_state.gallery_images = []
            st.rerun()


# ==== メインエリア ====

# --- 現在のギャラリー（常にインタラクティブ表示） ---
if st.session_state.gallery_images:
    gallery_count = len(st.session_state.gallery_images)
    st.subheader(f"🖼️ 生成ギャラリー（{gallery_count} 枚）")
    show_gallery(st.session_state.gallery_images, "main_gallery")
    st.markdown("---")

# --- 生成モニター（生成中のみ表示） ---
if st.session_state.generating:
    st.subheader("⏳ 画像生成中...")
    generation_monitor()
    st.markdown("---")

# --- 前回の生成結果メッセージ（rerun後も表示） ---
if st.session_state.last_gen_success is not None:
    if st.session_state.last_gen_success > 0:
        st.success(f"✅ {st.session_state.last_gen_success} 枚の画像を生成しました！")
    else:
        st.warning("⚠️ 画像を生成できませんでした")
if st.session_state.last_gen_errors:
    st.error(f"🔍 {len(st.session_state.last_gen_errors)} 件のエラー:")
    for err in st.session_state.last_gen_errors:
        st.error(err)

# --- 過去の会話履歴 ---
for msg_idx, message in enumerate(st.session_state.chat_history):
    with st.chat_message(message["role"]):
        st.markdown(message["text"])
        if "images" in message and message["images"]:
            show_gallery(message["images"], f"hist_{msg_idx}")


# ==== 入力エリア：プロンプト送信と画像生成 ====

st.markdown("---")
# 過去のプロンプトをワンクリックで入力欄にセット
if st.session_state.past_prompts:
    st.markdown("**💡 過去のプロンプトをクリックして入力欄にセット:**")
    cols = st.columns(3)
    for idx, past_prompt in enumerate(st.session_state.past_prompts[:6]):
        with cols[idx % 3]:
            btn_label = past_prompt if len(past_prompt) <= 18 else past_prompt[:18] + "..."
            st.button(btn_label, key=f"past_btn_{idx}", help=past_prompt, on_click=set_prompt, args=(past_prompt,), use_container_width=True)

# 生成枚数の選択（on_clickコールバックで更新 → プロンプトが消えない）
gallery_count = len(st.session_state.gallery_images)
remaining = 50 - gallery_count
is_max = remaining <= 0

st.markdown("**🔢 生成枚数:**")
count_cols = st.columns(4)
for i, count in enumerate([3, 5, 10, 20]):
    with count_cols[i]:
        selected = st.session_state.gen_count == count
        st.button(
            f"{'✅ ' if selected else ''}{count}枚",
            key=f"count_{count}",
            use_container_width=True,
            type="primary" if selected else "secondary",
            on_click=set_gen_count,
            args=(count,),
        )

chosen_count = st.session_state.gen_count

# ボタンラベル
if is_max:
    btn_label = "🚫 最大50枚に達しました（リセットしてください）"
elif st.session_state.generating:
    btn_label = "⏳ 生成中..."
elif gallery_count == 0:
    btn_label = f"✨ 画像を生成する（{chosen_count}枚）"
else:
    target = min(gallery_count + chosen_count, 50)
    btn_label = f"✨ さらに{chosen_count}枚追加生成する（現在 {gallery_count} 枚 → {target} 枚）"

# 入力フォーム
with st.form(key="prompt_form"):
    prompt = st.text_area(
        "プロンプトまたは修正指示を入力してください... (例: オフィス背景で明るく)",
        value=st.session_state.current_prompt,
        height=400,
    )
    submit_button = st.form_submit_button(
        label=btn_label,
        use_container_width=True,
        type="primary",
        disabled=is_max or st.session_state.generating,
    )

if submit_button and prompt and not is_max and not st.session_state.generating:
    # 前回のエラーをクリア
    st.session_state.last_gen_errors = []
    st.session_state.last_gen_success = None
    # フォーム送信時のプロンプトを保持
    st.session_state.current_prompt = prompt
    save_prompt(prompt)

    if not st.session_state.api_key:
        st.error("左のサイドバーから Gemini API Key を設定してください。")
        st.stop()

    # 今回生成する枚数（選択した枚数、ただし上限50枚を超えない）
    num_to_generate = min(st.session_state.gen_count, 50 - len(st.session_state.gallery_images))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_num = len(st.session_state.gallery_images) + 1

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

    # バックグラウンドスレッドで生成開始
    sid = st.session_state.session_id
    state = get_gen_state(sid)

    with state.lock:
        state.running = True
        state.stop_requested = False
        state.images = []
        state.errors = []
        state.total = num_to_generate
        state.completed = 0
        state.success_count = 0
        state.status = "生成を開始しています..."
        state.finished = False

    thread = threading.Thread(
        target=generation_worker,
        args=(sid, st.session_state.api_key, contents, num_to_generate,
              output_dir, timestamp, start_num),
        daemon=True,
    )
    thread.start()

    st.session_state.generating = True
    st.rerun()
