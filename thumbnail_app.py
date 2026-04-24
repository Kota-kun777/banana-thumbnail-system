import os
import io
import json
import time
import base64
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

# OpenAI Images 2.0（gpt-image-2 系）を任意で利用
try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

# ブラウザのlocalStorageに永続保存するためのコンポーネント
# （Streamlit Cloudのファイルシステムは揮発性のため、サーバー側ファイル保存だけでは
#  コンテナ再起動時にプロンプト履歴が消えてしまう問題への対処）
try:
    from streamlit_local_storage import LocalStorage
    _LS_AVAILABLE = True
except ImportError:
    _LS_AVAILABLE = False

# ==============================================================
# 画像生成モデル設定
# ==============================================================
GEMINI_IMAGE_MODEL = "gemini-3-pro-image-preview"
# OpenAI は 2026年時点の最新 gpt-image-2 をデフォルトに。アクセス不可の場合は
# サイドバーの詳細設定で gpt-image-1 系へ切り替え可能。
OPENAI_IMAGE_MODEL_DEFAULT = "gpt-image-2"
# gpt-image-2 は「幅・高さとも16の倍数」制約あり（1920x1080 等は NG）。
# 以下は 16:9 ぴったり＋16の倍数を満たすサイズ。
OPENAI_SIZE_OPTIONS = [
    "2048x1152",   # 16:9 高解像（2K相当）
    "1792x1008",   # 16:9 中解像
    "1536x864",    # 16:9 標準解像（推奨）
    "1024x576",    # 16:9 低解像（高速生成）
    "1024x1024",   # 1:1 正方形
    "1024x1536",   # 2:3 縦長
]
OPENAI_QUALITY_OPTIONS = ["high", "medium", "low", "auto"]

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


@st.cache_resource
def _init_gen_store():
    """スクリプト再実行でもリセットされない永続ストア"""
    return {"states": {}, "lock": threading.Lock()}

_gen_store = _init_gen_store()


def get_gen_state(session_id):
    with _gen_store["lock"]:
        if session_id not in _gen_store["states"]:
            _gen_store["states"][session_id] = GenerationState()
        return _gen_store["states"][session_id]


def _generate_one_gemini(api_key, prompt, image_bytes_list):
    """Gemini 3 Pro で画像を1枚生成。成功: (bytes, None) / 失敗: (None, 理由)"""
    client = genai.Client(api_key=api_key)
    contents = [prompt]
    for img_bytes in image_bytes_list:
        contents.append(types.Part.from_bytes(data=img_bytes, mime_type="image/png"))

    response = client.models.generate_content(
        model=GEMINI_IMAGE_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
        ),
    )

    if not response.candidates:
        block_reason = ""
        if hasattr(response, "prompt_feedback") and response.prompt_feedback:
            block_reason = str(getattr(response.prompt_feedback, "block_reason", ""))
        return None, f"ブロック（{block_reason}）" if block_reason else "ブロック"

    text_response = ""
    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            return part.inline_data.data, None
        if part.text:
            text_response = part.text

    if text_response:
        return None, f"画像データなし（API応答: {text_response[:100]}）"
    return None, "画像データなし"


def _crop_to_16_9(img_bytes):
    """画像を中央クロップで 16:9 に整える。失敗時は元のまま返す。"""
    try:
        img = Image.open(io.BytesIO(img_bytes))
        w, h = img.size
        target = 16 / 9
        current = w / h
        if abs(current - target) < 0.01:
            return img_bytes  # 既に16:9
        if current > target:
            # 横に広すぎ → 左右を切る
            new_w = int(round(h * target))
            left = (w - new_w) // 2
            cropped = img.crop((left, 0, left + new_w, h))
        else:
            # 縦に長い → 上下を切る（1536x1024 → 1536x864 など）
            new_h = int(round(w / target))
            top = (h - new_h) // 2
            cropped = img.crop((0, top, w, top + new_h))
        buf = io.BytesIO()
        cropped.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return img_bytes


def _generate_one_openai(api_key, prompt, image_bytes_list, model, size, quality,
                          crop_16_9=True):
    """OpenAI Images 2.0 で画像を1枚生成。成功: (bytes, None) / 失敗: (None, 理由)"""
    if not _OPENAI_AVAILABLE:
        return None, "openai パッケージが未インストール"

    client = OpenAI(api_key=api_key)
    kwargs = {
        "model": model,
        "prompt": prompt,
        "size": size,
        "quality": quality,
        "n": 1,
    }

    if image_bytes_list:
        # ファイル風オブジェクトにnameを付けて渡す（SDKがmime判定に使う）
        image_files = []
        for i, img_bytes in enumerate(image_bytes_list):
            bio = io.BytesIO(img_bytes)
            bio.name = f"input_{i}.png"
            image_files.append(bio)
        kwargs["image"] = image_files
        result = client.images.edit(**kwargs)
    else:
        result = client.images.generate(**kwargs)

    data = result.data[0]
    raw = None
    if getattr(data, "b64_json", None):
        raw = base64.b64decode(data.b64_json)
    elif getattr(data, "url", None):
        import urllib.request
        with urllib.request.urlopen(data.url) as resp:
            raw = resp.read()

    if raw is None:
        return None, "画像データなし"

    if crop_16_9:
        raw = _crop_to_16_9(raw)
    return raw, None


def generation_worker(session_id, provider, api_key, prompt, image_bytes_list,
                      num_to_generate, output_dir, timestamp, start_num,
                      openai_model=OPENAI_IMAGE_MODEL_DEFAULT,
                      openai_size="1536x1024",
                      openai_quality="high",
                      openai_crop_16_9=True):
    """バックグラウンドスレッドで画像を生成するワーカー関数"""
    state = get_gen_state(session_id)
    MAX_RETRIES = 3

    try:
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
                engine_label = "OpenAI" if provider == "openai" else "Gemini"
                with state.lock:
                    state.status = (
                        f"[{engine_label}] 生成中 [{i + 1}/{num_to_generate}] "
                        f"...（合計 {img_num} 枚目）{retry_label}"
                    )

                try:
                    if provider == "openai":
                        img_bytes, err = _generate_one_openai(
                            api_key, prompt, image_bytes_list,
                            openai_model, openai_size, openai_quality,
                            crop_16_9=openai_crop_16_9,
                        )
                    else:
                        img_bytes, err = _generate_one_gemini(
                            api_key, prompt, image_bytes_list,
                        )

                    if img_bytes is not None:
                        with open(filepath, "wb") as f:
                            f.write(img_bytes)
                        with state.lock:
                            state.images.append(str(filepath))
                            state.success_count += 1
                        break  # 成功 → 次の画像へ

                    # エンジンが失敗理由を返した
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(2 * (attempt + 1))
                        continue
                    with state.lock:
                        state.errors.append(f"画像 {img_num}: {err}")
                    break

                except Exception as e:
                    err_str = str(e)
                    # レート制限 or 一時エラーはリトライ
                    is_rate_limit = (
                        "429" in err_str
                        or "RESOURCE_EXHAUSTED" in err_str
                        or "rate_limit" in err_str.lower()
                    )
                    if attempt < MAX_RETRIES - 1 and is_rate_limit:
                        with state.lock:
                            state.status = (
                                f"⏳ レート制限のため待機中... "
                                f"[{i + 1}/{num_to_generate}]（{attempt + 1}回目）"
                            )
                        time.sleep(5 * (attempt + 1))
                        continue
                    with state.lock:
                        state.errors.append(f"画像 {img_num}: {err_str[:200]}")
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
def _get_secret(key):
    try:
        v = st.secrets.get(key, "")
        if v:
            return v
    except (FileNotFoundError, KeyError):
        pass
    return ""


def get_gemini_api_key():
    return (
        _get_secret("GEMINI_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or ""
    )


def get_openai_api_key():
    return _get_secret("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""


if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = get_gemini_api_key()
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = get_openai_api_key()
if "provider_key" not in st.session_state:
    st.session_state.provider_key = "gemini"
if "openai_model" not in st.session_state:
    st.session_state.openai_model = OPENAI_IMAGE_MODEL_DEFAULT
if "openai_size" not in st.session_state:
    st.session_state.openai_size = OPENAI_SIZE_OPTIONS[0]
if "openai_quality" not in st.session_state:
    st.session_state.openai_quality = OPENAI_QUALITY_OPTIONS[0]
if "openai_crop_16_9" not in st.session_state:
    st.session_state.openai_crop_16_9 = True

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
# 永続化戦略:
#   - ブラウザの localStorage を主ストレージにする（Streamlit Cloud のコンテナ再起動で
#     サーバー側ファイルが git の状態に戻ってしまっても、ブラウザには履歴が残る）
#   - サーバー側 JSON はフォールバック（LS無効時・別ブラウザからの初回アクセス時）
#   - LS 取得は非同期（初回 None → 次リランで値）。毎リラン取得を試みて、値が
#     取れたタイミングで session_state にマージする
past_prompts_file = Path(__file__).parent / "past_prompts.json"

# LocalStorage インスタンス
_ls_instance = LocalStorage() if _LS_AVAILABLE else None
_LS_KEY = "banana_past_prompts"


def _normalize_prompts(data):
    """list[str] に正規化。不正な形式は空リストに。"""
    if not isinstance(data, list):
        return []
    return [x for x in data if isinstance(x, str) and x.strip()]


def _load_prompts_from_ls():
    """localStorage から past_prompts を読み出す。未取得／失敗時は None。"""
    if _ls_instance is None:
        return None
    try:
        raw = _ls_instance.getItem(_LS_KEY)
    except Exception:
        return None
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except Exception:
            return None
        return _normalize_prompts(parsed)
    if isinstance(raw, list):
        return _normalize_prompts(raw)
    return None


def _load_prompts_from_file():
    """サーバー側JSONファイルから past_prompts を読み出す。"""
    if not past_prompts_file.exists():
        return []
    try:
        with open(past_prompts_file, "r", encoding="utf-8") as f:
            return _normalize_prompts(json.load(f))
    except Exception:
        return []


def _merge_prompts(*lists):
    """複数のプロンプトリストを順序・重複考慮でマージ（先頭優先）、最大50件。"""
    seen = set()
    result = []
    for lst in lists:
        if not lst:
            continue
        for p in lst:
            if p and p not in seen:
                seen.add(p)
                result.append(p)
    return result[:50]


# 毎リラン LS から値を取得（非同期取得の遅延対策のため都度実行）
_ls_current = _load_prompts_from_ls()

if "past_prompts" not in st.session_state:
    # 初回ハイドレーション: LS が取れていればそれを優先、なければファイルから
    if _ls_current:
        st.session_state.past_prompts = list(_ls_current)
        st.session_state["_ls_hydrated"] = True
    else:
        st.session_state.past_prompts = _load_prompts_from_file()
        st.session_state["_ls_hydrated"] = False
else:
    # 既にセッションに履歴がある状態で、LS から初めて値が取れたときにマージ
    # （初回 None → 次リランで値 というLSの非同期取得を確実に拾うため）
    if _ls_current and not st.session_state.get("_ls_hydrated", False):
        st.session_state.past_prompts = _merge_prompts(
            _ls_current, st.session_state.past_prompts
        )
        st.session_state["_ls_hydrated"] = True


def save_prompt(new_prompt):
    """プロンプトを履歴の先頭に追加し、LS／ファイル両方へ保存する。"""
    if not new_prompt:
        return
    # 保存直前に LS の最新値を取り込み（別タブ等の並行書き込み対策）
    ls_latest = _load_prompts_from_ls()
    base = _merge_prompts(st.session_state.past_prompts, ls_latest or [])
    # 新規プロンプトを先頭に移動（重複回避）
    if new_prompt in base:
        base.remove(new_prompt)
    base.insert(0, new_prompt)
    st.session_state.past_prompts = base[:50]

    # localStorage に保存（永続）
    if _ls_instance is not None:
        try:
            _ls_instance.setItem(
                _LS_KEY,
                json.dumps(st.session_state.past_prompts, ensure_ascii=False),
            )
        except Exception:
            pass  # LS書き込み失敗はサイレントに無視
    # サーバー側ファイルにもバックアップ保存（コンテナ存命中のみ有効）
    try:
        with open(past_prompts_file, "w", encoding="utf-8") as f:
            json.dump(st.session_state.past_prompts, f, ensure_ascii=False, indent=2)
    except Exception:
        pass  # 読み取り専用FS等でも動くように

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

    # --- 画像生成モデル選択 ---
    provider_display = st.radio(
        "画像生成モデル",
        options=["🍌 Gemini 3 Pro (nano-banana)", "🎨 OpenAI Images 2.0"],
        index=0 if st.session_state.provider_key == "gemini" else 1,
        key="provider_display",
        help=(
            "Gemini: 実在人物のイラスト化が緩い／文字再現はやや弱い\n"
            "OpenAI: 文字・構図が安定／実在の政治家などはほぼ生成不可"
        ),
    )
    provider_key = "gemini" if "Gemini" in provider_display else "openai"
    st.session_state.provider_key = provider_key

    # --- 選択モデルの APIキー状態 ---
    gemini_from_secrets = bool(_get_secret("GEMINI_API_KEY"))
    openai_from_secrets = bool(_get_secret("OPENAI_API_KEY"))

    if provider_key == "gemini":
        if gemini_from_secrets:
            st.success("✅ Gemini API Key 設定済み")
        else:
            gem_input = st.text_input(
                "Gemini API Key",
                value=st.session_state.gemini_api_key,
                type="password",
                key="gem_api_input",
            )
            if gem_input != st.session_state.gemini_api_key:
                st.session_state.gemini_api_key = gem_input
                st.rerun()
    else:
        if openai_from_secrets:
            st.success("✅ OpenAI API Key 設定済み")
        else:
            oai_input = st.text_input(
                "OpenAI API Key",
                value=st.session_state.openai_api_key,
                type="password",
                key="oai_api_input",
            )
            if oai_input != st.session_state.openai_api_key:
                st.session_state.openai_api_key = oai_input
                st.rerun()

        # OpenAI 詳細設定（デフォルトで良ければ触らなくてOK）
        with st.expander("🔧 OpenAI 詳細設定"):
            st.caption(f"モデル: `{OPENAI_IMAGE_MODEL_DEFAULT}` 固定")
            # 既定を 1536x864（16:9 ぴったり・標準解像）に寄せる
            if st.session_state.openai_size not in OPENAI_SIZE_OPTIONS:
                default_size_idx = OPENAI_SIZE_OPTIONS.index("1536x864")
            else:
                default_size_idx = OPENAI_SIZE_OPTIONS.index(st.session_state.openai_size)
            st.selectbox(
                "サイズ",
                options=OPENAI_SIZE_OPTIONS + ["カスタム"],
                index=default_size_idx,
                key="openai_size_choice",
                help=(
                    "2048x1152 / 1792x1008 / 1536x864 / 1024x576 は 16:9 "
                    "ぴったりでネイティブ生成（クロップ不要）。"
                    "カスタムは幅・高さとも16の倍数であること"
                ),
            )
            if st.session_state.openai_size_choice == "カスタム":
                custom_default = (
                    st.session_state.openai_size
                    if st.session_state.openai_size not in OPENAI_SIZE_OPTIONS
                    else "2048x1152"
                )
                custom_in = st.text_input(
                    "カスタムサイズ（幅・高さとも16の倍数）",
                    value=custom_default,
                    key="openai_size_custom",
                    help="例: 2048x1152（16:9）、1792x1008、1344x768 など",
                )
                st.session_state.openai_size = custom_in.strip()
            else:
                st.session_state.openai_size = st.session_state.openai_size_choice
            st.selectbox(
                "品質",
                options=OPENAI_QUALITY_OPTIONS,
                index=OPENAI_QUALITY_OPTIONS.index(st.session_state.openai_quality)
                if st.session_state.openai_quality in OPENAI_QUALITY_OPTIONS
                else 0,
                key="openai_quality",
            )
            st.checkbox(
                "16:9 にクロップ（YouTube向け）",
                value=st.session_state.openai_crop_16_9,
                key="openai_crop_16_9",
                help=(
                    "3:2 など 16:9 でないサイズを生成した場合に、中央"
                    "クロップで 16:9 に整える。1920x1080 のような 16:9 "
                    "ネイティブサイズでは何もしない"
                ),
            )

    st.header("👤 キャラクター設定")
    illustration_mode = st.radio(
        "すあし社長のイラスト",
        options=["焦っている（固定）", "通常", "含めない"],
        index=0,  # デフォルトは「焦っている（固定）」
        key="illustration_mode",
    )
    # 選択に応じた画像ファイルのパスを決定
    if illustration_mode == "通常":
        ill_path = Path(__file__).parent / "illustration.png"
    elif illustration_mode == "焦っている（固定）":
        ill_path = Path(__file__).parent / "illustration_panic.png"
    else:
        ill_path = None

    if ill_path is not None and not ill_path.exists():
        st.warning(f"{ill_path.name} が見つかりません。")

    # 旧APIとの互換のためのフラグ
    use_illustration = (illustration_mode != "含めない")

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

    # プロンプト履歴のバックアップ・復元（万一 LS もファイルも失われた際の保険）
    st.markdown("---")
    with st.expander("🗃️ 履歴のバックアップ／復元"):
        ls_cnt = len(_ls_current) if _ls_current else 0
        file_cnt = len(_load_prompts_from_file())
        ss_cnt = len(st.session_state.past_prompts)
        st.caption(
            f"ブラウザ保存: {ls_cnt} 件 ／ サーバー保存: {file_cnt} 件 ／ 表示中: {ss_cnt} 件"
        )
        if ss_cnt > 0:
            st.download_button(
                "💾 履歴をJSONでダウンロード",
                data=json.dumps(
                    st.session_state.past_prompts, ensure_ascii=False, indent=2
                ),
                file_name=f"banana_prompts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
            )
        restored_file = st.file_uploader(
            "📥 履歴JSONをアップロードして復元",
            type=["json"],
            key="history_uploader",
        )
        if restored_file is not None:
            try:
                imported = _normalize_prompts(json.load(restored_file))
                if imported:
                    merged = _merge_prompts(imported, st.session_state.past_prompts)
                    st.session_state.past_prompts = merged
                    if _ls_instance is not None:
                        try:
                            _ls_instance.setItem(
                                _LS_KEY,
                                json.dumps(merged, ensure_ascii=False),
                            )
                        except Exception:
                            pass
                    try:
                        with open(past_prompts_file, "w", encoding="utf-8") as f:
                            json.dump(merged, f, ensure_ascii=False, indent=2)
                    except Exception:
                        pass
                    st.success(f"{len(merged)} 件に復元しました")
                else:
                    st.error("JSONの形式が正しくありません（文字列の配列が必要）")
            except Exception as e:
                st.error(f"読み込みに失敗しました: {e}")

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

    provider = st.session_state.get("provider_key", "gemini")
    if provider == "openai":
        api_key = st.session_state.openai_api_key
        if not api_key:
            st.error("左のサイドバーから OpenAI API Key を設定してください。")
            st.stop()
    else:
        api_key = st.session_state.gemini_api_key
        if not api_key:
            st.error("左のサイドバーから Gemini API Key を設定してください。")
            st.stop()

    # 今回生成する枚数（選択した枚数、ただし上限50枚を超えない）
    num_to_generate = min(st.session_state.gen_count, 50 - len(st.session_state.gallery_images))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_num = len(st.session_state.gallery_images) + 1

    # 参考画像のバイト列を収集（エンジンに依存しない形で渡す）
    image_bytes_list = []

    selected_mode = st.session_state.get("illustration_mode", "焦っている（固定）")
    if selected_mode == "通常":
        gen_ill_path = Path(__file__).parent / "illustration.png"
    elif selected_mode == "焦っている（固定）":
        gen_ill_path = Path(__file__).parent / "illustration_panic.png"
    else:
        gen_ill_path = None

    if gen_ill_path is not None and gen_ill_path.exists():
        with open(gen_ill_path, "rb") as f:
            image_bytes_list.append(f.read())

    if uploaded_files:
        for file in uploaded_files:
            image_bytes_list.append(file.getvalue())

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
        args=(
            sid, provider, api_key, prompt, image_bytes_list, num_to_generate,
            output_dir, timestamp, start_num,
            st.session_state.openai_model,
            st.session_state.openai_size,
            st.session_state.openai_quality,
            st.session_state.openai_crop_16_9,
        ),
        daemon=True,
    )
    thread.start()

    st.session_state.generating = True
    st.rerun()
