import os
import io
import json
import time
import base64
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
import streamlit as st
from datetime import datetime
from pathlib import Path
from PIL import Image

from gallery_persistence import (
    append_gallery_images,
    clear_gallery,
    create_gallery_zip,
    load_gallery,
    normalize_retention_days,
)

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

# ギャラリーに蓄積できる最大枚数（この枚数に達するとリセットが必要になる）。
# 大きくしすぎると Streamlit Cloud のメモリ／表示が重くなる点だけ注意。
MAX_GALLERY = 200
DEFAULT_GENERATION_COUNT = 10
DEFAULT_CONCURRENCY = 10
DEFAULTS_VERSION = 20260628
ILLUSTRATION_MODE_KEY = f"illustration_mode_{DEFAULTS_VERSION}"
DEFAULT_GALLERY_RETENTION_DAYS = 7

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


def _generate_image_task(state, i, num_to_generate, provider, api_key, prompt,
                         image_bytes_list, output_dir, timestamp, start_num,
                         openai_model, openai_size, openai_quality, openai_crop_16_9):
    """画像を1枚生成する（リトライ込み）。ThreadPoolExecutor で並列実行される。"""
    MAX_RETRIES = 3
    if state.stop_requested:
        return

    img_num = start_num + i
    filename = f"replica_{timestamp}_{img_num:02d}.png"
    filepath = output_dir / filename

    for attempt in range(MAX_RETRIES):
        if state.stop_requested:
            return

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
                return  # 成功

            # エンジンが失敗理由を返した
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 * (attempt + 1))
                continue
            with state.lock:
                state.errors.append(f"画像 {img_num}: {err}")
            return

        except Exception as e:
            err_str = str(e)
            # レート制限 or 一時エラーはリトライ
            is_rate_limit = (
                "429" in err_str
                or "RESOURCE_EXHAUSTED" in err_str
                or "rate_limit" in err_str.lower()
            )
            if attempt < MAX_RETRIES - 1 and is_rate_limit:
                time.sleep(5 * (attempt + 1))
                continue
            with state.lock:
                state.errors.append(f"画像 {img_num}: {err_str[:200]}")
            return


def generation_worker(session_id, provider, api_key, prompt, image_bytes_list,
                      num_to_generate, output_dir, timestamp, start_num,
                      openai_model=OPENAI_IMAGE_MODEL_DEFAULT,
                      openai_size="1536x1024",
                      openai_quality="high",
                      openai_crop_16_9=True,
                      max_workers=DEFAULT_CONCURRENCY):
    """バックグラウンドスレッドで画像を並列生成するワーカー関数。

    各画像を ThreadPoolExecutor のタスクとして同時実行する。
    max_workers で同時実行数を制御（既定10）。API のレート制限に当たった場合は
    各タスク内のリトライ＋指数バックオフが自動で吸収する。
    """
    state = get_gen_state(session_id)
    engine_label = "OpenAI" if provider == "openai" else "Gemini"
    # 生成枚数より多いワーカーは無駄なので上限を絞る（最低1）
    workers = max(1, min(max_workers, num_to_generate))

    try:
        with state.lock:
            state.status = (
                f"[{engine_label}] 並列生成中（最大 {workers} 枚同時）"
                f" — 完了 0/{num_to_generate}"
            )

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _generate_image_task, state, i, num_to_generate, provider,
                    api_key, prompt, image_bytes_list, output_dir, timestamp,
                    start_num, openai_model, openai_size, openai_quality,
                    openai_crop_16_9,
                )
                for i in range(num_to_generate)
            ]
            for _ in as_completed(futures):
                with state.lock:
                    state.completed += 1
                    if state.stop_requested:
                        state.status = (
                            f"⏹️ 停止処理中...（{state.success_count}枚生成済み）"
                        )
                    else:
                        state.status = (
                            f"[{engine_label}] 並列生成中（最大 {workers} 枚同時）"
                            f" — 完了 {state.completed}/{num_to_generate}"
                            f"・成功 {state.success_count}"
                        )

        if state.stop_requested:
            with state.lock:
                state.status = f"⏹️ ユーザーにより停止（{state.success_count}枚生成済み）"

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
if "concurrency" not in st.session_state:
    st.session_state.concurrency = DEFAULT_CONCURRENCY

# ギャラリー蓄積用（ボタンを押すたびに追加、最大 MAX_GALLERY 枚）
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

# 🆕 2026-05-26: GitHub Gist 同期 (= デバイス横断の永続化・真の解決策)
# 真因: localStorage はブラウザ × デバイス単位のため デスクトップ PC と Mac で別 LS
#       → 別デバイスから見ると履歴が「消えた」ように見える
# 対策: GitHub Gist を 真の真実の源 (= source of truth) として 全デバイスから 同じ履歴 を参照
# 設定: Streamlit Secrets に github_token (= PAT, gist scope) を登録。
# gist_id は任意。未指定なら同じトークンの既存Gistを探し、初回保存時に自動作成する。
_GIST_FILENAME = "banana_past_prompts.json"
_GIST_DESCRIPTION = "Banana Thumbnail System persistence"


@st.cache_resource
def _init_gist_store():
    """全ブラウザセッションで共有するGistロックと自動検出ID。"""
    return {"lock": threading.RLock(), "gist_id": None}


_gist_store = _init_gist_store()


def _get_secret_alias(*keys):
    """Streamlit Secrets の大文字・小文字どちらの命名も受け付ける。"""
    for key in keys:
        try:
            value = st.secrets.get(key, "")
        except Exception:
            value = ""
        if value:
            return str(value).strip()
    return ""


def _get_gist_config():
    """Streamlit Secrets から GitHub Gist 設定を取得 (= 未設定なら None)。"""
    token = _get_secret_alias("github_token", "GITHUB_TOKEN")
    gist_id = _get_secret_alias("gist_id", "GIST_ID")
    if not token:
        return None
    return {"token": token, "gist_id": gist_id}


def _gist_request(cfg, url, *, method="GET", payload=None):
    """GitHub Gist API を呼ぶ。トークン値はログへ出さない。"""
    import urllib.request

    body = None
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method=method,
        headers={
            "Authorization": f"Bearer {cfg['token']}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "User-Agent": "banana-thumbnail-sync",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(req, timeout=8) as resp:
        raw = resp.read()
    return json.loads(raw.decode("utf-8")) if raw else {}


def _resolve_gist_id(cfg):
    """明示ID、同プロセスのキャッシュ、既存Gistの順で同期先を解決する。"""
    if cfg.get("gist_id"):
        _gist_store["gist_id"] = cfg["gist_id"]
        return cfg["gist_id"]
    if _gist_store.get("gist_id"):
        return _gist_store["gist_id"]

    data = _gist_request(cfg, "https://api.github.com/gists?per_page=100")
    if not isinstance(data, list):
        return None

    # 説明が一致するものを優先し、旧版のファイル名一致にも後方互換で対応。
    candidates = [
        gist for gist in data
        if isinstance(gist, dict)
        and _GIST_FILENAME in (gist.get("files") or {})
    ]
    matched = next(
        (gist for gist in candidates if gist.get("description") == _GIST_DESCRIPTION),
        candidates[0] if candidates else None,
    )
    if matched and matched.get("id"):
        _gist_store["gist_id"] = str(matched["id"])
        return _gist_store["gist_id"]
    return None


def _parse_gist_prompts(cfg, target):
    """通常レスポンスと1MB超Gistのraw_urlフォールバックを扱う。"""
    if not target:
        return []
    content = target.get("content", "")
    if target.get("truncated") and target.get("raw_url"):
        raw_data = _gist_request(cfg, target["raw_url"])
        # raw_url は JSON 文書そのものを返すため、_gist_request が既にparse済み。
        return _normalize_prompts(raw_data)
    if not content:
        return []
    try:
        return _normalize_prompts(json.loads(content))
    except Exception:
        return None


def _load_prompts_from_gist():
    """GitHub Gist から past_prompts を読み出す (= デバイス横断の真の保存先)。
    未設定 / 失敗時は None。 5 秒タイムアウト で UI ブロック回避。
    """
    cfg = _get_gist_config()
    if not cfg:
        return None
    try:
        with _gist_store["lock"]:
            gist_id = _resolve_gist_id(cfg)
            if not gist_id:
                return []  # トークン設定済み・まだ同期Gistなし = 初回
            data = _gist_request(cfg, f"https://api.github.com/gists/{gist_id}")
            target = (data.get("files") or {}).get(_GIST_FILENAME)
            return _parse_gist_prompts(cfg, target)
    except Exception as e:
        import sys
        print(f"[gist_load] ⚠️ Gist 取得失敗 (LS/ファイルで継続): {type(e).__name__}: {e}",
              file=sys.stderr)
        return None


def _sync_prompts_to_gist(prompts_list):
    """クラウド最新版を先にマージしてからGistへ保存する。

    戻り値は ``(成功したか, マージ済み履歴)``。取得失敗時は書き込まず、
    別デバイスの未取得データを古いブラウザから上書きしない。
    """
    cfg = _get_gist_config()
    if not cfg:
        return False, _normalize_prompts(prompts_list)
    try:
        with _gist_store["lock"]:
            remote = _load_prompts_from_gist()
            if remote is None:
                return False, _normalize_prompts(prompts_list)
            merged = _merge_prompts(prompts_list, remote)
            gist_id = _resolve_gist_id(cfg)
            file_payload = {
                _GIST_FILENAME: {
                    "content": json.dumps(merged, ensure_ascii=False, indent=2)
                }
            }
            if gist_id:
                _gist_request(
                    cfg,
                    f"https://api.github.com/gists/{gist_id}",
                    method="PATCH",
                    payload={"files": file_payload},
                )
            else:
                created = _gist_request(
                    cfg,
                    "https://api.github.com/gists",
                    method="POST",
                    payload={
                        "description": _GIST_DESCRIPTION,
                        "public": False,
                        "files": file_payload,
                    },
                )
                if not created.get("id"):
                    return False, merged
                _gist_store["gist_id"] = str(created["id"])
            return True, merged
    except Exception as e:
        import sys
        print(f"[gist_save] ⚠️ Gist 書込み失敗 (LS+ファイルは保存済): "
              f"{type(e).__name__}: {e}", file=sys.stderr)
        return False, _normalize_prompts(prompts_list)


def _save_prompts_to_gist(prompts_list):
    """後方互換用のboolラッパー。"""
    saved, _ = _sync_prompts_to_gist(prompts_list)
    return saved


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

# 🆕 Gist は初回セッション開始時のみ 取得 (= ネットワーク IO 抑制・session_state でキャッシュ)
if "_gist_loaded" not in st.session_state:
    _gist_current = _load_prompts_from_gist()
    st.session_state["_gist_loaded"] = True
    st.session_state["_gist_cached"] = _gist_current  # None なら未設定 or 取得失敗
else:
    _gist_current = st.session_state.get("_gist_cached")

if "past_prompts" not in st.session_state:
    # 初回ハイドレーション: Gist > LS > ファイル の優先順 + 全マージ
    # 真の真実の源 = Gist なので最優先・LS とファイルは ローカルに 何かあれば 拾う
    sources = []
    if _gist_current is not None:
        sources.append(_gist_current)
    if _ls_current:
        sources.append(_ls_current)
    file_loaded = _load_prompts_from_file()
    if file_loaded:
        sources.append(file_loaded)
    st.session_state.past_prompts = _merge_prompts(*sources)
    st.session_state["_ls_hydrated"] = bool(_ls_current or _gist_current)
else:
    # 既にセッションに履歴がある状態で、LS から初めて値が取れたときにマージ
    # （初回 None → 次リランで値 というLSの非同期取得を確実に拾うため）
    if _ls_current and not st.session_state.get("_ls_hydrated", False):
        st.session_state.past_prompts = _merge_prompts(
            _ls_current, st.session_state.past_prompts
        )
        st.session_state["_ls_hydrated"] = True


def save_prompt(new_prompt):
    """プロンプトを履歴の先頭に追加し、Gist／ファイル／LSへ保存する。

    🚨 2026-05-09 改: 順序を「ファイル → LS」 に逆転 (旧: LS → ファイル)
    旧版は LS write (streamlit_local_storage.setItem) が稀に rerun 例外を投げて、
    後続のファイル write がスキップされる事象が発生していた。
    ファイル mtime が 2/28 で固定されており、 ブラウザ LS が消えると新プロンプトが
    全て失われていた (社長 5/9 報告)。

    現行版では、まずGist最新版を取得してマージする。これにより、先に開いていた
    PCブラウザがスマホ側の新しい履歴を古い状態で上書きする競合も防止する。
    その後、アトミックなファイル書込み → LS の順でローカル退避する。
    """
    if not new_prompt:
        return
    # 保存直前に LS の最新値を取り込み（別タブ等の並行書き込み対策）
    ls_latest = _load_prompts_from_ls()
    base = _merge_prompts(st.session_state.past_prompts, ls_latest or [])
    # 新規プロンプトを先頭に移動（重複回避）
    if new_prompt in base:
        base.remove(new_prompt)
    base.insert(0, new_prompt)

    # ① Gist — 保存前にクラウド最新版をマージ（lost update 防止）。
    gist_saved, gist_merged = _sync_prompts_to_gist(base[:50])
    st.session_state.past_prompts = gist_merged if gist_saved else base[:50]
    if gist_saved:
        st.session_state["_gist_cached"] = list(st.session_state.past_prompts)

    # ② サーバー側ファイル — ブラウザ間共有のフォールバック (アトミック書込み)
    try:
        tmp_path = past_prompts_file.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(st.session_state.past_prompts, f, ensure_ascii=False, indent=2)
        # rename で原子的置換 (途中クラッシュでも本体は破損しない)
        os.replace(tmp_path, past_prompts_file)
    except Exception as e:
        import sys
        print(
            f"[save_prompt] ❌ ファイル書込み失敗: {type(e).__name__}: {e} "
            f"(path={past_prompts_file})",
            file=sys.stderr,
        )

    # ③ localStorage — 同じブラウザ向けの二次保存 (失敗してもファイルが残る)
    if _ls_instance is not None:
        try:
            _ls_instance.setItem(
                _LS_KEY,
                json.dumps(st.session_state.past_prompts, ensure_ascii=False),
            )
        except Exception as e:
            import sys
            print(
                f"[save_prompt] ⚠️ LocalStorage 書込み失敗 (ファイルは保存済): "
                f"{type(e).__name__}: {e}",
                file=sys.stderr,
            )

# 出力ディレクトリ
output_dir = Path(__file__).parent / "replica_output"
output_dir.mkdir(exist_ok=True, parents=True)

# ブラウザを閉じても復元できる共有ギャラリー。Streamlit Cloud の実行領域なので
# サーバー再起動時の保証はなく、確実に残す用途には下の ZIP ダウンロードを使う。
gallery_retention_days = normalize_retention_days(
    _get_secret_alias("gallery_retention_days", "GALLERY_RETENTION_DAYS")
    or os.environ.get("GALLERY_RETENTION_DAYS")
    or DEFAULT_GALLERY_RETENTION_DAYS,
    default=DEFAULT_GALLERY_RETENTION_DAYS,
)
st.session_state.gallery_images = load_gallery(
    output_dir,
    retention_days=gallery_retention_days,
    max_images=MAX_GALLERY,
)

# プロンプト入力用のセッション状態を初期化
if "current_prompt" not in st.session_state:
    st.session_state.current_prompt = ""

def set_prompt(text):
    st.session_state.current_prompt = text

# 生成枚数の選択用コールバック
if "gen_count" not in st.session_state:
    st.session_state.gen_count = DEFAULT_GENERATION_COUNT

# Migrate already-open Streamlit sessions from the old 5-image defaults once.
if st.session_state.get("_thumbnail_defaults_version") != DEFAULTS_VERSION:
    if st.session_state.get("concurrency", 5) == 5:
        st.session_state.concurrency = DEFAULT_CONCURRENCY
    if st.session_state.get("gen_count", 5) == 5:
        st.session_state.gen_count = DEFAULT_GENERATION_COUNT
    st.session_state._thumbnail_defaults_version = DEFAULTS_VERSION

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

        # 共有マニフェストへ先に追記し、別ブラウザでも復元できる状態にする。
        old_names = {img.name for img in st.session_state.gallery_images}
        st.session_state.gallery_images = append_gallery_images(
            output_dir,
            new_images,
            retention_days=gallery_retention_days,
            max_images=MAX_GALLERY,
        )

        # ギャラリーインデックスを新しい画像の先頭に移動
        new_indices = [
            idx for idx, img in enumerate(st.session_state.gallery_images)
            if img.name not in old_names
        ]
        if new_indices:
            st.session_state["gidx_main_gallery"] = new_indices[0]
        st.session_state.pop("gallery_archive_path", None)

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

    # --- 同時生成数（並列化） ---
    st.slider(
        "⚡ 同時生成数",
        min_value=1,
        max_value=20,
        value=st.session_state.concurrency,
        key="concurrency",
        help=(
            "複数の画像を同時に生成して高速化する。\n"
            "Gemini: 有料Tier1は150〜300 RPM なので 10〜20 でも基本OK"
            "（時間帯やプレビュー枠の混雑で一時的に429が出ても自動リトライで吸収）。\n"
            "OpenAI: 低Tier（Tier1=5枚/分）では大きくするとレート制限に"
            "当たりやすい（自動リトライで吸収するが速度は頭打ち）。\n"
            "1 にすると従来通り1枚ずつ順番に生成。"
        ),
    )

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
        options=["焦っている（固定）", "グッド（固定）", "通常", "含めない"],
        index=0,  # デフォルトは「焦っている（固定）」
        key=ILLUSTRATION_MODE_KEY,
    )
    # 選択に応じた画像ファイルのパスを決定
    if illustration_mode == "通常":
        ill_path = Path(__file__).parent / "illustration.png"
    elif illustration_mode == "グッド（固定）":
        ill_path = Path(__file__).parent / "illustration_good.png"
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

        # 🆕 Gist 同期ステータス表示
        _gist_cfg = _get_gist_config()
        _gist_cached = st.session_state.get("_gist_cached")
        if _gist_cfg is None:
            st.warning(
                "☁️ **Gist 同期: 未設定** "
                "(デバイス横断永続化を有効にするには Streamlit Secrets に "
                "`github_token` (PAT, gist scope) を登録してください。"
                "`gist_id` は任意です)"
            )
        elif _gist_cached is None:
            st.error(
                "☁️ **Gist 同期: エラー** (取得失敗・LS/ファイルで継続中。 "
                "PAT 権限 / GIST_ID を確認してください)"
            )
        else:
            st.success(f"☁️ **Gist 同期: 有効** (Gist 保存: {len(_gist_cached)} 件)")
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("🔄 Gist から再取得", use_container_width=True):
                    fresh = _load_prompts_from_gist()
                    if fresh is not None:
                        st.session_state["_gist_cached"] = fresh
                        st.session_state.past_prompts = _merge_prompts(
                            fresh, st.session_state.past_prompts
                        )
                        st.success(f"✅ {len(fresh)} 件を再取得")
                        st.rerun()
                    else:
                        st.error("再取得失敗")
            with col_b:
                if st.button("☁️ Gist へ手動 push", use_container_width=True):
                    saved, merged = _sync_prompts_to_gist(
                        st.session_state.past_prompts
                    )
                    if saved:
                        st.session_state.past_prompts = merged
                        st.session_state["_gist_cached"] = list(merged)
                        st.success("✅ Gist 更新成功")
                    else:
                        st.error("Gist 更新失敗")

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
                    gist_saved, gist_merged = _sync_prompts_to_gist(merged)
                    if gist_saved:
                        st.session_state.past_prompts = gist_merged
                        st.session_state["_gist_cached"] = list(gist_merged)
                    st.success(f"{len(merged)} 件に復元しました")
                else:
                    st.error("JSONの形式が正しくありません（文字列の配列が必要）")
            except Exception as e:
                st.error(f"読み込みに失敗しました: {e}")

    # ギャラリー状況の表示
    st.markdown("---")
    gallery_count = len(st.session_state.gallery_images)
    st.markdown(f"**📊 現在のギャラリー: {gallery_count} / {MAX_GALLERY} 枚**")
    st.caption(
        f"直近 {gallery_retention_days} 日分を共有ギャラリーに保持します。"
        "ブラウザを閉じても復元できますが、Streamlitサーバー再起動時は"
        "消える場合があるため、必要な画像はZIPでも保存してください。"
    )

    if st.button(
        "🔄 共有ギャラリーを再読み込み",
        key="refresh_shared_gallery",
        use_container_width=True,
    ):
        st.session_state.gallery_images = load_gallery(
            output_dir,
            retention_days=gallery_retention_days,
            max_images=MAX_GALLERY,
        )
        st.rerun()

    if gallery_count > 0:
        if st.button(
            "📦 全画像のZIPを準備",
            key="prepare_gallery_zip",
            use_container_width=True,
        ):
            with st.spinner("ZIPを作成しています..."):
                archive_path = create_gallery_zip(
                    output_dir,
                    st.session_state.gallery_images,
                    archive_key=st.session_state.session_id,
                )
            st.session_state["gallery_archive_path"] = str(archive_path)

        archive_value = st.session_state.get("gallery_archive_path")
        if archive_value:
            archive_path = Path(archive_value)
            if archive_path.exists():
                with open(archive_path, "rb") as archive_file:
                    st.download_button(
                        "💾 ZIPをダウンロード",
                        data=archive_file.read(),
                        file_name=(
                            "banana_thumbnails_"
                            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
                        ),
                        mime="application/zip",
                        key="download_gallery_zip",
                        use_container_width=True,
                    )

        if st.button(
            "🆕 ギャラリーをリセット（新しく始める）",
            key="reset_shared_gallery",
            use_container_width=True,
        ):
            clear_gallery(output_dir)
            st.session_state.gallery_images = []
            st.session_state.pop("gallery_archive_path", None)
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
remaining = MAX_GALLERY - gallery_count
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
    btn_label = f"🚫 最大{MAX_GALLERY}枚に達しました（リセットしてください）"
elif st.session_state.generating:
    btn_label = "⏳ 生成中..."
elif gallery_count == 0:
    btn_label = f"✨ 画像を生成する（{chosen_count}枚）"
else:
    target = min(gallery_count + chosen_count, MAX_GALLERY)
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

    # 今回生成する枚数（選択した枚数、ただし上限 MAX_GALLERY 枚を超えない）
    num_to_generate = min(st.session_state.gen_count, MAX_GALLERY - len(st.session_state.gallery_images))
    # 複数ブラウザから同時生成してもファイル名が衝突しないようマイクロ秒まで含める。
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    start_num = len(st.session_state.gallery_images) + 1

    # 参考画像のバイト列を収集（エンジンに依存しない形で渡す）
    image_bytes_list = []

    selected_mode = st.session_state.get(ILLUSTRATION_MODE_KEY, "焦っている（固定）")
    if selected_mode == "通常":
        gen_ill_path = Path(__file__).parent / "illustration.png"
    elif selected_mode == "グッド（固定）":
        gen_ill_path = Path(__file__).parent / "illustration_good.png"
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
            st.session_state.concurrency,
        ),
        daemon=True,
    )
    thread.start()

    st.session_state.generating = True
    st.rerun()
