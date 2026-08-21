"""Recent-gallery persistence helpers for the Streamlit app.

The Streamlit session is intentionally not the source of truth here.  A small
manifest in ``replica_output`` makes recent images visible again after a tab is
closed or when the app is opened from another browser connected to the same
server instance.
"""

from __future__ import annotations

import json
import os
import threading
import time
import zipfile
from pathlib import Path
from typing import Iterable


MANIFEST_FILENAME = "gallery_manifest.json"
BATCHES_FILENAME = "generation_batches.json"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
_LOCK = threading.RLock()


def normalize_retention_days(value: object, default: int = 7) -> int:
    """Return a safe retention period between one and thirty days."""
    try:
        days = int(value)
    except (TypeError, ValueError):
        days = default
    return max(1, min(days, 30))


def _manifest_path(output_dir: Path) -> Path:
    return output_dir / MANIFEST_FILENAME


def _batches_path(output_dir: Path) -> Path:
    return output_dir / BATCHES_FILENAME


def _safe_image_path(output_dir: Path, name: object) -> Path | None:
    """Resolve one manifest entry without allowing traversal outside the store."""
    if not isinstance(name, str) or not name or Path(name).name != name:
        return None
    candidate = output_dir / name
    if candidate.suffix.lower() not in IMAGE_SUFFIXES:
        return None
    return candidate


def _read_manifest_unlocked(output_dir: Path) -> tuple[list[str], float] | None:
    path = _manifest_path(output_dir)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return [], 0.0
    # v1 used a bare list. v2 also records the last reset time so an image that
    # finishes in the background after a browser closes can be auto-discovered
    # without resurrecting images that the user explicitly reset.
    if isinstance(data, list):
        raw_names = data
        reset_at = 0.0
    elif isinstance(data, dict) and isinstance(data.get("images"), list):
        raw_names = data["images"]
        try:
            reset_at = max(0.0, float(data.get("reset_at", 0)))
        except (TypeError, ValueError):
            reset_at = 0.0
    else:
        return [], 0.0
    result: list[str] = []
    seen: set[str] = set()
    for item in raw_names:
        image_path = _safe_image_path(output_dir, item)
        if image_path is not None and image_path.name not in seen:
            seen.add(image_path.name)
            result.append(image_path.name)
    return result, reset_at


def _write_manifest_unlocked(
    output_dir: Path,
    names: list[str],
    *,
    reset_at: float = 0.0,
) -> None:
    output_dir.mkdir(exist_ok=True, parents=True)
    path = _manifest_path(output_dir)
    temp_path = path.with_suffix(".json.tmp")
    temp_path.write_text(
        json.dumps(
            {"version": 2, "reset_at": reset_at, "images": names},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    os.replace(temp_path, path)


def _is_recent(path: Path, cutoff: float) -> bool:
    try:
        return path.is_file() and path.stat().st_mtime >= cutoff
    except OSError:
        return False


def _prune_expired_unlocked(output_dir: Path, cutoff: float) -> None:
    """Remove expired images and stale ZIP exports from the ephemeral server."""
    if not output_dir.exists():
        return
    for path in output_dir.iterdir():
        try:
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
            elif path.is_file() and path.name.startswith("gallery_export_"):
                # ZIP files are only download staging files; one day is ample.
                if path.stat().st_mtime < time.time() - 24 * 60 * 60:
                    path.unlink()
        except OSError:
            # A concurrent download/generation must not make the app unavailable.
            continue


def _load_gallery_unlocked(
    output_dir: Path,
    *,
    retention_days: int,
    max_images: int,
) -> list[Path]:
    output_dir.mkdir(exist_ok=True, parents=True)
    cutoff = time.time() - retention_days * 24 * 60 * 60
    _prune_expired_unlocked(output_dir, cutoff)

    manifest = _read_manifest_unlocked(output_dir)
    if manifest is None:
        names, reset_at = [], 0.0
    else:
        names, reset_at = manifest

    # A Streamlit fragment stops when its tab disconnects. The background image
    # thread may still finish, so recover recent unmanifested files on next open.
    effective_cutoff = max(cutoff, reset_at)
    known_names = set(names)
    discovered = [
        path
        for path in output_dir.iterdir()
        if path.suffix.lower() in IMAGE_SUFFIXES
        and path.name not in known_names
        and _is_recent(path, effective_cutoff)
    ]
    discovered.sort(key=lambda path: (path.stat().st_mtime, path.name))
    names.extend(path.name for path in discovered)

    valid: list[Path] = []
    for name in names:
        path = _safe_image_path(output_dir, name)
        if path is not None and _is_recent(path, effective_cutoff):
            valid.append(path)

    if max_images > 0:
        valid = valid[-max_images:]
    _write_manifest_unlocked(
        output_dir,
        [path.name for path in valid],
        reset_at=reset_at,
    )
    return valid


def load_gallery(
    output_dir: Path,
    *,
    retention_days: int = 7,
    max_images: int = 200,
) -> list[Path]:
    """Load the current shared gallery, pruning expired entries atomically."""
    retention_days = normalize_retention_days(retention_days)
    with _LOCK:
        return _load_gallery_unlocked(
            output_dir,
            retention_days=retention_days,
            max_images=max_images,
        )


def append_gallery_images(
    output_dir: Path,
    new_images: Iterable[Path],
    *,
    retention_days: int = 7,
    max_images: int = 200,
) -> list[Path]:
    """Merge newly generated files into the shared manifest without lost updates."""
    retention_days = normalize_retention_days(retention_days)
    with _LOCK:
        current = _load_gallery_unlocked(
            output_dir,
            retention_days=retention_days,
            max_images=max_images,
        )
        by_name = {path.name: path for path in current}
        ordered_names = [path.name for path in current]
        manifest = _read_manifest_unlocked(output_dir)
        reset_at = manifest[1] if manifest is not None else 0.0

        for raw_path in new_images:
            path = Path(raw_path)
            safe_path = _safe_image_path(output_dir, path.name)
            if safe_path is None or not safe_path.exists():
                continue
            if safe_path.name in by_name:
                ordered_names.remove(safe_path.name)
            by_name[safe_path.name] = safe_path
            ordered_names.append(safe_path.name)

        if max_images > 0:
            ordered_names = ordered_names[-max_images:]
        result = [by_name[name] for name in ordered_names]
        _write_manifest_unlocked(output_dir, ordered_names, reset_at=reset_at)
        return result


def clear_gallery(output_dir: Path) -> None:
    """Start a new working gallery without deleting generation history."""
    with _LOCK:
        _write_manifest_unlocked(output_dir, [], reset_at=time.time())


def _read_batches_unlocked(output_dir: Path) -> list[dict]:
    path = _batches_path(output_dir)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if isinstance(data, dict):
        data = data.get("batches", [])
    return data if isinstance(data, list) else []


def _write_batches_unlocked(output_dir: Path, batches: list[dict]) -> None:
    output_dir.mkdir(exist_ok=True, parents=True)
    path = _batches_path(output_dir)
    temp_path = path.with_suffix(".json.tmp")
    temp_path.write_text(
        json.dumps(
            {"version": 2, "batches": batches},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    os.replace(temp_path, path)


def _normalize_batches_unlocked(
    output_dir: Path,
    raw_batches: Iterable[object],
    *,
    cutoff: float,
    max_batches: int,
    max_images: int,
) -> tuple[list[dict], list[dict]]:
    """Return JSON-safe batches and UI batches containing resolved Paths."""
    candidates: list[dict] = []
    seen_ids: set[str] = set()
    for raw in raw_batches:
        if not isinstance(raw, dict):
            continue
        batch_id = str(raw.get("id", "")).strip()
        prompt = raw.get("prompt", "")
        provider = str(raw.get("provider", "")).strip() or "unknown"
        model = raw.get("model", "")
        size = raw.get("size", "")
        quality = raw.get("quality", "")
        model = model.strip() if isinstance(model, str) else ""
        size = size.strip() if isinstance(size, str) else ""
        quality = quality.strip() if isinstance(quality, str) else ""
        if not batch_id or batch_id in seen_ids or not isinstance(prompt, str):
            continue
        try:
            created_at = float(raw.get("created_at", 0))
        except (TypeError, ValueError):
            continue
        if created_at < cutoff:
            continue

        paths: list[Path] = []
        seen_names: set[str] = set()
        for name in raw.get("images", []):
            path = _safe_image_path(output_dir, name)
            if (
                path is not None
                and path.name not in seen_names
                and _is_recent(path, cutoff)
            ):
                seen_names.add(path.name)
                paths.append(path)
        if not paths:
            continue
        seen_ids.add(batch_id)
        candidates.append(
            {
                "id": batch_id,
                "created_at": created_at,
                "prompt": prompt,
                "provider": provider,
                "model": model,
                "size": size,
                "quality": quality,
                "images": paths,
            }
        )

    candidates.sort(key=lambda batch: (batch["created_at"], batch["id"]), reverse=True)
    kept: list[dict] = []
    image_count = 0
    for batch in candidates:
        if max_batches > 0 and len(kept) >= max_batches:
            break
        remaining = max_images - image_count if max_images > 0 else len(batch["images"])
        if max_images > 0 and remaining <= 0:
            break
        paths = batch["images"][:remaining] if max_images > 0 else batch["images"]
        if not paths:
            continue
        kept.append({**batch, "images": paths})
        image_count += len(paths)

    serializable = [
        {
            "id": batch["id"],
            "created_at": batch["created_at"],
            "prompt": batch["prompt"],
            "provider": batch["provider"],
            "model": batch["model"],
            "size": batch["size"],
            "quality": batch["quality"],
            "images": [path.name for path in batch["images"]],
        }
        for batch in kept
    ]
    return serializable, kept


def load_generation_batches(
    output_dir: Path,
    *,
    retention_days: int = 7,
    max_batches: int = 100,
    max_images: int = 500,
) -> list[dict]:
    """Load recent generation runs, newest first, with prompt snapshots."""
    retention_days = normalize_retention_days(retention_days)
    cutoff = time.time() - retention_days * 24 * 60 * 60
    with _LOCK:
        output_dir.mkdir(exist_ok=True, parents=True)
        _prune_expired_unlocked(output_dir, cutoff)
        serializable, batches = _normalize_batches_unlocked(
            output_dir,
            _read_batches_unlocked(output_dir),
            cutoff=cutoff,
            max_batches=max_batches,
            max_images=max_images,
        )
        _write_batches_unlocked(output_dir, serializable)
        return batches


def save_generation_batch(
    output_dir: Path,
    *,
    batch_id: str,
    prompt: str,
    provider: str,
    model: str = "",
    size: str = "",
    quality: str = "",
    images: Iterable[Path],
    retention_days: int = 7,
    max_batches: int = 100,
    max_images: int = 500,
    created_at: float | None = None,
) -> list[dict]:
    """Persist one completed generation run independently of browser state."""
    retention_days = normalize_retention_days(retention_days)
    cutoff = time.time() - retention_days * 24 * 60 * 60
    image_names: list[str] = []
    seen_names: set[str] = set()
    for raw_path in images:
        path = _safe_image_path(output_dir, Path(raw_path).name)
        if path is not None and path.exists() and path.name not in seen_names:
            seen_names.add(path.name)
            image_names.append(path.name)
    if not image_names:
        return load_generation_batches(
            output_dir,
            retention_days=retention_days,
            max_batches=max_batches,
            max_images=max_images,
        )

    with _LOCK:
        current = [
            batch
            for batch in _read_batches_unlocked(output_dir)
            if isinstance(batch, dict) and str(batch.get("id", "")) != str(batch_id)
        ]
        current.append(
            {
                "id": str(batch_id),
                "created_at": float(created_at if created_at is not None else time.time()),
                "prompt": str(prompt),
                "provider": str(provider),
                "model": str(model),
                "size": str(size),
                "quality": str(quality),
                "images": image_names,
            }
        )
        serializable, batches = _normalize_batches_unlocked(
            output_dir,
            current,
            cutoff=cutoff,
            max_batches=max_batches,
            max_images=max_images,
        )
        _write_batches_unlocked(output_dir, serializable)
        return batches


def create_gallery_zip(
    output_dir: Path,
    images: Iterable[Path],
    *,
    archive_key: str,
) -> Path:
    """Create a replaceable ZIP staging file for an explicit user download."""
    safe_key = "".join(ch for ch in archive_key if ch.isalnum())[:24] or "session"
    archive_path = output_dir / f"gallery_export_{safe_key}.zip"
    temp_path = archive_path.with_suffix(".zip.tmp")

    with _LOCK:
        with zipfile.ZipFile(
            temp_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
        ) as archive:
            seen: set[str] = set()
            for raw_path in images:
                safe_path = _safe_image_path(output_dir, Path(raw_path).name)
                if (
                    safe_path is not None
                    and safe_path.exists()
                    and safe_path.name not in seen
                ):
                    seen.add(safe_path.name)
                    archive.write(safe_path, arcname=safe_path.name)
        os.replace(temp_path, archive_path)
    return archive_path
