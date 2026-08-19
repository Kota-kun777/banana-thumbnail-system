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
    """Start a new gallery while leaving files available until retention cleanup."""
    with _LOCK:
        _write_manifest_unlocked(output_dir, [], reset_at=time.time())


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
