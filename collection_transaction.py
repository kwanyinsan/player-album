"""Single-writer, recoverable publication of managed collection outputs."""
from __future__ import annotations

import json
import os
import shutil
import time
from contextlib import contextmanager
from pathlib import Path


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


@contextmanager
def collection_lock(root: Path):
    """OS locks are released even if the process is killed; the file may remain."""
    with (root / ".collection.lock").open("a+b") as handle:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            # Windows byte-range locks may extend past end of file.
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("Another collection operation is running") from exc
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def recover(root: Path) -> bool:
    journal = root / ".collection-update.json"
    if not journal.exists():
        return False
    payload = json.loads(journal.read_text(encoding="utf-8"))
    operation = root / ".updates" / payload["operation"]
    if operation.resolve().parent != (root / ".updates").resolve():
        raise ValueError("Invalid update journal operation")
    if payload["status"] == "committed":
        journal.unlink()
        return True
    for entry in reversed(payload["entries"]):
        relative = Path(entry["path"])
        if relative.as_posix() not in {"global", "players", "review", "rejected", "logs.json", "project.db"}:
            if len(relative.parts) != 2 or relative.parts[0] != "videos" or not relative.name.startswith("V_"):
                raise ValueError("Invalid managed publication path")
        live, staged, backup = root / relative, operation / "stage" / relative, operation / "backup" / relative
        for path in (live, staged, backup):
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("Publication path escapes collection")
        if entry["install"] and live.exists() and not staged.exists():
            staged.parent.mkdir(parents=True, exist_ok=True)
            live.rename(staged)
        if backup.exists():
            live.parent.mkdir(parents=True, exist_ok=True)
            backup.rename(live)
    journal.unlink()
    return True


def prune_backups(root: Path, keep: int = 1) -> None:
    """Remove only old committed global-export backups, never video analysis."""
    if keep < 1 or (root / ".collection-update.json").exists():
        raise ValueError("Cannot prune active updates or discard every recovery backup")
    updates = root / ".updates"
    if not updates.exists():
        return
    committed = []
    for marker in updates.glob("*/committed.json"):
        if not marker.resolve().is_relative_to(root.resolve()):
            raise ValueError("Backup marker escapes collection")
        payload = json.loads(marker.read_text(encoding="utf-8"))
        committed.append((payload["committed_at"], marker.parent))
    for _, operation in sorted(committed, reverse=True)[keep:]:
        if operation.resolve().parent != updates.resolve() or (operation / "backup").is_symlink():
            raise ValueError("Backup workspace is not an owned operation directory")
        for name in ("global", "players", "review", "rejected", "logs.json", "project.db"):
            target = operation / "backup" / name
            if not target.resolve().is_relative_to(updates.resolve()) or target.is_symlink():
                raise ValueError("Backup path escapes managed update workspace")
            if target.is_dir():
                protected = {"models", "videos", "highlights", "input", "media"}
                if any(p.is_dir() and p.name.lower() in protected for p in target.rglob("*")):
                    continue
                shutil.rmtree(target)
            elif target.is_file():
                target.unlink()


def publish(root: Path, operation: Path, new_video_ids: list[str]) -> None:
    paths = [f"videos/{v}" for v in new_video_ids]
    paths += ["global", "players", "review", "rejected", "logs.json", "project.db"]
    stage = operation / "stage"
    for relative in paths:
        if not (root / relative).resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Managed output escapes collection: {relative}")
    payload = {"operation": operation.name, "status": "publishing", "entries": [
        {"path": p, "install": (stage / p).exists()} for p in paths
    ]}
    for video_id in new_video_ids:
        if (root / "videos" / video_id).exists():
            raise ValueError(f"Video output already exists: {video_id}")
    journal = root / ".collection-update.json"
    atomic_json(journal, payload)
    try:
        for entry in payload["entries"]:
            live = root / entry["path"]
            backup = operation / "backup" / entry["path"]
            if live.exists():
                backup.parent.mkdir(parents=True, exist_ok=True)
                live.rename(backup)
            if entry["install"]:
                live.parent.mkdir(parents=True, exist_ok=True)
                (stage / entry["path"]).rename(live)
        payload["status"] = "committed"
        atomic_json(journal, payload)
    except BaseException:
        recover(root)
        raise
    journal.unlink()
    atomic_json(operation / "committed.json", {"committed_at": time.time(), "operation": operation.name})
