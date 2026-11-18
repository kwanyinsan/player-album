"""Resolve staged artifacts with an optional read-only collection fallback."""
from pathlib import Path, PureWindowsPath


def artifact_path(root: Path, relative: str, source_root: Path | None = None) -> Path:
    path = Path(relative.replace("\\", "/"))
    if path.is_absolute() or PureWindowsPath(relative).drive or ".." in path.parts:
        raise ValueError(f"Artifact must be a relative path within the run: {relative}")
    for base in (root, source_root):
        if base is None:
            continue
        target = (base / path).resolve()
        if not target.is_relative_to(base.resolve()):
            raise ValueError(f"Missing or external artifact: {relative}")
        if target.is_file():
            return target
    raise ValueError(f"Missing or external artifact: {relative}")
