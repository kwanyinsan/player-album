from __future__ import annotations

from pathlib import Path
from typing import Iterable

from config import DetectorConfig


class YOLOEPlayerDetector:
    """Small wrapper around Ultralytics so the backend can be swapped later."""

    def __init__(self, config: DetectorConfig) -> None:
        self.config = config
        self.model = None
        self.prompt_warning: str | None = None
        self.force_full_precision = False
        self.runtime_warnings: list[str] = []

    def load(self) -> None:
        if self.model is not None:
            return
        try:
            from ultralytics import YOLOE

            model_class = YOLOE
        except ImportError:
            try:
                from ultralytics import YOLO

                model_class = YOLO
            except ImportError as exc:
                raise RuntimeError(
                    "Ultralytics is required for YOLOE detection/tracking. "
                    "Install it in the environment instead of copying it into this project."
                ) from exc

        self.model = model_class(str(self.config.yoloe_model))
        self._set_prompt_if_supported()

    def _set_prompt_if_supported(self) -> None:
        if self.model is None:
            return
        set_classes = getattr(self.model, "set_classes", None)
        if set_classes is None:
            self.prompt_warning = "Loaded model does not expose set_classes(); prompt was not applied."
            return
        try:
            set_classes([self.config.prompt])
        except TypeError:
            # Some Ultralytics builds have changed YOLOE prompt signatures over time.
            try:
                set_classes(classes=[self.config.prompt])
            except Exception as exc:  # pragma: no cover - depends on installed Ultralytics version.
                self.prompt_warning = f"Could not apply YOLOE prompt {self.config.prompt!r}: {exc}"
        except Exception as exc:  # pragma: no cover - depends on installed Ultralytics version.
            self.prompt_warning = f"Could not apply YOLOE prompt {self.config.prompt!r}: {exc}"

    def reset_model(self) -> None:
        self.model = None

    def disable_half_precision(self, reason: str) -> None:
        if not self.force_full_precision:
            self.runtime_warnings.append(f"Disabled half precision for YOLOE tracking: {reason}")
        self.force_full_precision = True
        self.reset_model()

    def track_video(self, video_path: Path) -> Iterable[object]:
        self.load()
        assert self.model is not None
        half = self.config.half and self.config.device.lower() != "cpu" and not self.force_full_precision
        return self.model.track(
            source=str(video_path),
            stream=True,
            tracker=self.config.tracker,
            imgsz=self.config.imgsz,
            conf=self.config.conf,
            iou=self.config.iou,
            device=self.config.device,
            half=half,
            persist=False,
            verbose=False,
            save=False,
            show=False,
        )
