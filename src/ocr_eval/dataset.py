from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


@dataclass
class Sample:
    id: str  # path relative to data_dir, without extension (e.g. "story_en_p1")
    image_path: Path
    text_path: Path

    @property
    def mime(self) -> str:
        return MIME_TYPES[self.image_path.suffix.lower()]

    @cached_property
    def reference(self) -> str:
        return self.text_path.read_text(encoding="utf-8-sig")

    @cached_property
    def image_sha256(self) -> str:
        return hashlib.sha256(self.image_path.read_bytes()).hexdigest()


def load_dataset(data_dir: Path) -> tuple[list[Sample], list[Path]]:
    """Pair every image with the .txt file of the same name. Returns (samples, images without text)."""
    samples: list[Sample] = []
    missing: list[Path] = []
    for image in sorted(data_dir.rglob("*")):
        if not image.is_file() or image.suffix.lower() not in MIME_TYPES:
            continue
        text = image.with_suffix(".txt")
        if not text.exists():
            missing.append(image)
            continue
        sample_id = image.relative_to(data_dir).with_suffix("").as_posix()
        samples.append(Sample(sample_id, image, text))
    return samples, missing
