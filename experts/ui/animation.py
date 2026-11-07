"""Lecture des animations pré-rendues (bandes PNG décrites par assets/sprites.json).

Chaque état est une bande horizontale d'images de 256 x 256 px : l'image i est à
x = i * 256. Les mouvements (saut, tremblement, étirement) et les onomatopées
sont déjà dans les images, donc l'affichage se résume à choisir la bonne image
au bon moment : pas de rendu SVG à chaque image, très peu de CPU.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtGui import QPixmap

from experts.core.companions import ASSETS_DIR

MANIFEST = "sprites.json"


@dataclass(frozen=True, slots=True)
class Clip:
    path: Path
    frames: int
    fps: float
    loop: bool

    @property
    def duration(self) -> float:
        return self.frames / self.fps


def frame_index(clip: Clip, elapsed: float) -> int:
    """Image à montrer `elapsed` secondes après le début de l'état."""
    i = int(max(elapsed, 0.0) * clip.fps)
    return i % clip.frames if clip.loop else min(i, clip.frames - 1)


def load_clips(expert: str, assets_dir: Path = ASSETS_DIR) -> dict[str, Clip]:
    """Clips de l'expert, ou {} s'il n'a pas d'animations (repli sur le SVG fixe)."""
    manifest = assets_dir / MANIFEST
    if not manifest.is_file():
        return {}
    data = json.loads(manifest.read_text(encoding="utf-8"))
    states = data.get("experts", {}).get(expert, {}).get("etats", {})
    clips: dict[str, Clip] = {}
    for state, info in states.items():
        path = assets_dir / str(info["fichier"])
        if path.is_file() and int(info["images"]) > 0 and float(info["fps"]) > 0:
            clips[state] = Clip(path, int(info["images"]), float(info["fps"]), bool(info["boucle"]))
    return clips


class Sheets:
    """Bandes chargées une fois par expert et découpées à la demande."""

    def __init__(self, clips: dict[str, Clip], frame_size: int = 256) -> None:
        self.clips = clips
        self.frame_size = frame_size
        self._strips = {state: QPixmap(str(clip.path)) for state, clip in clips.items()}

    def frame(self, state: str, elapsed: float) -> QPixmap | None:
        clip = self.clips.get(state) or self.clips.get("idle")
        if clip is None:
            return None
        strip = self._strips[state if state in self.clips else "idle"]
        if strip.isNull():
            return None
        s = self.frame_size
        return strip.copy(frame_index(clip, elapsed) * s, 0, s, s)
