"""Fenêtre du compagnon : petite, sans bordure, translucide, toujours au premier plan.

Choix :
- Qt.Tool + FramelessWindowHint + WindowStaysOnTopHint : pas d'entrée dans la
  barre des tâches, pas de bordure, au-dessus de l'éditeur ;
- le sprite est le SVG de l'expert (net à toutes les tailles) ; l'état est montré
  par une animation et un halo coloré, sans dessiner quatre images par expert ;
- environ 64 px, rangé au bord droit de l'écran : il ne doit jamais masquer le
  code. La bulle s'ouvre à sa gauche et se referme seule ;
- déplaçable à la souris.
"""

from __future__ import annotations

import math
from html import escape
from typing import Any

from PySide6.QtCore import QPoint, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QGuiApplication, QMouseEvent, QPainter, QPaintEvent, QPen
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from experts.core.companions import CompanionSpec

SPRITE_SIZE = 64
HALO = {
    "idle": None,
    "think": QColor(90, 160, 255),
    "alert": QColor(240, 80, 80),
    "suggest": QColor(250, 200, 60),
}
BUBBLE_MS = {"diagnostics": 9000, "suggestion": 15000, "info": 5000}


class SpriteWidget(QWidget):
    clicked = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedSize(QSize(SPRITE_SIZE + 16, SPRITE_SIZE + 16))
        self._renderers: dict[str, QSvgRenderer] = {}
        self.state = "idle"
        self._phase = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(33)  # ~30 images/s ; très peu de CPU pour un widget de 80 px

    def set_companion(self, spec: CompanionSpec) -> None:
        self._renderers = {state: QSvgRenderer(str(path), self) for state, path in spec.sprites.items()}
        self.setToolTip(f"{spec.display_name} ({spec.character})")
        self.update()

    def set_state(self, state: str) -> None:
        if state in HALO:
            self.state = state
            self.update()

    def _tick(self) -> None:
        self._phase = (self._phase + 0.033) % 1000
        self.update()

    def offset(self) -> QPoint:
        t = self._phase
        if self.state == "alert":
            return QPoint(round(2 * math.sin(t * 40)) if (t % 2) < 0.4 else 0, 0)  # petite secousse
        if self.state == "think":
            return QPoint(0, round(2 * math.sin(t * 6)))
        return QPoint(0, round(3 * math.sin(t * 2)))  # respiration

    def paintEvent(self, event: QPaintEvent) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        halo = HALO.get(self.state)
        if halo is not None:
            color = QColor(halo)
            color.setAlphaF(0.55 + 0.35 * math.sin(self._phase * 5))
            pen = QPen(color, 3, Qt.PenStyle.DashLine if self.state == "think" else Qt.PenStyle.SolidLine)
            p.setPen(pen)
            p.drawEllipse(QRectF(3, 3, self.width() - 6, self.height() - 6))
        renderer = self._renderers.get(self.state)
        if renderer is not None and renderer.isValid():
            box = renderer.defaultSize().scaled(SPRITE_SIZE, SPRITE_SIZE, Qt.AspectRatioMode.KeepAspectRatio)
            o = self.offset()
            x = (self.width() - box.width()) / 2 + o.x()
            y = (self.height() - box.height()) / 2 + o.y()
            renderer.render(p, QRectF(x, y, box.width(), box.height()))
        p.end()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and not self.property("dragged"):
            self.clicked.emit()
        self.setProperty("dragged", False)
        super().mouseReleaseEvent(event)


class Bubble(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("bubble")
        self.setStyleSheet(
            "#bubble { background: rgba(30, 32, 38, 235); border: 1px solid rgba(255,255,255,40);"
            " border-radius: 12px; } QLabel { color: #e8e8e8; font-size: 12px; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        self.label = QLabel(self)
        self.label.setWordWrap(True)
        self.label.setTextFormat(Qt.TextFormat.RichText)
        self.label.setMaximumWidth(320)
        layout.addWidget(self.label)
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self.hide)
        self.hide()

    def show_text(self, html: str, ms: int) -> None:
        self.label.setText(html)
        self.adjustSize()
        self.show()
        self._hide_timer.start(ms)


class CompanionWindow(QWidget):
    """Reçoit les évènements du LSP et les traduit en animation et en bulle."""

    sprite_clicked = Signal()

    def __init__(self, companions: dict[str, CompanionSpec], default: str = "python") -> None:
        super().__init__(None)
        self.companions = companions
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowTitle("Experts")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.bubble = Bubble(self)
        self.sprite = SpriteWidget(self)
        layout.addWidget(self.bubble, 0, Qt.AlignmentFlag.AlignBottom)
        layout.addWidget(self.sprite, 0, Qt.AlignmentFlag.AlignBottom)
        self.sprite.clicked.connect(self.sprite_clicked)
        self.current = companions[default]
        self.sprite.set_companion(self.current)
        self._anchor: QPoint | None = None  # coin bas-droit, fixe même quand la bulle s'ouvre
        self._drag_from: QPoint | None = None

    # --- placement ---
    def place_default(self) -> None:
        screen = QGuiApplication.primaryScreen().availableGeometry()
        self._anchor = QPoint(screen.right() - 16, screen.bottom() - 48)
        self.reposition()

    def reposition(self) -> None:
        self.adjustSize()
        if self._anchor is not None:
            self.move(self._anchor.x() - self.width(), self._anchor.y() - self.height())

    # --- évènements du LSP ---
    def handle_event(self, event: dict[str, Any]) -> None:
        kind = event.get("type")
        if kind == "focus":
            spec = self.companions.get(str(event.get("companion")))
            if spec is not None and spec is not self.current:
                self.current = spec
                self.sprite.set_companion(spec)
                self.say(f"<b>{escape(spec.display_name)}</b> prend le relais.", BUBBLE_MS["info"])
        elif kind == "state":
            self.sprite.set_state(str(event.get("state")))
        elif kind == "diagnostics":
            self._show_diagnostics(event.get("items") or [])
        elif kind == "suggestion":
            self.say(f"💡 {escape(str(event.get('text', '')))}", BUBBLE_MS["suggestion"])

    def _show_diagnostics(self, items: list[dict[str, Any]]) -> None:
        errors = [i for i in items if i.get("severity") == "error"]
        shown = errors or [i for i in items if i.get("severity") == "warning"]
        if not shown:
            self.bubble.hide()
            self.reposition()
            return
        first = shown[0]
        more = f" <i>(+{len(shown) - 1})</i>" if len(shown) > 1 else ""
        self.say(
            f"<b>Ligne {int(first.get('line', 1))}</b> · {escape(str(first.get('code', '')))}<br>"
            f"{escape(str(first.get('message', '')))}{more}",
            BUBBLE_MS["diagnostics"],
        )

    def say(self, html: str, ms: int) -> None:
        self.bubble.show_text(html, ms)
        self.reposition()

    # --- déplacement à la souris ---
    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_from = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_from is not None:
            self.move(event.globalPosition().toPoint() - self._drag_from)
            self._anchor = self.frameGeometry().bottomRight()
            self.sprite.setProperty("dragged", True)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._drag_from = None
