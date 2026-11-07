"""Fenêtre du compagnon : petite, sans bordure, translucide, toujours au premier plan.

Choix :
- Qt.Tool + FramelessWindowHint + WindowStaysOnTopHint : pas d'entrée dans la
  barre des tâches, pas de bordure, au-dessus de l'éditeur ;
- le personnage est animé par des bandes PNG pré-rendues (une par état, voir
  animation.py) ; l'état d'animation vient de l'arbre de comportement
  (behavior.py), évalué 10 fois par seconde ;
- aucun cercle autour du personnage : l'humeur passe par l'animation, et
  l'erreur ou la suggestion par la bulle ;
- environ 64 px de personnage, rangé au bord droit de l'écran : il ne doit jamais
  masquer le code. La bulle s'ouvre à sa gauche et se referme seule ;
- déplaçable à la souris (attention : il n'aime pas trop ça).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from html import escape
from typing import Any

from PySide6.QtCore import QPoint, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QGuiApplication, QMouseEvent, QPainter, QPaintEvent
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from experts.core.companions import CompanionSpec
from experts.ui.animation import Sheets, load_clips
from experts.ui.behavior import Behavior

FRAME_PX = 128  # une image de 256 px affichée à 128 : le personnage y fait ~65 px
SPRITE_SIZE = 64  # repli SVG quand un expert n'a pas d'animations
CHAR_BOX = (0.2, 0.36, 0.6, 0.58)  # zone du personnage dans l'image (x, y, l, h)
DRAG_THRESHOLD_PX = 4
BEHAVIOR_MS = 100
LSP_STATES = frozenset({"idle", "think", "alert", "suggest"})
BUBBLE_MS = {"diagnostics": 9000, "suggestion": 15000, "info": 5000}


class SpriteWidget(QWidget):
    clicked = Signal()

    def __init__(self, parent: QWidget | None = None, clock: Callable[[], float] = time.monotonic) -> None:
        super().__init__(parent)
        self.clock = clock
        self.setFixedSize(QSize(FRAME_PX, FRAME_PX))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Clique pour poser une question")
        self._renderers: dict[str, QSvgRenderer] = {}
        self.sheets: Sheets | None = None
        self.state = "idle"  # état du serveur LSP
        self.anim = "idle"  # humeur → animation
        self._anim_since = clock()
        self.selected = False  # panneau de question ouvert
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.update)
        self._timer.start(42)  # ~24 images/s, au-dessus du fps le plus rapide des bandes (12)

    def set_companion(self, spec: CompanionSpec) -> None:
        clips = load_clips(spec.name, spec.sprites["idle"].parent.parent)
        self.sheets = Sheets(clips) if clips else None
        self._renderers = {state: QSvgRenderer(str(path), self) for state, path in spec.sprites.items()}
        self.setToolTip(f"{spec.display_name} ({spec.character})")
        self._anim_since = self.clock()
        self.update()

    def set_state(self, state: str) -> None:
        if state in LSP_STATES:
            self.state = state
            self.update()

    def set_anim(self, anim: str) -> None:
        if anim != self.anim:
            self.anim = anim
            self._anim_since = self.clock()
            self.update()

    def set_selected(self, selected: bool) -> None:
        self.selected = selected
        self.update()

    def _char_rect(self) -> QRectF:
        x, y, w, h = CHAR_BOX
        return QRectF(self.width() * x, self.height() * y, self.width() * w, self.height() * h)

    def paintEvent(self, event: QPaintEvent) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        now = self.clock()
        frame = self.sheets.frame(self.anim, now - self._anim_since) if self.sheets else None
        if frame is not None:
            p.drawPixmap(self.rect(), frame)
        else:
            renderer = self._renderers.get(self.state)
            if renderer is not None and renderer.isValid():
                keep = Qt.AspectRatioMode.KeepAspectRatio
                box = renderer.defaultSize().scaled(SPRITE_SIZE, SPRITE_SIZE, keep)
                target = self._char_rect()
                x = target.center().x() - box.width() / 2
                y = target.bottom() - box.height()
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

    def __init__(
        self,
        companions: dict[str, CompanionSpec],
        default: str = "python",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(None)
        self.companions = companions
        self.clock = clock
        self.behavior = Behavior(clock())
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowTitle("Experts")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.bubble = Bubble(self)
        self.sprite = SpriteWidget(self, clock)
        layout.addWidget(self.bubble, 0, Qt.AlignmentFlag.AlignBottom)
        layout.addWidget(self.sprite, 0, Qt.AlignmentFlag.AlignBottom)
        self.sprite.clicked.connect(self._on_click)
        self.current = companions[default]
        self.sprite.set_companion(self.current)
        self._anchor: QPoint | None = None  # coin bas-droit, fixe même quand la bulle s'ouvre
        self._drag_from: QPoint | None = None
        self._press_at: QPoint | None = None
        self._moved = False
        self._behavior_timer = QTimer(self)
        self._behavior_timer.timeout.connect(self.tick_behavior)
        self._behavior_timer.start(BEHAVIOR_MS)

    # --- comportement ---
    def tick_behavior(self) -> None:
        self.sprite.set_anim(self.behavior.tick(self.clock()))

    def _on_click(self) -> None:
        if self.behavior.click(self.clock()):
            self.tick_behavior()
            self.sprite_clicked.emit()
        else:
            self.say("<b>Grrr !</b> Je boude. Laisse-moi tranquille un moment.", BUBBLE_MS["info"])

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
            state = str(event.get("state"))
            if state == "think" and self.sprite.state != "think":
                self.behavior.model_request()
            self.sprite.set_state(state)
        elif kind == "diagnostics":
            items = event.get("items") or []
            self.behavior.diagnostics(self.clock(), sum(1 for i in items if i.get("severity") == "error"))
            self._show_diagnostics(items)
        elif kind == "activity":
            self.behavior.activity(self.clock())
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
            self._press_at = event.globalPosition().toPoint()
            self._drag_from = self._press_at - self.frameGeometry().topLeft()
            self._moved = False

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_from is None or self._press_at is None:
            return
        pos = event.globalPosition().toPoint()
        if not self._moved:
            if (pos - self._press_at).manhattanLength() < DRAG_THRESHOLD_PX:
                return  # tremblement de la main : c'est encore un clic
            self._moved = True
            self.behavior.drag_start(self.clock(), self._press_at.x(), self._press_at.y())
            self.sprite.setProperty("dragged", True)
        self.move(pos - self._drag_from)
        self._anchor = self.frameGeometry().bottomRight()
        self.behavior.drag_move(self.clock(), pos.x(), pos.y())

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._moved:
            self.behavior.drag_end(self.clock(), moved=True)
            self.tick_behavior()
        self._drag_from = self._press_at = None
        self._moved = False
