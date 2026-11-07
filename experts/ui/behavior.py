"""Arbre de comportement de l'expert (sélecteur à priorité, sans Qt).

Évalué environ 10 fois par seconde ; la première branche vraie gagne :

1. déplacé à la souris : rapide → peur, sinon surpris ;
2. trop déplacé (≥ 5 fois en 2 min, ou colère > 60) → fâché, boude 30 s et ignore les clics ;
3. réaction ponctuelle en cours (clic, nouvelle erreur, correction réussie, repas) ;
4. faim : énergie < 30 % → faim, puis mange une barrette de RAM après 10 s sans frappe ;
5. inactif depuis 5 min → dort ;
6. sinon → idle.

Choix : l'arbre est plat, donc une liste ordonnée de conditions écrite à la main
suffit (py_trees serait une dépendance de trop). Le temps est passé en argument :
tout se teste sans horloge ni écran.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

FAST_DRAG_PX_S = 1500.0
MOVES_WINDOW_S = 120.0
MOVES_LIMIT = 5
ANGER_LIMIT = 60.0
SULK_S = 30.0
HUNGRY_BELOW = 30.0
MEAL_AFTER_IDLE_S = 10.0
SLEEP_AFTER_S = 300.0
SPEED_WINDOW_S = 0.1

# Durée des réactions ponctuelles (= durée de leur animation).
REACTION_S = {"content": 1.5, "surpris": 0.6, "mange": 2.0}

STATES = ("idle", "content", "faim", "mange", "fache", "surpris", "peur", "dort")


@dataclass
class Blackboard:
    """Mémoire de l'expert, partagée par les branches de l'arbre."""

    anger: float = 0.0  # 0-100 : +15 par déplacement, +25 si brusque, -1 par seconde
    energy: float = 100.0  # 0-100 % : -1 par requête au modèle, -1 par minute
    moves: deque[float] = field(default_factory=deque)  # fins de drag, fenêtre de 120 s
    dragging: bool = False
    drag_speed: float = 0.0  # px/s, lissée sur 100 ms
    drag_peak: float = 0.0
    last_activity: float = 0.0  # dernière frappe (didChange) ou interaction
    errors: int | None = None  # nombre d'erreurs au dernier diagnostic
    sulk_until: float = 0.0
    reaction: str | None = None
    reaction_until: float = 0.0
    last_tick: float | None = None


class Behavior:
    def __init__(self, now: float = 0.0) -> None:
        self.bb = Blackboard(last_activity=now)
        self._samples: deque[tuple[float, float, float]] = deque()  # (t, x, y)
        self.state = "idle"

    # --- entrées ---
    def drag_start(self, t: float, x: float, y: float) -> None:
        self.bb.dragging = True
        self.bb.drag_speed = self.bb.drag_peak = 0.0
        self._samples.clear()
        self._samples.append((t, x, y))
        self.bb.last_activity = t

    def drag_move(self, t: float, x: float, y: float) -> None:
        if not self.bb.dragging:
            return
        self._samples.append((t, x, y))
        while len(self._samples) > 2 and t - self._samples[0][0] > SPEED_WINDOW_S:
            self._samples.popleft()
        t0, x0, y0 = self._samples[0]
        dt = t - t0
        if dt > 0:
            self.bb.drag_speed = ((x - x0) ** 2 + (y - y0) ** 2) ** 0.5 / dt
            self.bb.drag_peak = max(self.bb.drag_peak, self.bb.drag_speed)

    def drag_end(self, t: float, moved: bool) -> None:
        """`moved` : faux si l'expert n'a pas bougé (c'était un simple clic)."""
        bb = self.bb
        bb.dragging = False
        bb.drag_speed = 0.0
        if not moved:
            return
        bb.moves.append(t)
        bb.anger = min(100.0, bb.anger + (25.0 if bb.drag_peak > FAST_DRAG_PX_S else 15.0))
        self._prune_moves(t)
        if len(bb.moves) >= MOVES_LIMIT or bb.anger > ANGER_LIMIT:
            bb.sulk_until = t + SULK_S
            bb.moves.clear()

    def click(self, t: float) -> bool:
        """Vrai si l'expert accepte le clic (il l'ignore quand il boude)."""
        self.bb.last_activity = t
        if self.sulking(t):
            return False
        self._react("content", t)
        return True

    def activity(self, t: float) -> None:
        """Frappe dans l'éditeur : réveille l'expert et repousse le repas."""
        self.bb.last_activity = t

    def model_request(self) -> None:
        self.bb.energy = max(0.0, self.bb.energy - 1.0)

    def diagnostics(self, t: float, errors: int) -> None:
        before = self.bb.errors
        self.bb.errors = errors
        if before is None:
            return
        if errors > before:
            self._react("surpris", t)  # nouvelle erreur
        elif before > 0 and errors == 0:
            self._react("content", t)  # tout est corrigé, validateurs au vert

    def fix_applied(self, t: float) -> None:
        self._react("content", t)

    # --- évaluation ---
    def sulking(self, t: float) -> bool:
        return t < self.bb.sulk_until

    def tick(self, t: float) -> str:
        bb = self.bb
        if bb.last_tick is not None:
            dt = max(0.0, t - bb.last_tick)
            bb.anger = max(0.0, bb.anger - dt)
            bb.energy = max(0.0, bb.energy - dt / 60.0)
        bb.last_tick = t
        self._prune_moves(t)
        self.state = self._select(t)
        return self.state

    def _select(self, t: float) -> str:
        bb = self.bb
        if bb.dragging:
            return "peur" if bb.drag_speed > FAST_DRAG_PX_S else "surpris"
        if self.sulking(t):
            return "fache"
        if bb.anger > ANGER_LIMIT:
            bb.sulk_until = t + SULK_S
            return "fache"
        if bb.reaction is not None and t < bb.reaction_until:
            return bb.reaction
        bb.reaction = None
        if bb.energy < HUNGRY_BELOW:
            if t - bb.last_activity >= MEAL_AFTER_IDLE_S:
                bb.energy = 100.0
                self._react("mange", t)
                return "mange"
            return "faim"
        if t - bb.last_activity >= SLEEP_AFTER_S:
            return "dort"
        return "idle"

    def _react(self, state: str, t: float) -> None:
        self.bb.reaction = state
        self.bb.reaction_until = t + REACTION_S[state]

    def _prune_moves(self, t: float) -> None:
        while self.bb.moves and t - self.bb.moves[0] > MOVES_WINDOW_S:
            self.bb.moves.popleft()
