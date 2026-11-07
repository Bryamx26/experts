"""Animations des experts : arbre de comportement (sans Qt) et lecture des bandes PNG."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experts.core.companions import ASSETS_DIR
from experts.ui.behavior import STATES, Behavior


def drag(b: Behavior, t: float, distance: float, duration: float) -> None:
    b.drag_start(t, 0, 0)
    steps = 10
    for k in range(1, steps + 1):
        b.drag_move(t + duration * k / steps, distance * k / steps, 0)


def test_idle_then_sleep_then_wake() -> None:
    b = Behavior(now=0)
    assert b.tick(1) == "idle"
    assert b.tick(299) == "idle"
    assert b.tick(301) == "dort"
    b.activity(302)  # première frappe
    assert b.tick(302.1) == "idle"


def test_drag_speed_picks_fear_or_surprise() -> None:
    b = Behavior()
    drag(b, 10, distance=100, duration=1.0)  # 100 px/s
    assert b.tick(11) == "surpris"
    b.drag_end(11, moved=True)
    drag(b, 20, distance=400, duration=0.1)  # 4000 px/s
    assert b.tick(20.1) == "peur"
    b.drag_end(20.1, moved=True)
    assert b.tick(20.2) == "idle"


def test_too_many_moves_makes_it_sulk_and_ignore_clicks() -> None:
    b = Behavior()
    for k in range(5):
        drag(b, 10 + k, distance=50, duration=0.5)
        b.drag_end(10.5 + k, moved=True)
    assert b.tick(15) == "fache"
    assert b.click(16) is False  # il boude : le clic n'ouvre pas la question
    assert b.tick(44) == "fache"
    assert b.tick(46) != "fache"  # 30 s plus tard, c'est oublié
    assert b.click(47) is True


def test_moves_spread_over_time_are_forgiven() -> None:
    b = Behavior()
    for k in range(5):
        drag(b, k * 200.0, distance=50, duration=0.5)
        b.drag_end(k * 200.0 + 0.5, moved=True)
        assert b.tick(k * 200.0 + 1) != "fache"


def test_click_shows_joy_then_back_to_idle() -> None:
    b = Behavior()
    assert b.click(5) is True
    assert b.tick(5.1) == "content"
    assert b.tick(7) == "idle"


def test_new_error_surprises_and_fix_pleases() -> None:
    b = Behavior()
    b.diagnostics(1, 0)  # premier état connu : pas de réaction
    assert b.tick(1.1) == "idle"
    b.diagnostics(2, 2)
    assert b.tick(2.1) == "surpris"
    assert b.tick(3) == "idle"
    b.diagnostics(4, 0)  # validateurs au vert
    assert b.tick(4.1) == "content"


def test_hunger_then_meal_restores_energy() -> None:
    b = Behavior()
    for _ in range(75):
        b.model_request()
    b.activity(100)
    assert b.tick(100) == "faim"  # énergie 25 %, mais on tape encore
    assert b.tick(111) == "mange"  # 10 s sans frappe : il mange la barrette de RAM
    assert b.bb.energy == 100
    assert b.tick(112) == "mange"
    assert b.tick(114) == "idle"


def test_every_expert_has_every_animation() -> None:
    manifest = json.loads((ASSETS_DIR / "sprites.json").read_text())
    size = manifest["frame_size"][0]
    for expert in ("python", "docker", "kubernetes"):
        states = manifest["experts"][expert]["etats"]
        assert set(states) == set(STATES)
        for info in states.values():
            path = ASSETS_DIR / info["fichier"]
            assert path.is_file()
            width, height = _png_size(path)
            assert (width, height) == (size * info["images"], size)


def _png_size(path: Path) -> tuple[int, int]:
    header = path.read_bytes()[:24]
    assert header[:8] == b"\x89PNG\r\n\x1a\n"
    return int.from_bytes(header[16:20], "big"), int.from_bytes(header[20:24], "big")


def test_frame_index() -> None:
    pytest.importorskip("PySide6.QtGui")
    from experts.ui.animation import Clip, frame_index

    loop = Clip(Path("x.png"), frames=8, fps=8, loop=True)
    once = Clip(Path("x.png"), frames=6, fps=12, loop=False)
    assert [frame_index(loop, t) for t in (0, 0.13, 1.0, 1.13)] == [0, 1, 0, 1]
    assert frame_index(once, 0.25) == 3 and frame_index(once, 10) == 5  # reste sur la dernière image
