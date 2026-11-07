"""Fenêtre du compagnon : python -m experts.ui (ou experts-ui)."""

from __future__ import annotations

import logging
import sys
import threading

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import QApplication, QMenu, QSystemTrayIcon

from experts.core.bridge import create_token
from experts.core.companions import load_companions
from experts.core.config import Settings
from experts.ui.bridge_server import BridgeServer
from experts.ui.controller import Controller
from experts.ui.window import CompanionWindow

HOTKEY = "<ctrl>+<alt>+e"
log = logging.getLogger("experts.ui")


class HotkeyRelay(QObject):
    """pynput tourne dans son propre thread : on repasse dans le thread Qt par un signal."""

    triggered = Signal()


def start_hotkey(relay: HotkeyRelay) -> None:
    try:
        from pynput import keyboard
    except Exception as exc:  # pas de serveur X, Wayland strict, macOS sans permission…
        log.warning("Raccourci global indisponible : %s", exc)
        return

    def run() -> None:
        try:
            with keyboard.GlobalHotKeys({HOTKEY: relay.triggered.emit}) as hk:
                hk.join()
        except Exception as exc:
            log.warning("Raccourci global indisponible : %s", exc)

    threading.Thread(target=run, daemon=True, name="hotkey").start()


def build_tray(app: QApplication, window: CompanionWindow) -> QSystemTrayIcon:
    tray = QSystemTrayIcon(QIcon(str(window.current.sprites["idle"])), app)
    menu = QMenu()
    toggle = QAction("Afficher / masquer (Ctrl+Alt+E)", menu)
    toggle.triggered.connect(lambda: window.setVisible(not window.isVisible()))
    quit_action = QAction("Quitter", menu)
    quit_action.triggered.connect(app.quit)
    menu.addAction(toggle)
    menu.addAction(quit_action)
    tray.setContextMenu(menu)
    tray.setToolTip("Experts")
    tray.show()
    return tray


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    settings = Settings.from_env()
    window = CompanionWindow(load_companions())
    try:
        bridge = BridgeServer(create_token(settings.data_dir), port=settings.ui_port)
    except OSError as exc:
        log.error("%s", exc)
        sys.exit(1)
    controller = Controller(window, bridge)
    bridge.event_received.connect(controller.handle_event)
    window.place_default()
    window.show()
    tray = build_tray(app, window)
    relay = HotkeyRelay()
    relay.triggered.connect(lambda: window.setVisible(not window.isVisible()))
    start_hotkey(relay)
    code = app.exec()
    bridge.close()
    tray.hide()
    sys.exit(code)


if __name__ == "__main__":
    main()
