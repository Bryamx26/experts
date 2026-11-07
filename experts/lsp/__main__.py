"""Lance le serveur LSP sur stdin/stdout : python -m experts.lsp"""

from __future__ import annotations

import logging

from experts.lsp.server import server


def main() -> None:
    # Les logs vont sur stderr : stdout est réservé au protocole LSP.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    server.start_io()


if __name__ == "__main__":
    main()
