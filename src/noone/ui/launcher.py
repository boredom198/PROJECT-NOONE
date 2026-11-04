"""PyWebView desktop launcher for PROJECT NOONE (Component 6)."""
from __future__ import annotations

import logging
import sys

from noone.ui.paths import log_dir
from noone.ui.server import AppContext, serve


def _configure_logging() -> None:
    try:
        handler: logging.Handler = logging.FileHandler(
            log_dir() / "noone-ui.log", encoding="utf-8"
        )
    except Exception:
        handler = logging.StreamHandler()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[handler],
    )


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    _configure_logging()
    log = logging.getLogger("noone.ui")

    is_dev = "--dev" in args or "--backend-only" in args
    target_port = 8765 if is_dev else 0
    for i, arg in enumerate(args):
        if arg == "--port" and i + 1 < len(args):
            try:
                target_port = int(args[i + 1])
            except ValueError:
                pass

    ctx = AppContext(dev_mode=is_dev)
    try:
        server, port = serve(ctx, port=target_port)
    except OSError as exc:
        msg = f"Failed to bind NOONE backend to 127.0.0.1:{target_port} - port is already in use or unavailable: {exc}"
        log.error(msg)
        print(f"ERROR: {msg}", file=sys.stderr)
        return 1

    url = f"http://127.0.0.1:{port}/"
    if "--dev" in args:
        url = "http://127.0.0.1:5173/"
        log.info("Dev mode: PyWebView loading Vite dev server at %s (Backend at 127.0.0.1:%s)", url, port)
    if "--backend-only" in args:
        print(f"NOONE backend running at: http://127.0.0.1:{port}/ (dev_mode={is_dev})")
        print(f"API endpoints available at http://127.0.0.1:{port}/api/...")
        try:
            import time

            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass
        finally:
            ctx.shutdown()
            server.shutdown()
        return 0

    import webview

    window = webview.create_window(
        "PROJECT NOONE",
        url,
        width=1440,
        height=900,
        min_size=(1080, 680),
        background_color="#080b11",
    )

    def _on_closed() -> None:
        log.info("Window closed; shutting down backend")
        try:
            ctx.shutdown()
        finally:
            server.shutdown()

    window.events.closed += _on_closed
    webview.start(debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
