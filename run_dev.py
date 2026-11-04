#!/usr/bin/env python3
"""PROJECT NOONE — Unified Development Runner.

Starts both the Python backend (port 8765) and the Vite frontend (port 5173)
concurrently, displays interleaved prefixed output, monitors process health,
and guarantees clean termination of all child process trees on shutdown.

Usage:
    python run_dev.py
    .venv\\Scripts\\python.exe run_dev.py
"""
from __future__ import annotations

import atexit
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path


def _get_port_pid(port: int) -> int | None:
    """Find the process ID listening on a given port on Windows."""
    try:
        out = subprocess.check_output("netstat -ano", shell=True, text=True)
        for line in out.splitlines():
            if f":{port}" in line and "LISTENING" in line:
                parts = line.strip().split()
                return int(parts[-1])
    except Exception:
        pass
    return None


def _kill_process_tree(pid: int) -> None:
    """Forcefully terminate a process and all its children on Windows."""
    if not pid:
        return
    if sys.platform == "win32":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except Exception:
            pass
    else:
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            pass


def _pipe_reader(stream, prefix: str, stop_event: threading.Event) -> None:
    """Stream child process output with a clear source prefix."""
    try:
        for line in iter(stream.readline, ""):
            if not line:
                break
            stripped = line.rstrip("\r\n")
            if stripped:
                print(f"[{prefix}] {stripped}", flush=True)
            if stop_event.is_set():
                break
    except Exception:
        pass
    finally:
        try:
            stream.close()
        except Exception:
            pass


def _wait_for_url(url: str, timeout: float = 6.0, headers: dict[str, str] | None = None) -> bool:
    """Poll a URL until it responds or timeout expires."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                if resp.status in (200, 403, 404):
                    return True
        except urllib.error.HTTPError as e:
            if e.code in (200, 403, 404):
                return True
        except Exception:
            pass
        time.sleep(0.3)
    return False


def main() -> int:
    root_dir = Path(__file__).resolve().parent
    frontend_dir = root_dir / "frontend"

    # 1. Resolve Python executable
    venv_python = root_dir / ".venv" / "Scripts" / "python.exe"
    if venv_python.is_file():
        python_exe = str(venv_python)
    else:
        python_exe = sys.executable

    # 2. Resolve npm executable
    npm_cmd = shutil.which("npm.cmd") or shutil.which("npm") or "npm"

    # Environment with unbuffered Python output and dev flag
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["NOONE_DEV"] = "1"

    print("=" * 66, flush=True)
    print("  PROJECT NOONE — UNIFIED DEVELOPMENT RUNNER", flush=True)
    print("=" * 66, flush=True)
    print(f"Root:     {root_dir}", flush=True)
    print(f"Python:   {python_exe}", flush=True)
    print(f"Frontend: {frontend_dir}", flush=True)

    # 3. Release any stale development processes from prior runs
    for port in (8765, 5173):
        stale_pid = _get_port_pid(port)
        if stale_pid:
            print(f"[NOONE DEV] Releasing stale process on port {port} (PID {stale_pid})...", flush=True)
            _kill_process_tree(stale_pid)
            time.sleep(0.5)

    print("=" * 66, flush=True)

    stop_event = threading.Event()
    cleaned_up = False
    backend_proc: subprocess.Popen | None = None
    frontend_proc: subprocess.Popen | None = None

    def _cleanup() -> None:
        nonlocal cleaned_up
        if cleaned_up:
            return
        cleaned_up = True
        stop_event.set()
        print("\n[NOONE DEV] Shutting down development services...", flush=True)

        # 1. Kill backend process tree
        if backend_proc and backend_proc.pid:
            _kill_process_tree(backend_proc.pid)
            print("[NOONE DEV] Backend stopped.", flush=True)

        # 2. Kill frontend process tree
        if frontend_proc and frontend_proc.pid:
            _kill_process_tree(frontend_proc.pid)
            print("[NOONE DEV] Frontend stopped.", flush=True)

        # 3. Double-check ports 8765 and 5173 to ensure no orphan survived
        for port in (8765, 5173):
            remaining = _get_port_pid(port)
            if remaining:
                _kill_process_tree(remaining)

        print("[NOONE DEV] All services terminated cleanly.", flush=True)

    atexit.register(_cleanup)

    def _handle_signal(signum, frame) -> None:
        _cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handle_signal)

    try:
        # Step 1: Start Backend Process
        print("\n[NOONE DEV] Starting Python backend on 127.0.0.1:8765...", flush=True)
        backend_proc = subprocess.Popen(
            [python_exe, "-m", "noone.ui", "--backend-only"],
            cwd=str(root_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        t_backend = threading.Thread(
            target=_pipe_reader,
            args=(backend_proc.stdout, "backend", stop_event),
            daemon=True,
        )
        t_backend.start()

        # Verify backend readiness
        backend_ready = _wait_for_url(
            "http://127.0.0.1:8765/api/bootstrap",
            timeout=5.0,
            headers={"X-NOONE-Token": "__NOONE_TOKEN_VALUE__"},
        )
        if not backend_ready and backend_proc.poll() is not None:
            ret = backend_proc.poll()
            print(f"[NOONE DEV] ERROR: Backend failed to start (exit code {ret}).", file=sys.stderr, flush=True)
            _cleanup()
            return 1

        # Step 2: Start Frontend Process
        print("[NOONE DEV] Starting Vite dev server on port 5173...", flush=True)
        frontend_proc = subprocess.Popen(
            [npm_cmd, "run", "dev"],
            cwd=str(frontend_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        t_frontend = threading.Thread(
            target=_pipe_reader,
            args=(frontend_proc.stdout, "frontend", stop_event),
            daemon=True,
        )
        t_frontend.start()

        # Verify frontend readiness
        frontend_ready = _wait_for_url("http://localhost:5173/", timeout=10.0)
        if not frontend_ready and frontend_proc.poll() is not None:
            ret = frontend_proc.poll()
            print(f"[NOONE DEV] ERROR: Frontend failed to start (exit code {ret}).", file=sys.stderr, flush=True)
            _cleanup()
            return 1

        print("\n" + "=" * 66, flush=True)
        print("  SERVICES RUNNING & READY:", flush=True)
        print("  - Backend:  http://127.0.0.1:8765/ (dev_mode=True)", flush=True)
        print("  - Frontend: http://localhost:5173/", flush=True)
        print("  - Proxy:    http://localhost:5173/api/ -> http://127.0.0.1:8765/api/", flush=True)
        print("  Press Ctrl+C to stop both processes cleanly.", flush=True)
        print("=" * 66 + "\n", flush=True)

        # Step 3: Supervision loop
        while not stop_event.is_set():
            ret_backend = backend_proc.poll()
            if ret_backend is not None:
                print(
                    f"\n[NOONE DEV] ERROR: Backend process (PID {backend_proc.pid}) exited unexpectedly with code {ret_backend}!",
                    file=sys.stderr,
                    flush=True,
                )
                break

            ret_frontend = frontend_proc.poll()
            if ret_frontend is not None:
                print(
                    f"\n[NOONE DEV] ERROR: Frontend process (PID {frontend_proc.pid}) exited unexpectedly with code {ret_frontend}!",
                    file=sys.stderr,
                    flush=True,
                )
                break

            time.sleep(0.5)

    except KeyboardInterrupt:
        pass
    finally:
        _cleanup()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
