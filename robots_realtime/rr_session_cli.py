"""CLI entry point.

Usage:
    uv run -m robots_realtime configs/sessions/yam_sim_dummy.yaml
    uv run -m robots_realtime configs/sessions/yam_sim_dummy.yaml --no-tui
    uv run -m robots_realtime configs/sessions/yam_sim_dummy.yaml --save-root /data/rec

    # Legacy Python module path (backward compatibility):
    uv run -m robots_realtime configs.sessions.yam_sim_dummy  --no-tui
"""

from __future__ import annotations

import argparse
import importlib
import os
import signal
import sys


def _force_exit(sig, frame):
    """SIGTERM handler: give session.stop() 3 s then hard-kill the process group."""
    import threading
    import time

    def _kill_group():
        time.sleep(3.0)
        try:
            os.killpg(os.getpgid(0), signal.SIGKILL)
        except Exception:
            os._exit(1)

    threading.Thread(target=_kill_group, daemon=True).start()


signal.signal(signal.SIGTERM, _force_exit)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="robots_realtime",
        description="Launch a robots_realtime session.",
    )
    parser.add_argument(
        "session",
        help=(
            "Path to a YAML session config file (e.g. configs/sessions/yam_sim_dummy.yaml), "
            "or a dotted Python module path containing make_session() "
            "(e.g. configs.sessions.yam_sim_dummy)."
        ),
    )
    parser.add_argument(
        "--save-root",
        default=None,
        help="Override the session's default save_root for recordings.",
    )
    parser.add_argument(
        "--no-tui",
        action="store_true",
        help="Disable the Rich TUI and just block until Ctrl-C.",
    )
    parser.add_argument(
        "--pub-port",
        type=int,
        default=None,
        help="Override the MessageBus publisher/XSUB port. Defaults to 5555.",
    )
    parser.add_argument(
        "--sub-port",
        type=int,
        default=None,
        help="Override the MessageBus subscriber/XPUB port. Defaults to 5556.",
    )
    parser.add_argument(
        "--attach-bus",
        action="store_true",
        help=(
            "Attach to a MessageBus that is already running (see `rr-bus`) "
            "instead of starting one. Lets several sessions -- e.g. an "
            "always-on camera daemon plus a restartable arm session -- share "
            "one bus, and leaves the bus up when this session exits."
        ),
    )
    parser.add_argument(
        "--control-port",
        type=int,
        default=8792,
        help=(
            "Port for the HTTP control surface the cockpit drives "
            "(/status, /record/start, /record/save, ...). Defaults to 8792."
        ),
    )
    parser.add_argument(
        "--no-control",
        action="store_true",
        help="Do not start the HTTP control surface (keyboard-only session).",
    )
    args = parser.parse_args()

    session_arg: str = args.session

    # Determine whether this is a YAML file path or a Python module path
    is_yaml = session_arg.endswith(".yaml") or session_arg.endswith(".yml")
    is_file = os.path.exists(session_arg)

    if is_yaml or is_file:
        # YAML file path
        from robots_realtime.runtime.config import load_session
        try:
            session = load_session(
                session_arg,
                pub_port=args.pub_port,
                sub_port=args.sub_port,
            )
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error loading session config '{session_arg}': {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # Legacy Python module path
        try:
            mod = importlib.import_module(session_arg)
        except ModuleNotFoundError as e:
            print(f"Error: could not import '{session_arg}': {e}", file=sys.stderr)
            sys.exit(1)

        if not hasattr(mod, "make_session"):
            print(
                f"Error: '{session_arg}' has no make_session() function.",
                file=sys.stderr,
            )
            sys.exit(1)

        session = mod.make_session()
        if args.pub_port is not None or args.sub_port is not None:
            session.configure_bus_ports(pub_port=args.pub_port, sub_port=args.sub_port)

    if args.attach_bus:
        session.attach_to_existing_bus()

    # Allow save-root override
    if args.save_root:
        from pathlib import Path
        session._save_root = Path(args.save_root)

    session.start()

    # HTTP control surface — lets the cockpit drive the same session the
    # keyboard drives. Started after session.start() so /status is truthful
    # the moment it answers, and never fatal: a bound port must not stop a
    # recording session from running keyboard-only.
    control = None
    if not args.no_control:
        from robots_realtime.runtime.control_server import ControlServer
        control = ControlServer(session, port=args.control_port)
        if control.start():
            print(f"session control → {control.url}  (cockpit drives this)")
        else:
            print(
                f"session control → port {args.control_port} busy; "
                "continuing keyboard-only. Free the port or pass --control-port N."
            )
            control = None

    try:
        if args.no_tui:
            print(f"Session running. Ctrl-C to stop.  Recordings → {session.save_root}")
            session.wait()
        else:
            from robots_realtime.runtime.tui import run_tui
            run_tui(session)
    finally:
        if control is not None:
            control.stop()
        session.stop()

    os._exit(0)


if __name__ == "__main__":
    main()
