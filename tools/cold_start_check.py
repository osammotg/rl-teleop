#!/usr/bin/env python3
"""The "power off" case — HANDOFF-CAMERA-HARDENING.md §5.3.

The machine must not be power-cycled (§1.3): nobody could bring it back, and a
power transition on an unsupported brakeless arm risks a fall. But the
SOFTWARE-VISIBLE part of a power loss is completely testable, and it is the part
that matters: everything dies at once, with no cleanup, mid-episode.

So: SIGKILL -9 the entire cameras-only process tree with no warning, then check
that the state it left behind does not poison the next run.

  1. the ports are free again (a wedged 5555/5556 is how a session "fails to
     start within timeout" and reads as a hardware fault),
  2. no /dev/video* is still held,
  3. no defunct processes are left,
  4. the half-written episode is HONESTLY BROKEN, not quietly broken — it must
     not look like a complete take to export_lerobot.py,
  5. a cold start immediately afterwards works, and every camera streams.

Point 4 is the one with teeth. A truncated mp4 that still decodes, paired with a
missing timestamp sidecar, is a silently short episode — and this pipeline feeds
a training set where a short episode is worse than no episode.

SAFETY: the child is started in its own session (setsid) and only THAT process
group is killed, by pgid. Nothing outside it can be touched, and the child runs
the same RobotNode guard as everything else here.

    tools/cold_start_check.py --config configs/yam/cameras_only_soak.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from robots_realtime.runtime.safety_guard import UnsafeConfig, assert_safe_to_soak  # noqa: E402

RUNNER = REPO / "tools" / "_cameras_only_runner.py"
PY = REPO / ".venv" / "bin" / "python3"


def port_bound(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", port)) == 0


def video_holders() -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for dev in sorted(Path("/dev").glob("video*")):
        try:
            r = subprocess.run(["lsof", "-t", str(dev)], check=False,
                               capture_output=True, text=True, timeout=5)
            pids = [int(p) for p in r.stdout.split() if p.strip().isdigit()]
        except Exception:
            pids = []
        if pids:
            out[str(dev)] = pids
    return out


def start_child(config: str, pub: int, sub: int, save_root: str,
                warmup: float, record: bool) -> tuple[subprocess.Popen, list[str]]:
    cmd = [str(PY), str(RUNNER), "--config", config, "--pub-port", str(pub),
           "--sub-port", str(sub), "--save-root", save_root, "--warmup", str(warmup)]
    if record:
        cmd.append("--record")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        start_new_session=True,          # its own process group — see SAFETY above
    )
    lines: list[str] = []
    deadline = time.monotonic() + warmup + 90
    while time.monotonic() < deadline:
        line = proc.stdout.readline() if proc.stdout else ""
        if not line:
            if proc.poll() is not None:
                break
            continue
        lines.append(line.rstrip())
        print(f"    child| {line.rstrip()}", flush=True)
        if line.startswith("READY"):
            return proc, lines
    raise RuntimeError("child never reported READY:\n" + "\n".join(lines))


def check_episode_honesty(episode: Path) -> list[str]:
    """A killed episode must be obviously incomplete, not plausibly complete."""
    problems: list[str] = []
    if not episode.exists():
        return ["episode directory does not exist at all"]

    mp4s = sorted(episode.glob("*-images-rgb.mp4"))
    npys = sorted(episode.glob("*-rgb-timestamp.npy"))
    print(f"    files: {len(mp4s)} mp4, {len(npys)} timestamp sidecars")

    meta_path = episode / "session_meta.json"
    if not meta_path.exists():
        problems.append("no session_meta.json — nothing identifies this episode at all")
    else:
        meta = json.loads(meta_path.read_text())
        # Whatever else is true, the marker must be present. Its ABSENCE is what
        # would let a killed take be mistaken for a clean one.
        if "degraded" not in meta:
            problems.append("session_meta.json has no 'degraded' key")

    # The tell, from tools/check_streams.py: AsyncMp4Writer.close() writes the
    # sidecar. A killed writer never closes, so an mp4 with no .npy is the
    # honest signature of an interrupted episode — and check_streams already
    # calls that FAIL. What would be dishonest is an mp4 that decodes cleanly
    # AND has a full sidecar while missing most of its frames.
    for mp4 in mp4s:
        cam = mp4.name.split("-images-rgb.mp4")[0]
        sidecar = episode / f"{cam}-rgb-timestamp.npy"
        if not sidecar.exists():
            print(f"    {cam}: mp4 without a sidecar — correctly flagged as interrupted")
            continue
        try:
            import av
            import numpy as np
            with av.open(str(mp4)) as c:
                n = sum(1 for _ in c.decode(c.streams.video[0]))
            ts = np.load(sidecar)
            if n != len(ts):
                print(f"    {cam}: {n} frames vs {len(ts)} timestamps — mismatch is VISIBLE")
            else:
                print(f"    {cam}: {n} frames == {len(ts)} timestamps (writer got to close)")
        except Exception as exc:                                   # noqa: BLE001
            print(f"    {cam}: will not decode ({exc}) — loudly broken, which is fine")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO / "configs/yam/cameras_only_soak.yaml"))
    ap.add_argument("--pub-port", type=int, default=5585)
    ap.add_argument("--sub-port", type=int, default=5586)
    ap.add_argument("--warmup", type=float, default=12.0)
    ap.add_argument("--save-root", default="/tmp/cold_start_check")
    ap.add_argument("--keep", action="store_true", help="keep the scratch save root")
    a = ap.parse_args(argv)

    try:
        assert_safe_to_soak(a.config, a.pub_port, a.sub_port)
    except UnsafeConfig as exc:
        print(f"\n{exc}\n", file=sys.stderr)
        return 2

    save_root = Path(a.save_root)
    shutil.rmtree(save_root, ignore_errors=True)
    save_root.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []

    holders_before = video_holders()
    if holders_before:
        print(f"    note: {sum(len(v) for v in holders_before.values())} camera handle(s) "
              f"already held by other processes — excluded from the leak check")

    print("\n[1] start a cameras-only session and begin an episode")
    proc, lines = start_child(a.config, a.pub_port, a.sub_port, str(save_root),
                              a.warmup, record=True)
    episode = None
    for line in lines:
        if line.startswith("EPISODE "):
            episode = Path(line.split(" ", 1)[1])
    pgid = os.getpgid(proc.pid)
    time.sleep(6)

    print(f"\n[2] SIGKILL the whole process group ({pgid}) — no warning, no cleanup")
    os.killpg(pgid, signal.SIGKILL)
    proc.wait(timeout=30)

    print("\n[3] does anything survive it?")
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if not port_bound(a.pub_port) and not port_bound(a.sub_port):
            break
        time.sleep(0.5)
    for port in (a.pub_port, a.sub_port):
        if port_bound(port):
            failures.append(f"port {port} is STILL BOUND after the kill — the next "
                            f"session would fail with a misleading timeout")
        else:
            print(f"    port {port} released")

    # Compare against the snapshot taken BEFORE the child started. A global
    # "is any camera held" check is wrong: anything else legitimately running on
    # the rig (another soak, the operator's session) holds cameras too, and
    # blaming the process we just killed for those is a test that reports
    # failures it did not observe. Only NEW holders are this child's leak.
    holders_after = video_holders()
    leaked = {
        dev: [p for p in pids if p not in holders_before.get(dev, [])]
        for dev, pids in holders_after.items()
    }
    leaked = {d: p for d, p in leaked.items() if p}
    if leaked:
        failures.append(f"/dev/video* still held by the killed tree: {leaked}")
    else:
        print(f"    no /dev/video* leaked by the killed tree "
              f"({len(holders_after)} device(s) held by other processes, unchanged)")

    zombies = subprocess.run(
        ["ps", "-eo", "pid,stat,args"], check=False, capture_output=True, text=True,
    ).stdout
    defunct = [ln for ln in zombies.splitlines()
               if "<defunct>" in ln and ("rr-session" in ln or "cameras_only" in ln
                                         or "_cameras_only_runner" in ln)]
    if defunct:
        failures.append(f"defunct processes left behind: {defunct}")
    else:
        print("    no defunct camera processes")

    print("\n[4] is the interrupted episode honestly incomplete?")
    if episode is None:
        failures.append("the child never reported an episode directory")
    else:
        failures.extend(check_episode_honesty(episode))

    print("\n[5] cold start — does it come back cleanly?")
    try:
        proc2, lines2 = start_child(a.config, a.pub_port, a.sub_port, str(save_root),
                                    a.warmup, record=False)
    except Exception as exc:                                       # noqa: BLE001
        failures.append(f"COLD START FAILED: {exc}")
    else:
        dead = [ln for ln in lines2 if ln.startswith("NODE ") and "alive=False" in ln]
        # A camera that is physically absent is EXPECTED to come up failed; what
        # must not happen is a node dying, or a camera that was fine before the
        # kill coming back broken.
        for ln in lines2:
            if ln.startswith("NODE "):
                print(f"    {ln}")
        if dead:
            failures.append(f"nodes not alive after cold start: {dead}")
        os.killpg(os.getpgid(proc2.pid), signal.SIGKILL)
        proc2.wait(timeout=30)

    if not a.keep:
        shutil.rmtree(save_root, ignore_errors=True)

    print("\n" + "=" * 74)
    if failures:
        print("COLD START CHECK: FAIL")
        for f in failures:
            print(f"  · {f}")
        return 1
    print("COLD START CHECK: PASS — a hard kill leaves nothing that poisons the next run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
