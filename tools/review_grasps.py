#!/usr/bin/env python3
"""Visual review of the grasp windows that will become ACT training data.

The point: before a 35k-step training run, look at every clip that goes in and
throw out the bad ones. This renders one card per grasp window showing the two
cameras the policy will actually see, at the three moments that matter, plus a
coverage map of where in the workspace the grasps happened.

WHY IT IMPORTS FROM export_lerobot RATHER THAN RE-DERIVING: a review tool that
computes its own episode selection and window boundaries is reviewing a dataset
that does not exist. `usable_grasps` and `grasp_windows` are imported, so what
you see here is what gets written, including the overlap clipping that makes
neighbouring windows shorter than pre_s + post_s.

THE THREE MOMENTS, and why these three:

    t_lo ............... t_close ............... t_hi
    window start        gripper closes         window end
    |                   |                      |
    the FIRST frame     the grasp itself       the lift
    the policy sees.    is the bag actually    did it come up,
    at deploy IK        under the fingers?     or did it slip?
    hands off here,
    so this frame must
    look reachable.

CAMERAS. Only camera_top and camera_left are shown, because those are the only
two the exporter writes (observation.images.top / .wrist). camera_scan looks at
the packet mat, is excluded from training, and showing it here would invite
judging the data on pixels the model never receives.

FRAME SEEKING. cv2's h264 seek lands on a keyframe, which can be most of a
second away -- at t_close that is the difference between "holding the bag" and
"about to touch it". So targets are visited in ascending order and decoded
FORWARD into position: seek to a little before, then read frame by frame to the
exact index. Accurate like a full decode, ~50x cheaper.

Usage:
    uv run python tools/review_grasps.py
    uv run python tools/review_grasps.py --pre-s 2.0 --post-s 1.0 --open
    uv run python tools/review_grasps.py --hover-x 0.43 --hover-y -0.25 --hover-z 0.17
"""
from __future__ import annotations

import argparse
import base64
import html
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robots_realtime.labeling import constants as C
from tools.export_lerobot import (  # noqa: E402
    CAMERAS,
    episode_dirs,
    grasp_windows,
    grasp_windows_indexed,
    load_json,
    usable_grasps,
    zone_label,
)

THUMB_W = 300
JPEG_Q = 82

# Coverage grid. 6x6 over the mat region keeps cells ~3cm, which is the scale
# the packets were actually placed at; finer just makes every cell read n=1.
GRID_N = 6


# ── collect ─────────────────────────────────────────────────────────────────
# Operator ruler + top-down depth, 2026-07-30. Read for the scatter background
# only -- nothing gates on it. Missing file is not an error: the tray is context,
# not data, and a review that refuses to run because a drawing is absent is worse
# than one that runs without the drawing.
TRAY_JSON = Path(__file__).resolve().parents[2] / "yam-pick-pipeline/results/tray_box.json"


def load_tray():
    try:
        t = json.loads(TRAY_JSON.read_text())
        cx, cy = t["centre_xy"]
        sx, sy = t["size_xyz"][0], t["size_xyz"][1]
        return {"x0": cx - sx / 2, "x1": cx + sx / 2,
                "y0": cy - sy / 2, "y1": cy + sy / 2}
    except Exception:
        return {"x0": 0.0, "x1": 0.0, "y0": 0.0, "y1": 0.0}


def failed_attempts(root: Path):
    """Grasps that did NOT succeed, for the scatter only.

    They never become training data -- usable_grasps filters them out long
    before here -- but where they happened is the whole evidence base for
    calling one part of the mat worse than another. Showing the successes alone
    would present a zone choice with the reason for it edited out.
    """
    out = []
    for ep in episode_dirs(root):
        ann = load_json(ep / "annotations.json") or {}
        if (ann.get("episode_meta") or {}).get("outcome") == "aborted":
            continue
        for i, a in enumerate(ann.get("grasp_attempts") or []):
            if a.get("outcome") == "success":
                continue
            pose = a.get("ee_pose") or []
            if len(pose) < 2:
                continue
            out.append({"uid": f"{ep.name}!{i}", "ep": ep.name, "idx": i,
                        "x": float(pose[0]), "y": float(pose[1]),
                        "outcome": a.get("outcome") or "?"})
    return out


def collect(root: Path, pre_s: float, post_s: float,
            x_min: float | None = None, y_max: float | None = None,
            arm: str = "left"):
    """One record per grasp window, in exporter order. Also the rejected episodes."""
    records, rejects = [], []

    for ep in episode_dirs(root):
        # workspace_gate=False so the dropped grasps still appear, badged, with
        # the reason visible. The gate itself is applied below from the same
        # constant the exporter uses -- never re-derived here.
        grasps, why = usable_grasps(ep, workspace_gate=False, arm=arm)
        if why:
            rejects.append((ep.name, why))
            continue

        stamps = {}
        for cam in CAMERAS:
            f = ep / f"{cam}-rgb-timestamp.npy"
            if f.exists():
                stamps[cam] = np.load(f)
        if not stamps:
            rejects.append((ep.name, "no camera timestamps"))
            continue

        # Window clipping needs the recorded span. The camera streams bracket it
        # closely enough for review, and using them avoids opening both mcaps
        # (~2s per episode) just to read two numbers.
        t0 = max(float(s[0]) for s in stamps.values())
        t1 = min(float(s[-1]) for s in stamps.values())

        ordered = sorted((g for g in grasps if g.get("t") is not None),
                         key=lambda g: float(g["t"]))
        # Indexed form keeps grasp identity, so a window that clips to nothing
        # drops ONLY its own grasp. The old code rejected the whole episode on a
        # length mismatch, which threw away every grasp in a take whose wrist
        # camera died near the end -- while the exporter wrote those same grasps.
        idx_windows = grasp_windows_indexed(ordered, t0, t1, pre_s, post_s)
        if not idx_windows:
            rejects.append((ep.name, f"{len(ordered)} grasps, no window inside the recorded span"))
            continue
        if len(idx_windows) != len(ordered):
            n_clip = len(ordered) - len(idx_windows)
            rejects.append((ep.name,
                            f"NOTE {n_clip}/{len(ordered)} grasps outside the camera span "
                            f"(a camera stopped early) -- the other {len(idx_windows)} are shown"))
        ordered = [ordered[i] for i, _, _ in idx_windows]
        windows = [(lo, hi) for _, lo, hi in idx_windows]

        ann = load_json(ep / "annotations.json") or {}
        kit = {k.get("bag_id"): k for k in ((ann.get("episode_meta") or {}).get("kitting_list") or [])}
        flags = load_json(ep / "operator_flags.json") or {}
        ep_tags = sorted({f.get("tag") for f in flags.get("flags", []) if f.get("tag")})

        for i, (g, (lo, hi)) in enumerate(zip(ordered, windows)):
            pose = g.get("ee_pose") or []
            x, y, z = (list(pose[:3]) + [None, None, None])[:3]
            bag = kit.get(g.get("bag_id")) or {}
            # zone_label is the EXPORTER's classifier, imported not re-derived.
            # The sliders in the page recompute this in JS for exploration, but
            # what lands here -- and what the printed export command reproduces
            # -- is always this call. See the module docstring.
            zone = zone_label(g, x_min, y_max)
            records.append({
                "zone": zone,
                "ep": ep.name,
                "date": ep.parent.name,
                "dir": str(ep),
                "idx": i,
                "uid": f"{ep.name}#{i}",
                "bag_id": g.get("bag_id"),
                "part_no": bag.get("part_no") or "",
                "part_name": bag.get("name") or "",
                "attempt": g.get("attempt"),
                "t_close": float(g["t"]),
                "t_lo": lo, "t_hi": hi,
                "dur": hi - lo,
                "x": x, "y": y, "z": z,
                "excluded": zone != "in",
                "ep_tags": ep_tags,
                "frames": {},
            })
    return records, rejects


# ── frames ──────────────────────────────────────────────────────────────────
def grab_frames(records, outdir: Path, quiet: bool = False):
    """Decode the 3 moments x 2 cameras for every record. One pass per video."""
    outdir.mkdir(parents=True, exist_ok=True)
    by_ep: dict[str, list] = {}
    for r in records:
        by_ep.setdefault(r["dir"], []).append(r)

    for n, (epdir, recs) in enumerate(sorted(by_ep.items()), 1):
        ep = Path(epdir)
        if not quiet:
            print(f"  [{n}/{len(by_ep)}] {ep.name}  ({len(recs)} grasps)", flush=True)

        for cam, suffix in CAMERAS.items():
            mp4 = ep / f"{cam}-images-rgb.mp4"
            npy = ep / f"{cam}-rgb-timestamp.npy"
            if not mp4.exists() or not npy.exists():
                continue
            ts = np.load(npy)

            # (frame index, record, moment) sorted ascending so one forward pass
            # can satisfy every target.
            targets = []
            for r in recs:
                for moment, t in (("lo", r["t_lo"]), ("close", r["t_close"]), ("hi", r["t_hi"])):
                    targets.append((int(np.argmin(np.abs(ts - t))), r, moment))
            targets.sort(key=lambda a: a[0])

            cap = cv2.VideoCapture(str(mp4))
            if not cap.isOpened():
                continue
            pos = -1
            try:
                for idx, r, moment in targets:
                    if pos < 0 or idx < pos or idx - pos > 90:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, idx - 45))
                        pos = max(0, idx - 45)
                    frame = None
                    while pos <= idx:
                        ok, f = cap.read()
                        if not ok:
                            break
                        frame = f
                        pos += 1
                    if frame is None:
                        continue
                    h, w = frame.shape[:2]
                    thumb = cv2.resize(frame, (THUMB_W, max(1, int(h * THUMB_W / w))),
                                       interpolation=cv2.INTER_AREA)
                    name = f"{r['ep']}_{r['idx']}_{suffix}_{moment}.jpg"
                    cv2.imwrite(str(outdir / name), thumb,
                                [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_Q])
                    r["frames"][f"{suffix}_{moment}"] = name
            finally:
                cap.release()


# ── coverage ────────────────────────────────────────────────────────────────
def coverage(records, hover):
    """Bin kept grasps into a GRID_N x GRID_N map and locate the handoff pose.

    Bounds come from the KEPT grasps only. Including the excluded outliers would
    stretch the map 30cm to the west and squash everything real into two cells.
    """
    kept = [r for r in records if not r["excluded"] and r["x"] is not None]
    if not kept:
        return None
    xs = [r["x"] for r in kept]
    ys = [r["y"] for r in kept]
    pad = 1e-6
    x0, x1 = min(xs) - pad, max(xs) + pad
    y0, y1 = min(ys) - pad, max(ys) + pad

    def cell(x, y):
        cx = int((x - x0) / (x1 - x0) * GRID_N)
        cy = int((y - y0) / (y1 - y0) * GRID_N)
        if not (0 <= cx < GRID_N and 0 <= cy < GRID_N):
            return None
        return cx, cy

    counts = [[0] * GRID_N for _ in range(GRID_N)]
    for r in kept:
        c = cell(r["x"], r["y"])
        if c:
            counts[c[0]][c[1]] += 1
            r["cell"] = f"{c[0]},{c[1]}"

    # The verdict reads the 3x3 NEIGHBOURHOOD, not the single cell. Cells are
    # ~30x49mm while the grasp scatter is ~80mm per axis and the calibration is
    # good to ~5mm -- so one cell is inside the noise, and a dense region with a
    # single empty cell in it would otherwise read as "no data here". Measured
    # case: the corpus mean (0.43, -0.25) lands in a cell of n=1 whose neighbours
    # hold 25 grasps. Reporting n=1 there would be true and useless.
    verdict, hcell, hood = "NO HANDOFF POSE GIVEN", None, None
    if hover is not None:
        c = cell(hover[0], hover[1])
        if c is None:
            verdict = "OUT OF DISTRIBUTION — outside the grasped region entirely"
        else:
            hcell = c
            hood = sum(counts[i][j]
                       for i in range(max(0, c[0] - 1), min(GRID_N, c[0] + 2))
                       for j in range(max(0, c[1] - 1), min(GRID_N, c[1] + 2)))
            n = counts[c[0]][c[1]]
            if hood == 0:
                verdict = "OUT OF DISTRIBUTION — no grasps within one cell"
            elif hood < 5:
                verdict = f"THIN — only {hood} grasps nearby (cell n={n})"
            else:
                verdict = f"IN DISTRIBUTION — {hood} grasps nearby (cell n={n})"

    return {"counts": counts, "x0": x0, "x1": x1, "y0": y0, "y1": y1,
            "hcell": hcell, "hood": hood, "verdict": verdict,
            "hover": hover, "n": len(kept)}


# ── zone scatter ────────────────────────────────────────────────────────────
# Robot base frame, metres. +x away from the robot toward the packet mat, -y to
# the robot's left. Drawn with x ACROSS and y UP (y negated) so the picture
# matches the operator's view of the table rather than the raw axes.
SVG_W, SVG_H, SVG_PAD = 620, 430, 44


def scatter_svg(records, fails, tray):
    """Every grasp as one dot, with the tray footprint and the two zone edges.

        y_max ─── ─── ─── ─── ─── ───   drag me
              ┌──────────────────────┐
              │  ●   ●    ●    ●   ● │  in zone
              │    ●   ●     ●   ●   │
              └──────────────────────┘
             x_min                       drag me too

    The dots are positioned once, in Python, from real coordinates. Only their
    CLASS changes as the sliders move -- so a dot can never end up somewhere the
    exporter would not put it.
    """
    pts = [r for r in records if r["x"] is not None and r["y"] is not None]
    if not pts:
        return "<p class='muted'>No poses to map.</p>"

    xs = [r["x"] for r in pts + fails] + [tray["x0"], tray["x1"]]
    ys = [r["y"] for r in pts + fails] + [tray["y0"], tray["y1"]]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    mx = (x1 - x0) * 0.06 or 0.01
    my = (y1 - y0) * 0.06 or 0.01
    x0, x1, y0, y1 = x0 - mx, x1 + mx, y0 - my, y1 + my

    def px(x):
        return SVG_PAD + (x - x0) / (x1 - x0) * (SVG_W - 2 * SVG_PAD)

    def py(y):                       # y negated: -y is up the page
        return SVG_PAD + (y1 - y) / (y1 - y0) * (SVG_H - 2 * SVG_PAD)

    # gridlines every 10 cm, labelled, so distances are readable off the page
    grid = []
    gx = math.ceil(x0 * 10) / 10
    while gx <= x1:
        X = px(gx)
        grid.append(f'<line class="gl" x1="{X:.1f}" y1="{SVG_PAD}" '
                    f'x2="{X:.1f}" y2="{SVG_H - SVG_PAD}"/>'
                    f'<text class="gt" x="{X:.1f}" y="{SVG_H - SVG_PAD + 15}" '
                    f'text-anchor="middle">{gx:.1f}</text>')
        gx = round(gx + 0.1, 4)
    gy = math.ceil(y0 * 10) / 10
    while gy <= y1:
        Y = py(gy)
        grid.append(f'<line class="gl" x1="{SVG_PAD}" y1="{Y:.1f}" '
                    f'x2="{SVG_W - SVG_PAD}" y2="{Y:.1f}"/>'
                    f'<text class="gt" x="{SVG_PAD - 7}" y="{Y + 3.5:.1f}" '
                    f'text-anchor="end">{gy:+.1f}</text>')
        gy = round(gy + 0.1, 4)

    tx, tw = px(tray["x0"]), px(tray["x1"]) - px(tray["x0"])
    ty, th = py(tray["y1"]), py(tray["y0"]) - py(tray["y1"])

    # Failures first so a successful grasp is never hidden under a cross.
    dots = []
    for f in fails:
        X, Y = px(f["x"]), py(f["y"])
        dots.append(
            f'<g class="xmark"><line x1="{X - 4:.1f}" y1="{Y - 4:.1f}" '
            f'x2="{X + 4:.1f}" y2="{Y + 4:.1f}"/>'
            f'<line x1="{X + 4:.1f}" y1="{Y - 4:.1f}" x2="{X - 4:.1f}" y2="{Y + 4:.1f}"/>'
            f'<title>{html.escape(f["ep"])}  x {f["x"]:+.3f}  y {f["y"]:+.3f}  '
            f'{html.escape(f["outcome"])} (never training data)</title></g>')
    for r in sorted(pts, key=lambda r: r["zone"] == "in"):   # in-zone drawn last
        dots.append(
            f'<circle class="dot" data-uid="{html.escape(r["uid"])}" '
            f'cx="{px(r["x"]):.1f}" cy="{py(r["y"]):.1f}" r="5" '
            f'data-x="{r["x"]:.4f}" data-y="{r["y"]:.4f}">'
            f'<title>{html.escape(r["ep"])} #{r["idx"]}  '
            f'x {r["x"]:+.3f}  y {r["y"]:+.3f}  success</title></circle>')

    return f"""<svg id="scat" viewBox="0 0 {SVG_W} {SVG_H}" width="100%">
  <g>{''.join(grid)}</g>
  <rect class="tray" x="{tx:.1f}" y="{ty:.1f}" width="{tw:.1f}" height="{th:.1f}"/>
  <text class="tl" x="{tx + 6:.1f}" y="{ty + 15:.1f}">tray</text>
  <rect id="zbox" class="zbox" x="0" y="0" width="0" height="0"/>
  <line id="lx" class="edge" y1="{SVG_PAD}" y2="{SVG_H - SVG_PAD}"/>
  <line id="ly" class="edge" x1="{SVG_PAD}" x2="{SVG_W - SVG_PAD}"/>
  <g id="dots">{''.join(dots)}</g>
  <text class="ax" x="{SVG_W / 2:.0f}" y="{SVG_H - 6}" text-anchor="middle">
    x forward (m) &rarr;</text>
  <text class="ax" x="13" y="{SVG_H / 2:.0f}" text-anchor="middle"
    transform="rotate(-90 13 {SVG_H / 2:.0f})">&larr; y toward robot's left (m)</text>
</svg>
<script>
window.SCAT = {{x0:{x0:.5f}, x1:{x1:.5f}, y0:{y0:.5f}, y1:{y1:.5f},
  pad:{SVG_PAD}, w:{SVG_W}, h:{SVG_H}}};
</script>"""


# ── render ──────────────────────────────────────────────────────────────────
def render(records, rejects, cov, args, outdir: Path, fails=(), tray=None):
    kept = [r for r in records if not r["excluded"]]
    esc = html.escape
    tray = tray or load_tray()
    fails = list(fails)

    def cov_html():
        if not cov:
            return "<p class='muted'>No kept grasps to map.</p>"
        c = cov
        vk = "good" if c["verdict"].startswith("IN DIST") else (
            "warn" if c["verdict"].startswith("THIN") else "bad")
        cells = []
        mx = max(max(row) for row in c["counts"]) or 1
        for cx in range(GRID_N - 1, -1, -1):          # high x at the top
            for cy in range(GRID_N):
                n = c["counts"][cx][cy]
                lvl = 0 if n == 0 else 1 + int(3.99 * n / mx)
                mark = " hov" if c["hcell"] == (cx, cy) else ""
                xa = c["x0"] + (c["x1"] - c["x0"]) * cx / GRID_N
                ya = c["y0"] + (c["y1"] - c["y0"]) * cy / GRID_N
                title = f"x {xa:+.3f} y {ya:+.3f} — {n} grasp{'' if n == 1 else 's'}"
                dis = "" if n else " disabled"
                cells.append(
                    f'<button class="cell l{lvl}{mark}" data-cell="{cx},{cy}"'
                    f' title="{title}"{dis}>{n or ""}</button>')
        vclass = vk
        hov = ""
        if c["hover"]:
            hov = (f'<div class="hovline">handoff pose '
                   f'<code>x {c["hover"][0]:+.3f}  y {c["hover"][1]:+.3f}  '
                   f'z {c["hover"][2]:+.3f}</code> '
                   f'<span class="verdict {vclass}">{esc(c["verdict"])}</span></div>')
        return (f'<div class="covwrap"><div class="axis-y">y (lateral) →</div>'
                f'<div class="covgrid">{"".join(cells)}</div>'
                f'<div class="axis-x">x (forward) ↑</div></div>{hov}'
                f'<p class="muted">Click a cell to filter. Bounds x '
                f'{c["x0"]:+.3f}..{c["x1"]:+.3f}, y {c["y0"]:+.3f}..{c["y1"]:+.3f} m, '
                f'{c["n"]} kept grasps.</p>')

    def card(r):
        thumbs = []
        for suffix, label in (("top", "top"), ("wrist", "wrist")):
            row = []
            for moment, mlabel in (("lo", "start"), ("close", "GRASP"), ("hi", "lift")):
                f = r["frames"].get(f"{suffix}_{moment}")
                cls = "shot key" if moment == "close" else "shot"
                if f:
                    row.append(f'<figure class="{cls}"><img loading="lazy" '
                               f'src="frames/{esc(f)}" alt="{suffix} {mlabel}">'
                               f'<figcaption>{mlabel}</figcaption></figure>')
                else:
                    row.append(f'<figure class="{cls} miss"><div class="ph">no frame</div>'
                               f'<figcaption>{mlabel}</figcaption></figure>')
            thumbs.append(f'<div class="camrow"><span class="camlabel">{label}</span>'
                          f'<div class="shots">{"".join(row)}</div></div>')

        badges = ['<span class="zbadge"></span>']   # filled by JS, see paint()
        for t in r["ep_tags"]:
            badges.append(f'<span class="badge warn">flag {esc(t)}</span>')
        if r["dur"] < args.pre_s + args.post_s - 0.05:
            badges.append(f'<span class="badge warn">clipped {r["dur"]:.2f}s</span>')

        checked = "" if r["excluded"] else " checked"
        pos = (f'{r["x"]:+.3f} {r["y"]:+.3f} {r["z"]:+.3f}'
               if r["x"] is not None else "no pose")
        part = ""
        if r["part_no"]:
            part = (f'<div class="part">{esc(r["part_no"])} '
                    f'{esc(r["part_name"])}</div>')
        excl_cls = " excl" if r["excluded"] else ""
        dx = "" if r["x"] is None else f"{r['x']:.4f}"
        dy = "" if r["y"] is None else f"{r['y']:.4f}"
        return (
            f'<article class="card{excl_cls}" '
            f'data-uid="{esc(r["uid"])}" data-ep="{esc(r["ep"])}" '
            f'data-x="{dx}" data-y="{dy}" '
            f'data-cell="{esc(r.get("cell", ""))}" data-excl="{int(r["excluded"])}">'
            f'<header><label class="pick"><input type="checkbox"{checked}>'
            f'<span class="uid">{esc(r["ep"])}</span>'
            f'<span class="gi">#{r["idx"]}</span></label>'
            f'<div class="meta"><code>{pos}</code>'
            f'<span class="sep">·</span><span>{r["dur"]:.2f}s</span>'
            f'<span class="sep">·</span><span>bag {r["bag_id"]}</span>'
            f'{" ".join(badges)}</div>{part}'
            f'</header>{"".join(thumbs)}</article>')

    rej = "".join(f"<tr><td><code>{esc(e)}</code></td><td>{esc(w)}</td></tr>"
                  for e, w in rejects)

    payload = base64.b64encode(json.dumps(
        [{"uid": r["uid"], "ep": r["ep"], "dir": r["dir"], "idx": r["idx"],
          "t_close": r["t_close"], "t_lo": r["t_lo"], "t_hi": r["t_hi"],
          "x": r["x"], "y": r["y"], "zone": r["zone"],
          "excluded": r["excluded"]} for r in records]
    ).encode()).decode()

    xm = C.GRASP_WORKSPACE_X_MIN if args.zone_x_min is None else args.zone_x_min
    ym = C.GRASP_ZONE_Y_MAX if args.zone_y_max is None else args.zone_y_max
    # Slider travel is the data's own spread, so an edge can never be dragged
    # somewhere no grasp lives (a bound past every point reads as "0 selected"
    # with no way to tell whether that is the filter or a bug).
    pxs = [r["x"] for r in records if r["x"] is not None]
    pys = [r["y"] for r in records if r["y"] is not None]

    return TEMPLATE.format(
        n_total=len(records), n_kept=len(kept),
        n_excl=len(records) - len(kept), n_eps=len({r["ep"] for r in records}),
        n_rej=len(rejects), pre=args.pre_s, post=args.post_s, grid=GRID_N,
        scatter=scatter_svg(records, fails, tray),
        n_fail=len(fails),
        xmin=f"{xm:.3f}", ymax=("" if ym is None else f"{ym:.3f}"),
        ymax_off=("true" if ym is None else "false"),
        x_lo=f"{min(pxs, default=0.0) - 0.01:.3f}", x_hi=f"{max(pxs, default=1.0) + 0.01:.3f}",
        y_lo=f"{min(pys, default=-0.5) - 0.01:.3f}", y_hi=f"{max(pys, default=0.0) + 0.01:.3f}",
        root=esc(str(args.root)), repo=esc(args.repo_id),
        coverage=cov_html(), cards="".join(card(r) for r in records),
        rejects=rej or '<tr><td colspan="2" class="muted">none</td></tr>',
        payload=payload)


TEMPLATE = """<!doctype html>
<meta charset="utf-8"><title>Grasp review — {n_kept} training windows</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root {{
  --bg:#0e1113; --panel:#161a1d; --line:#252b30; --line2:#1b2024;
  --ink:#e6e9ea; --dim:#8d979c; --faint:#5e686d;
  --ok:#5db87a; --warn:#d8a13e; --bad:#e0705c; --acc:#4aa8b8;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
}}
@media (prefers-color-scheme: light) {{
  :root {{ --bg:#f4f5f3; --panel:#fff; --line:#dcdfdc; --line2:#eceeec;
    --ink:#171a1b; --dim:#5c6568; --faint:#8c9497;
    --ok:#2f7d4a; --warn:#8a6410; --bad:#a8402c; --acc:#0f6d7d; }}
}}
* {{ box-sizing:border-box; margin:0; padding:0 }}
body {{ background:var(--bg); color:var(--ink); font:14px/1.5 system-ui,sans-serif; padding:22px }}
.wrap {{ max-width:1500px; margin:0 auto }}
h1 {{ font-size:19px; font-weight:650; letter-spacing:-.01em }}
h2 {{ font-size:11px; font-weight:600; letter-spacing:.13em; text-transform:uppercase;
     color:var(--dim); margin:26px 0 11px; padding-bottom:6px; border-bottom:1px solid var(--line) }}
code {{ font-family:var(--mono); font-size:12px }}
.muted {{ color:var(--faint); font-size:12px; margin-top:8px }}
.sep {{ color:var(--faint); margin:0 3px }}

.stats {{ display:flex; flex-wrap:wrap; gap:22px; margin:12px 0 4px; font-family:var(--mono); font-size:12px; color:var(--dim) }}
.stats b {{ color:var(--ink); font-weight:650; font-size:15px }}

.bar {{ position:sticky; top:0; z-index:20; background:var(--panel); border:1px solid var(--line);
  border-radius:9px; padding:11px 13px; margin:16px 0 6px; display:flex; flex-wrap:wrap; gap:9px; align-items:center }}
button {{ font:inherit; font-size:12.5px; color:var(--ink); background:var(--bg);
  border:1px solid var(--line); border-radius:6px; padding:6px 11px; cursor:pointer }}
button:hover:not(:disabled) {{ border-color:var(--acc); color:var(--acc) }}
button:disabled {{ opacity:.32; cursor:default }}
button.on {{ background:var(--acc); border-color:var(--acc); color:#04191d; font-weight:600 }}
.count {{ margin-left:auto; font-family:var(--mono); font-size:12px; color:var(--dim) }}
.count b {{ color:var(--acc); font-size:14px }}

.covwrap {{ display:inline-grid; grid-template-columns:auto auto; grid-template-rows:auto auto;
  gap:6px; align-items:center }}
.covgrid {{ display:grid; grid-template-columns:repeat({grid},34px); gap:3px }}
.axis-y {{ grid-column:2; font-size:10px; color:var(--faint); font-family:var(--mono) }}
.axis-x {{ grid-column:1; grid-row:2; writing-mode:vertical-rl; transform:rotate(180deg);
  font-size:10px; color:var(--faint); font-family:var(--mono) }}
.cell {{ height:34px; padding:0; border-radius:4px; font-family:var(--mono); font-size:11px }}
.cell.l0 {{ background:var(--line2); color:var(--faint); border-color:var(--line2) }}
.cell.l1 {{ background:color-mix(in srgb,var(--acc) 16%,transparent) }}
.cell.l2 {{ background:color-mix(in srgb,var(--acc) 34%,transparent) }}
.cell.l3 {{ background:color-mix(in srgb,var(--acc) 56%,transparent) }}
.cell.l4 {{ background:color-mix(in srgb,var(--acc) 78%,transparent); color:#04191d; font-weight:700 }}
.cell.hov {{ outline:2px solid var(--warn); outline-offset:1px }}
.hovline {{ margin-top:11px; font-size:12.5px }}
.verdict {{ font-family:var(--mono); font-size:11px; font-weight:700; letter-spacing:.06em;
  padding:3px 8px; border-radius:4px; margin-left:7px }}
.verdict.good {{ background:color-mix(in srgb,var(--ok) 20%,transparent); color:var(--ok) }}
.verdict.warn {{ background:color-mix(in srgb,var(--warn) 20%,transparent); color:var(--warn) }}
.verdict.bad {{ background:color-mix(in srgb,var(--bad) 20%,transparent); color:var(--bad) }}

.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(470px,1fr)); gap:11px }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:9px; padding:11px }}
.card.excl {{ opacity:.5 }}
.card.sel {{ border-color:var(--acc) }}
.card.hide {{ display:none }}
.pick {{ display:flex; align-items:center; gap:7px; cursor:pointer }}
.pick input {{ width:15px; height:15px; accent-color:var(--acc); cursor:pointer }}
.uid {{ font-family:var(--mono); font-size:12px }}
.gi {{ font-family:var(--mono); font-size:11px; color:var(--faint) }}
.meta {{ display:flex; flex-wrap:wrap; align-items:center; gap:5px; margin-top:5px;
  font-size:11.5px; color:var(--dim) }}
.part {{ font-family:var(--mono); font-size:11px; color:var(--faint); margin-top:3px }}
.badge {{ font-family:var(--mono); font-size:10px; font-weight:600; padding:2px 6px; border-radius:3px }}
.badge.bad {{ background:color-mix(in srgb,var(--bad) 20%,transparent); color:var(--bad) }}
.badge.warn {{ background:color-mix(in srgb,var(--warn) 20%,transparent); color:var(--warn) }}

.camrow {{ display:flex; align-items:center; gap:8px; margin-top:9px }}
.camlabel {{ font-family:var(--mono); font-size:10px; color:var(--faint);
  writing-mode:vertical-rl; transform:rotate(180deg); letter-spacing:.09em; text-transform:uppercase }}
.shots {{ display:grid; grid-template-columns:repeat(3,1fr); gap:4px; flex:1 }}
.shot img {{ width:100%; display:block; border-radius:4px; background:var(--line2) }}
.shot figcaption {{ font-family:var(--mono); font-size:9px; color:var(--faint);
  text-align:center; margin-top:2px; letter-spacing:.05em }}
.shot.key img {{ outline:1.5px solid var(--acc); outline-offset:-1.5px }}
.shot.key figcaption {{ color:var(--acc); font-weight:700 }}
.ph {{ aspect-ratio:16/9; background:var(--line2); border-radius:4px; display:grid;
  place-items:center; font-size:10px; color:var(--faint) }}

table {{ border-collapse:collapse; font-size:12.5px; width:100% }}
td {{ padding:6px 12px 6px 0; border-bottom:1px solid var(--line2); color:var(--dim) }}
dialog {{ background:var(--panel); color:var(--ink); border:1px solid var(--line);
  border-radius:10px; padding:16px; max-width:min(760px,92vw) }}
dialog::backdrop {{ background:#000a; }}
textarea {{ width:100%; height:290px; background:var(--bg); color:var(--ink); font-family:var(--mono);
  font-size:11.5px; border:1px solid var(--line); border-radius:6px; padding:9px; resize:vertical }}
.dlgbar {{ display:flex; gap:8px; margin-top:11px; align-items:center }}

/* zone picker */
.zone {{ display:grid; grid-template-columns:minmax(360px,1.15fr) minmax(280px,.85fr);
  gap:20px; align-items:start; background:var(--panel); border:1px solid var(--line);
  border-radius:10px; padding:15px }}
@media (max-width:940px) {{ .zone {{ grid-template-columns:1fr }} }}
#scat {{ display:block; max-width:100%; height:auto; touch-action:none }}
.gl {{ stroke:var(--line); stroke-width:1 }}
.gt {{ fill:var(--faint); font-family:var(--mono); font-size:9px }}
.ax {{ fill:var(--faint); font-family:var(--mono); font-size:10px }}
.tray {{ fill:color-mix(in srgb,var(--warn) 9%,transparent); stroke:var(--warn);
  stroke-width:1; stroke-dasharray:4 3 }}
.tl {{ fill:var(--warn); font-family:var(--mono); font-size:10px; opacity:.85 }}
.zbox {{ fill:color-mix(in srgb,var(--acc) 11%,transparent); stroke:none }}
.edge {{ stroke:var(--acc); stroke-width:2; stroke-dasharray:6 4; cursor:grab }}
.edge.drag {{ cursor:grabbing; stroke-width:3 }}
.dot {{ fill:var(--faint); stroke:var(--bg); stroke-width:1.2; transition:fill .1s }}
.dot.in {{ fill:var(--ok) }}
.dot.out {{ fill:var(--line); stroke:var(--faint); stroke-width:1 }}
.dot.pk {{ stroke:var(--ink); stroke-width:2.5 }}
.xmark line {{ stroke:var(--bad); stroke-width:2; stroke-linecap:round }}
.zctl label {{ display:block; font-size:11px; color:var(--dim); margin:0 0 4px;
  font-family:var(--mono); letter-spacing:.04em }}
.zctl label b {{ color:var(--acc); font-size:13px }}
.zrow {{ margin-bottom:15px }}
.zrow input[type=range] {{ width:100%; accent-color:var(--acc) }}
.zsum {{ font-family:var(--mono); font-size:12px; line-height:1.85; margin:4px 0 13px;
  padding:11px 12px; background:var(--bg); border:1px solid var(--line); border-radius:7px }}
.zsum .k {{ color:var(--dim) }}
.zsum .v {{ color:var(--ink); font-weight:650 }}
.zsum .big {{ color:var(--acc); font-size:17px; font-weight:700 }}
.legend {{ display:flex; flex-wrap:wrap; gap:13px; font-size:11px; color:var(--dim);
  font-family:var(--mono); margin-top:9px }}
.legend i {{ display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:4px }}
.cmd {{ font-family:var(--mono); font-size:11px; background:var(--bg); color:var(--ink);
  border:1px solid var(--line); border-radius:7px; padding:10px 11px; white-space:pre-wrap;
  word-break:break-all; line-height:1.65; -webkit-user-select:all; user-select:all }}
.tog {{ display:flex; align-items:center; gap:7px; font-size:12px; color:var(--dim);
  margin-bottom:11px; cursor:pointer }}
.tog input {{ width:15px; height:15px; accent-color:var(--acc); cursor:pointer }}
.badge.ok {{ background:color-mix(in srgb,var(--ok) 20%,transparent); color:var(--ok) }}
</style>
<div class="wrap">
<h1>Grasp review</h1>
<div class="stats">
  <span><b>{n_kept}</b> training windows</span>
  <span><b>{n_total}</b> grasps found</span>
  <span><b>{n_excl}</b> excluded</span>
  <span><b>{n_eps}</b> episodes</span>
  <span><b>{n_rej}</b> episodes rejected</span>
  <span>window &minus;{pre}s / +{post}s</span>
</div>

<h2>Pick the zone</h2>
<div class="zone">
  <div>
    {scatter}
    <div class="legend">
      <span><i style="background:var(--ok)"></i>in zone &rarr; trains</span>
      <span><i style="background:var(--line);outline:1px solid var(--faint)"></i>dropped</span>
      <span style="color:var(--bad)">&#10005; failed grasp (never trains)</span>
      <span style="color:var(--warn)">&#9647; tray footprint</span>
    </div>
  </div>
  <div class="zctl">
    <div class="zsum">
      <div><span class="big" id="zn">0</span> <span class="k">grasp windows selected</span></div>
      <div><span class="k">dropped near x&nbsp;</span><span class="v" id="zdx">0</span>
           <span class="sep">·</span>
           <span class="k">dropped past y&nbsp;</span><span class="v" id="zdy">0</span></div>
      <div><span class="k">episodes contributing&nbsp;</span><span class="v" id="zep">0</span>
           <span class="sep">·</span>
           <span class="k">failures inside zone&nbsp;</span><span class="v" id="zf">0</span></div>
    </div>
    <div class="zrow">
      <label>x_min <b id="lx_v">0</b> m &nbsp;<span class="k">drop grasps nearer than this</span></label>
      <input type="range" id="sx" min="{x_lo}" max="{x_hi}" step="0.001" value="{xmin}">
    </div>
    <div class="zrow">
      <label class="tog"><input type="checkbox" id="yon"> use a lateral bound (y_max)</label>
      <label>y_max <b id="ly_v">off</b> m &nbsp;<span class="k">drop grasps past this, toward the robot's right</span></label>
      <input type="range" id="sy" min="{y_lo}" max="{y_hi}" step="0.001" value="-0.130">
    </div>
    <p class="muted" style="margin-bottom:7px">Export this zone:</p>
    <div class="cmd" id="cmd"></div>
    <p class="muted">The page filters in the browser so you can explore. The dataset
    is cut by <code>export_lerobot.in_zone()</code> from the two numbers above &mdash;
    run the command and the counts must match.</p>
  </div>
</div>

<h2>Coverage of the kept grasps</h2>
{coverage}

<h2>Every window that goes into the model</h2>
<div class="bar">
  <button id="all">Select all</button>
  <button id="none">Select none</button>
  <button id="inv">Invert</button>
  <button id="fex" class="on">Hiding excluded</button>
  <button id="clr" disabled>Clear cell filter</button>
  <button id="copy">Copy selection</button>
  <span class="count"><b id="nsel">0</b> selected</span>
</div>
<div class="grid" id="grid">{cards}</div>

<h2>Episodes that never made it this far</h2>
<table><tbody>{rejects}</tbody></table>

<dialog id="dlg">
  <b>Selected grasp windows</b>
  <p class="muted">Episode list first, then the full keep-list as JSON.</p>
  <textarea id="out" readonly></textarea>
  <div class="dlgbar"><button id="cp">Copy to clipboard</button>
    <button id="close">Close</button><span class="muted" id="cpmsg"></span></div>
</dialog>
</div>
<script>
const DATA = JSON.parse(atob("{payload}"));
const byUid = Object.fromEntries(DATA.map(d => [d.uid, d]));
const cards = [...document.querySelectorAll('.card')];
let cellFilter = null, hideExcluded = true;

/* ── zone picker ──────────────────────────────────────────────────────────
   MIRRORS export_lerobot.in_zone(). If you change one, change the other:
   this decides what you SEE, that decides what the dataset CONTAINS, and the
   printed command is the bridge between them. Fails open on a missing pose,
   same as the exporter, for the same reason. */
const S = window.SCAT;
const sx = document.getElementById('sx'), sy = document.getElementById('sy');
const yon = document.getElementById('yon');
const dots = [...document.querySelectorAll('.dot')];
const FAILS = [...document.querySelectorAll('.xmark')].map(g => {{
  const t = g.querySelector('title').textContent.match(/x\\s*([-+.\\d]+)\\s+y\\s*([-+.\\d]+)/);
  return t ? {{x: +t[1], y: +t[2]}} : null;
}}).filter(Boolean);

const zx = () => +sx.value;
const zy = () => yon.checked ? +sy.value : null;

function inZone(x, y) {{
  if (x === null || x === undefined || x === '' || Number.isNaN(+x)) return true;
  if (+x < zx()) return false;
  const ym = zy();
  if (ym !== null && +y > ym) return false;
  return true;
}}

const toPx = (x) => S.pad + (x - S.x0) / (S.x1 - S.x0) * (S.w - 2 * S.pad);
const toPy = (y) => S.pad + (S.y1 - y) / (S.y1 - S.y0) * (S.h - 2 * S.pad);

function zonePaint() {{
  const ym = zy();
  sy.disabled = !yon.checked;
  document.getElementById('lx_v').textContent = zx().toFixed(3);
  document.getElementById('ly_v').textContent = ym === null ? 'off' : ym.toFixed(3);

  /* edges + shaded region */
  const X = toPx(zx()), Y = ym === null ? S.pad : toPy(ym);
  const lx = document.getElementById('lx'), ly = document.getElementById('ly');
  lx.setAttribute('x1', X); lx.setAttribute('x2', X);
  ly.setAttribute('y1', Y); ly.setAttribute('y2', Y);
  ly.style.display = ym === null ? 'none' : '';
  const zb = document.getElementById('zbox');
  zb.setAttribute('x', X); zb.setAttribute('y', Y);
  zb.setAttribute('width', Math.max(0, S.w - S.pad - X));
  zb.setAttribute('height', Math.max(0, S.h - S.pad - Y));

  let nIn = 0, dropX = 0, dropY = 0;
  const eps = new Set();
  dots.forEach(d => {{
    const x = +d.dataset.x, y = +d.dataset.y;
    const ok = inZone(x, y);
    d.classList.toggle('in', ok);
    d.classList.toggle('out', !ok);
    if (ok) {{ nIn++; eps.add(byUid[d.dataset.uid].ep); }}
    else if (x < zx()) dropX++;
    else dropY++;
  }});
  document.getElementById('zn').textContent = nIn;
  document.getElementById('zdx').textContent = dropX;
  document.getElementById('zdy').textContent = dropY;
  document.getElementById('zep').textContent = eps.size;
  document.getElementById('zf').textContent =
    FAILS.filter(f => inZone(f.x, f.y)).length;

  const yflag = ym === null ? '' : ` \\\\\\n  --zone-y-max {{Y}}`.replace('{{Y}}', ym.toFixed(3));
  document.getElementById('cmd').textContent =
    `.venv/bin/python tools/export_lerobot.py \\\\\\n` +
    `  --root {root} \\\\\\n  --repo-id {repo} \\\\\\n` +
    `  --zone-x-min ${{zx().toFixed(3)}}` + yflag;

  cards.forEach(c => {{
    const ok = inZone(c.dataset.x, c.dataset.y);
    c.dataset.excl = ok ? '0' : '1';
    c.classList.toggle('excl', !ok);
    const b = c.querySelector('.zbadge');
    if (ok) {{ b.className = 'badge ok zbadge'; b.textContent = 'in zone'; }}
    else if (+c.dataset.x < zx()) {{
      b.className = 'badge bad zbadge';
      b.textContent = 'dropped \\u00b7 x < ' + zx().toFixed(3);
    }} else {{
      b.className = 'badge bad zbadge';
      b.textContent = 'dropped \\u00b7 y > ' + (ym === null ? '' : ym.toFixed(3));
    }}
    c.querySelector('input').checked = ok;
  }});
  paint();
}}

/* drag the edges directly -- the sliders are the same two numbers */
function drag(line, horizontal) {{
  line.addEventListener('pointerdown', ev => {{
    if (horizontal && !yon.checked) return;
    line.classList.add('drag');
    line.setPointerCapture(ev.pointerId);
    const svg = document.getElementById('scat');
    const move = e => {{
      const r = svg.getBoundingClientRect();
      if (horizontal) {{
        const py = (e.clientY - r.top) / r.height * S.h;
        const v = S.y1 - (py - S.pad) / (S.h - 2 * S.pad) * (S.y1 - S.y0);
        sy.value = Math.min(+sy.max, Math.max(+sy.min, v));
      }} else {{
        const px = (e.clientX - r.left) / r.width * S.w;
        const v = S.x0 + (px - S.pad) / (S.w - 2 * S.pad) * (S.x1 - S.x0);
        sx.value = Math.min(+sx.max, Math.max(+sx.min, v));
      }}
      zonePaint();
    }};
    const up = e => {{
      line.classList.remove('drag');
      line.releasePointerCapture(ev.pointerId);
      svg.removeEventListener('pointermove', move);
      svg.removeEventListener('pointerup', up);
    }};
    svg.addEventListener('pointermove', move);
    svg.addEventListener('pointerup', up);
  }});
}}
drag(document.getElementById('lx'), false);
drag(document.getElementById('ly'), true);

sx.oninput = sy.oninput = yon.onchange = zonePaint;
yon.checked = !{ymax_off};
if ('{ymax}') sy.value = '{ymax}';

/* clicking a dot scrolls to its card */
dots.forEach(d => d.addEventListener('click', () => {{
  const c = cards.find(c => c.dataset.uid === d.dataset.uid);
  if (!c) return;
  if (c.classList.contains('hide')) {{ hideExcluded = false; fex.click(); }}
  c.scrollIntoView({{behavior: 'smooth', block: 'center'}});
  dots.forEach(o => o.classList.remove('pk'));
  d.classList.add('pk');
}}));

const boxOf = c => c.querySelector('input');
const sel = () => cards.filter(c => boxOf(c).checked);

function paint() {{
  cards.forEach(c => {{
    const okCell = !cellFilter || c.dataset.cell === cellFilter;
    const okExcl = !hideExcluded || c.dataset.excl === '0';
    c.classList.toggle('hide', !(okCell && okExcl));
    c.classList.toggle('sel', boxOf(c).checked);
  }});
  document.getElementById('nsel').textContent = sel().length;
}}

document.getElementById('grid').addEventListener('change', paint);

const visible = () => cards.filter(c => !c.classList.contains('hide'));
document.getElementById('all').onclick  = () => {{ visible().forEach(c => boxOf(c).checked = true);  paint(); }};
document.getElementById('none').onclick = () => {{ visible().forEach(c => boxOf(c).checked = false); paint(); }};
document.getElementById('inv').onclick  = () => {{ visible().forEach(c => boxOf(c).checked = !boxOf(c).checked); paint(); }};

const fex = document.getElementById('fex');
fex.onclick = () => {{
  hideExcluded = !hideExcluded;
  fex.classList.toggle('on', hideExcluded);
  fex.textContent = hideExcluded ? 'Hiding excluded' : 'Showing excluded';
  paint();
}};

const clr = document.getElementById('clr');
document.querySelectorAll('.cell[data-cell]:not([disabled])').forEach(b => {{
  b.onclick = () => {{
    cellFilter = (cellFilter === b.dataset.cell) ? null : b.dataset.cell;
    document.querySelectorAll('.cell').forEach(o => o.classList.toggle('on', o.dataset.cell === cellFilter));
    clr.disabled = !cellFilter;
    paint();
  }};
}});
clr.onclick = () => {{
  cellFilter = null;
  document.querySelectorAll('.cell').forEach(o => o.classList.remove('on'));
  clr.disabled = true; paint();
}};

const dlg = document.getElementById('dlg'), out = document.getElementById('out');
document.getElementById('copy').onclick = () => {{
  const picked = sel().map(c => byUid[c.dataset.uid]);
  const eps = [...new Set(picked.map(p => p.ep))].sort();
  out.value =
    `# ${{picked.length}} grasp windows across ${{eps.length}} episodes\\n` +
    eps.join('\\n') +
    `\\n\\n# keep-list\\n` +
    JSON.stringify({{keep: picked.map(p => ({{episode: p.ep, grasp: p.idx, t_close: p.t_close}}))}}, null, 2);
  dlg.showModal();
}};
document.getElementById('cp').onclick = async () => {{
  const msg = document.getElementById('cpmsg');
  try {{ await navigator.clipboard.writeText(out.value); msg.textContent = 'Copied.'; }}
  catch {{ out.select(); msg.textContent = 'Press Cmd/Ctrl+C.'; }}
  setTimeout(() => msg.textContent = '', 2500);
}};
document.getElementById('close').onclick = () => dlg.close();

zonePaint();          // paints the zone AND calls paint() for the cards
</script>
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="recordings")
    ap.add_argument("--out", default=".review-grasps")
    ap.add_argument("--repo-id", default="ETHRC/yam_grasp_v2",
                    help="repo-id printed in the page's export command")
    ap.add_argument("--pre-s", type=float, default=2.0)
    ap.add_argument("--post-s", type=float, default=1.0)
    ap.add_argument("--zone-x-min", type=float, default=None,
                    help=f"initial x bound (default {C.GRASP_WORKSPACE_X_MIN})")
    ap.add_argument("--zone-y-max", type=float, default=None,
                    help=f"initial y bound (default {C.GRASP_ZONE_Y_MAX} = off)")
    ap.add_argument("--hover-x", type=float, default=None,
                    help="planned IK handoff pose, robot base frame (metres)")
    ap.add_argument("--hover-y", type=float, default=None)
    ap.add_argument("--hover-z", type=float, default=0.17)
    ap.add_argument("--no-frames", action="store_true",
                    help="skip video decode, reuse whatever is already in out/frames")
    ap.add_argument("--arm", default="left", choices=("left", "right"),
                    help="which arm's annotations_<arm>.json to review. Also selects "
                         "the wrist camera (left->camera_left, right->camera_right), "
                         "matching what the exporter writes as observation.images.wrist.")
    ap.add_argument("--open", action="store_true")
    a = ap.parse_args(argv)

    root, outdir = Path(a.root), Path(a.out)
    # The wrist camera is arm-specific. CAMERAS is a module global read by
    # collect() and grab_frames(); rebinding it here keeps those two reading the
    # SAME dict rather than each deciding for itself which wrist to show.
    if a.arm == "right":
        global CAMERAS
        CAMERAS = {"camera_top": "top", "camera_right": "wrist"}
    records, rejects = collect(root, a.pre_s, a.post_s, a.zone_x_min, a.zone_y_max,
                               arm=a.arm)
    if not records:
        print("no grasp windows found", file=sys.stderr)
        return 1

    fails = failed_attempts(root)
    tray = load_tray()
    xm = C.GRASP_WORKSPACE_X_MIN if a.zone_x_min is None else a.zone_x_min
    ym = C.GRASP_ZONE_Y_MAX if a.zone_y_max is None else a.zone_y_max
    kept = [r for r in records if not r["excluded"]]
    print(f"{len(records)} successful grasps, {len(kept)} in zone "
          f"(x >= {xm}" + ("" if ym is None else f", y <= {ym}") + f"), "
          f"{len(fails)} failed attempts, {len(rejects)} episodes rejected")

    frames_dir = outdir / "frames"
    if a.no_frames:
        for r in records:
            for suffix in CAMERAS.values():
                for moment in ("lo", "close", "hi"):
                    f = f"{r['ep']}_{r['idx']}_{suffix}_{moment}.jpg"
                    if (frames_dir / f).exists():
                        r["frames"][f"{suffix}_{moment}"] = f
    else:
        print(f"decoding {len(records) * 6} frames...")
        grab_frames(records, frames_dir)

    hover = None
    if a.hover_x is not None and a.hover_y is not None:
        hover = (a.hover_x, a.hover_y, a.hover_z)
    cov = coverage(records, hover)
    if cov:
        print(f"coverage: {cov['verdict']}")

    outdir.mkdir(parents=True, exist_ok=True)
    index = outdir / "index.html"
    index.write_text(render(records, rejects, cov, a, outdir, fails, tray))
    print(f"\n  {index.resolve()}")
    if a.open:
        import webbrowser
        webbrowser.open(index.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
