#!/usr/bin/env python3
"""Kitting data-collection review — corpus QA + pick/place distribution.

Standalone (stdlib only). Reads recordings/<date>/episode_*/annotations.json
(+ corrections.json overrides if present) and the shared compartments.json,
and emits:
  1. a corpus / label-quality audit (usable vs aborted vs mislabelled),
  2. pick-location statistics + per-bag consistency (the ACT "same op every
     time" signal),
  3. place accuracy vs the calibrated compartments,
  4. a clarity verdict,
  5. an HTML report with a top-down scatter (picks + places over compartments)
     and a playable card per episode (top / scan / wrist video).

Usage:
  python3 review_corpus.py [recordings_dir] [--html out.html] [--media-prefix recordings]

--media-prefix is the URL path under which the recordings tree is reachable from
the page (review_watch.sh symlinks it into the served dir as "recordings/").
"""
from __future__ import annotations
import glob, html as _html, json, math, os, struct, sys, statistics as st
from robots_realtime.labeling import constants as C

# camera stem -> (label, per-frame timestamp sidecar), in player order.
#
# The sidecar is what makes label review possible: it holds one wall-clock time
# per video frame, on the SAME clock as annotations.json t_start/t_end (the mcap
# clock). Verified on episode_195935 — every camera's t0 is within 0.05 s of the
# mcap t_start, and mapping a grasp time to a frame lands within 15 ms. So
# "jump the video to this grasp" is just  seek = t_event - t0_of_that_camera.
#
#   wall clock  ──▶  1785088790.138  (grasp bag1 att1)
#                          │  minus camera t0 (1785088775.554)
#                          ▼
#   video time  ──▶  14.58 s into camera_top-images-rgb.mp4
#
# Each camera has its OWN t0 (they differ by up to ~60 ms), so switching cameras
# must re-derive the offset or the playhead drifts. See setCam() in write_html.
CAMS = [("camera_top-images-rgb.mp4", "top", "camera_top-rgb-timestamp.npy"),
        ("camera_scan-images-rgb.mp4", "scan", "camera_scan-rgb-timestamp.npy"),
        ("camera_left-images-rgb.mp4", "wrist", "camera_left-rgb-timestamp.npy")]


def npy_endpoints(path):
    """First and last float64 of a 1-D .npy, without numpy.

    This module is deliberately stdlib-only — review_watch.sh runs it with the
    system python, not the venv, so it keeps working when the venv is broken or
    absent. Mapping labels onto video only needs t0 (and t1 to clamp), so parse
    the header and read 8 bytes from each end instead of taking a dependency.

    Returns (t0, t1), or None if the file isn't the plain 1-D '<f8' we expect.
    """
    try:
        with open(path, "rb") as f:
            if f.read(6) != b"\x93NUMPY":
                return None
            major = f.read(2)[0]
            hlen = int.from_bytes(f.read(2 if major == 1 else 4), "little")
            hdr = f.read(hlen).decode("latin1")
            if "'<f8'" not in hdr or "'fortran_order': False" not in hdr:
                return None
            n = int(hdr.split("'shape': (")[1].split(",")[0])
            if n < 1:
                return None
            data0 = f.tell()
            t0 = struct.unpack("<d", f.read(8))[0]
            f.seek(data0 + (n - 1) * 8)
            return t0, struct.unpack("<d", f.read(8))[0]
    except Exception:
        return None

def load_ann(path):
    d = json.load(open(path))
    cpath = os.path.join(os.path.dirname(path), "corrections.json")
    if os.path.exists(cpath):
        try:
            corr = json.load(open(cpath))
            ga = {f"{g['bag_id']}:{g['attempt']}": g for g in d.get("grasp_attempts", [])}
            for k, ov in corr.get("grasp_attempts", {}).items():
                if k in ga: ga[k].update(ov)
            pe = {str(p['bag_id']): p for p in d.get("place_events", [])}
            for k, ov in corr.get("place_events", {}).items():
                if k in pe: pe[k].update(ov)
            for k, ov in corr.get("episode_meta", {}).items():
                d.setdefault("episode_meta", {})[k] = ov
        except Exception as e:
            print(f"  ! corrections merge failed for {path}: {e}", file=sys.stderr)
    return d

def fmt(x, n=3):
    return "n/a" if x is None else f"{x:.{n}f}"

def hbar(frac, width=24):
    frac = max(0.0, min(1.0, frac))
    f = int(round(frac * width))
    return "█" * f + "·" * (width - f)

def main():
    args = [a for a in sys.argv[1:]]
    html_out = None
    if "--html" in args:
        i = args.index("--html"); html_out = args[i+1]; del args[i:i+2]
    media_prefix = "recordings"
    if "--media-prefix" in args:
        i = args.index("--media-prefix"); media_prefix = args[i+1]; del args[i:i+2]
    root = args[0] if args else "recordings"

    comp_path = os.path.join(root, "compartments.json")
    comps = json.load(open(comp_path))["compartments"] if os.path.exists(comp_path) else []

    ann_files = sorted(glob.glob(os.path.join(root, "*", "episode_*", "annotations.json")))
    mcap_dirs = set(os.path.dirname(p) for p in glob.glob(os.path.join(root, "*", "episode_*", "yam_left.mcap")))
    ann_dirs = set(os.path.dirname(p) for p in ann_files)

    episodes = []
    for f in ann_files:
        d = load_ann(f)
        ep = os.path.basename(os.path.dirname(f))
        m = d.get("episode_meta", {})
        g = d.get("grasp_attempts", [])
        p = d.get("place_events", [])
        qpath = os.path.join(os.path.dirname(f), "qa.json")
        qa = json.load(open(qpath)) if os.path.exists(qpath) else {}
        ofpath = os.path.join(os.path.dirname(f), "operator_flags.json")
        opflags = []
        if os.path.exists(ofpath):
            try:
                opflags = [x.get("tag") for x in json.load(open(ofpath)).get("flags", [])]
            except Exception:
                opflags = []
        episodes.append(dict(ep=ep, dir=os.path.dirname(f), outcome=m.get("outcome"),
                             grasps=g, places=p, flags=d.get("flags", []),
                             segs=d.get("segments", []), kit=m.get("kitting_list", []),
                             qa=qa, opflags=opflags,
                             meta=m))   # t_start/t_end anchor the review timeline

    # ---- 1. corpus / label quality ----
    n = len(episodes)
    usable = [e for e in episodes if e["grasps"]]
    aborted = [e for e in episodes if e["outcome"] == "aborted"]
    mislabel = [e for e in episodes if e["outcome"] == "success" and not e["grasps"]]
    no_ann = sorted(mcap_dirs - ann_dirs)
    no_mcap = sorted(ann_dirs - mcap_dirs)

    # recording-health from qa.json sidecars (written by the watcher / qa_label).
    dead_gripper = [e for e in episodes if e["qa"].get("mcap_ok") and e["qa"].get("gripper_actuated") is False]
    corrupt = [e for e in episodes if e["qa"].get("mcap_ok") is False]
    true_abort = [e for e in aborted if e not in dead_gripper and e not in corrupt]

    all_grasps = [g for e in usable for g in e["grasps"]]
    all_places = [p for e in usable for p in e["places"]]
    by_outcome = {}
    for g in all_grasps:
        by_outcome[g.get("outcome", "?")] = by_outcome.get(g.get("outcome", "?"), 0) + 1
    regrasps = sum(1 for g in all_grasps if g.get("regrasp_of") is not None)

    L = []
    P = L.append
    P("=" * 72)
    P("KITTING DATA-COLLECTION REVIEW")
    P("=" * 72)
    P("")
    P("1. CORPUS & LABEL QUALITY")
    P("-" * 72)
    P(f"  labeled episodes ............ {n}")
    P(f"  usable (>=1 grasp) .......... {len(usable)}   ({len(usable)*100//max(n,1)}%)")
    P(f"  aborted (empty) ............. {len(aborted)}   (genuine operator abort: {len(true_abort)})")
    if dead_gripper or corrupt:
        P(f"  >>> DEAD GRIPPER (channel never closed, not a real abort) .. {len(dead_gripper)}  {[e['ep'] for e in dead_gripper]}")
        P(f"  >>> CORRUPT mcap (unreadable) ............................. {len(corrupt)}  {[e['ep'] for e in corrupt]}")
    P(f"  >>> MISLABEL: success w/ 0 grasp .. {len(mislabel)}  {[e['ep'] for e in mislabel]}")
    P(f"  >>> mcap dir w/ NO annotations ... {len(no_ann)}  {[os.path.basename(d) for d in no_ann]}")
    if no_mcap:
        P(f"  >>> annotations w/ NO yam mcap ... {len(no_mcap)}  {[os.path.basename(d) for d in no_mcap]}")
    P("")
    P(f"  grasp attempts (usable eps) . {len(all_grasps)}")
    for k in ("success", "slip", "drop", "empty"):
        c = by_outcome.get(k, 0)
        P(f"      {k:8} {c:4}  {hbar(c/max(len(all_grasps),1))}")
    P(f"  regrasp attempts (auto-detected) {regrasps}")
    P(f"  place events ................ {len(all_places)}")
    op_tally = {}
    for e in episodes:
        for tg in set(e["opflags"]):
            op_tally[tg] = op_tally.get(tg, 0) + 1
    if op_tally:
        P("  operator flags (g/x/s at record) " + ", ".join(f"{k}={v}" for k, v in sorted(op_tally.items())))
        # Read the shared constant rather than keeping a second copy of the rule.
        # This file already had the tag right while export_lerobot.py had it
        # wrong, and the two disagreeing is exactly what let this report name
        # episodes the exporter then went on to export anyway.
        bad_eps = [e["ep"] for e in episodes
                   if any(t in C.OPERATOR_BAD_TAGS for t in e["opflags"])]
        if bad_eps:
            P(f"      >>> operator-flagged BAD (exclude from training): {bad_eps}")
    P("")
    P("  per-episode:")
    P(f"    {'episode':30}{'outcome':9}{'grasp':6}{'place':6}{'flags':6}")
    for e in episodes:
        tag = "  << mislabel" if (e["outcome"] == "success" and not e["grasps"]) else ""
        P(f"    {e['ep']:30}{str(e['outcome']):9}{len(e['grasps']):<6}{len(e['places']):<6}{len(e['flags']):<6}{tag}")

    # ---- 2. pick-location distribution ----
    P("")
    P("2. PICK-LOCATION DISTRIBUTION  (grasp ee_pose, robot base frame, metres)")
    P("-" * 72)
    succ = [g for g in all_grasps if g.get("ee_pose") and g.get("outcome") == "success"]
    xs = [g["ee_pose"][0] for g in succ]; ys = [g["ee_pose"][1] for g in succ]; zs = [g["ee_pose"][2] for g in succ]
    if len(succ) >= 2:
        def desc(v): return f"mean {fmt(st.mean(v))}  std {fmt(st.pstdev(v))}  min {fmt(min(v))}  max {fmt(max(v))}  range {fmt(max(v)-min(v))}"
        P(f"  successful grasps: {len(succ)}")
        P(f"    x (fwd)  {desc(xs)}")
        P(f"    y (lat)  {desc(ys)}")
        P(f"    z (up)   {desc(zs)}")
        spread = math.hypot(st.pstdev(xs), st.pstdev(ys))
        P(f"    planar spread (hypot of x,y std) = {fmt(spread)} m")
    else:
        P("  <2 successful grasps with pose — not enough to distribute yet.")

    # per-bag consistency across episodes (the "same operation every time" signal)
    P("")
    P("  per-bag pick consistency across episodes  (low std = repeatable = good for ACT):")
    bag_pts = {}
    for g in succ:
        bag_pts.setdefault(g["bag_id"], []).append(g["ee_pose"])
    P(f"    {'bag':5}{'n':4}{'x std':9}{'y std':9}{'z std':9}{'spread(m)':10}")
    for bag in sorted(bag_pts):
        pts = bag_pts[bag]
        if len(pts) >= 2:
            sx = st.pstdev([p[0] for p in pts]); sy = st.pstdev([p[1] for p in pts]); sz = st.pstdev([p[2] for p in pts])
            P(f"    {bag:<5}{len(pts):<4}{fmt(sx):9}{fmt(sy):9}{fmt(sz):9}{fmt(math.hypot(sx,sy)):10}")
        else:
            P(f"    {bag:<5}{len(pts):<4}{'—':9}{'—':9}{'—':9}{'(single sample)':10}")

    # ---- 3. place accuracy ----
    P("")
    P("3. PLACE ACCURACY  vs calibrated compartments")
    P("-" * 72)
    # A release whose XY sits on top of the same bag's grasp XY is a mis-detected
    # release (gripper hysteresis fired on a transient width blip over the pick,
    # not a real placement over box 2). Separate these from true placements.
    SUSPECT_M = 0.08
    ev, clean_ev, suspect = [], [], 0
    for e in usable:
        gxy = {g["bag_id"]: g["ee_pose"][:2] for g in e["grasps"] if g.get("ee_pose")}
        for p in e["places"]:
            if p.get("in_target_region") is None:
                continue
            ev.append(p)
            rp, gp = p.get("achieved_ee_pose"), gxy.get(p.get("bag_id"))
            if rp and gp and math.hypot(rp[0]-gp[0], rp[1]-gp[1]) < SUSPECT_M:
                suspect += 1
            else:
                clean_ev.append(p)
    if ev:
        def acc(evs, lbl):
            if not evs:
                P(f"  {lbl}: none"); return
            inreg = sum(1 for p in evs if p["in_target_region"])
            offs = [p["xy_offset_m"] for p in evs if p.get("xy_offset_m") is not None]
            P(f"  {lbl:28} in-region {inreg}/{len(evs)} ({inreg*100//len(evs)}%)  "
              f"offset mean {fmt(st.mean(offs)) if offs else 'n/a'} max {fmt(max(offs)) if offs else 'n/a'} m  {hbar(inreg/len(evs))}")
        P(f"  evaluable place events ...... {len(ev)}")
        acc(ev, "raw (all)")
        acc(clean_ev, "clean (real placements)")
        P(f"  >>> SUSPECT 'released-at-pick' (release within {int(SUSPECT_M*100)}cm of grasp) .. {suspect}/{len(ev)}")
        geo = any(f.get("kind") == "geometric_targets" for e in usable for f in e["flags"])
        if suspect:
            P("  note: those are a release mis-detect (fumble at the pick). Re-label with the")
            P("        transport gate (qa_label / label_episode --min-transport 0.10) to drop them.")
        elif geo:
            P("  note: targets GEOMETRY-ASSIGNED (--geometric-targets): each placement -> nearest")
            P("        distinct compartment, so this measures placement PRECISION (how centered),")
            P("        not SAP correctness. Residual offset = real placement scatter vs 7cm cells.")
            P("        (True SAP correctness needs per-pick OCR intent + scan<->base calibration.)")
        else:
            P("  note: 0 released-at-pick (transport gate active). Calibration is OK (good placements")
            P("        center ~5cm). Residual offset is bag->compartment ASSIGNMENT (pick order != kit")
            P("        order) — re-label with --geometric-targets, or wire cockpit OCR intent.")
    else:
        P("  no evaluable place events yet.")

    # ---- 4. clarity verdict ----
    P("")
    P("4. CLARITY VERDICT")
    P("-" * 72)
    issues = []
    if dead_gripper: issues.append(f"{len(dead_gripper)} episode(s) with a DEAD GRIPPER (follower gripper never closed) — check gripper teleop/CAN before recording more.")
    if corrupt: issues.append(f"{len(corrupt)} episode(s) with a CORRUPT mcap — unrecoverable, re-record.")
    if mislabel: issues.append(f"{len(mislabel)} 'success' episode(s) with 0 grasps (re-label offline with gripper --open-ref/--closed-ref).")
    if no_ann: issues.append(f"{len(no_ann)} recorded episode(s) never labelled.")
    if ev and sum(1 for p in ev if p['in_target_region']) == 0: issues.append("0% placements in target region — check compartment calibration.")
    if len(usable) < 20: issues.append(f"only {len(usable)} usable episodes — ACT typically wants 30-50+ clean demos.")
    if issues:
        P("  NOT CLEAR — blockers before this is ACT-ready:")
        for i, s in enumerate(issues, 1): P(f"    {i}. {s}")
    else:
        P("  CLEAR — corpus looks consistent and ready to scale.")
    P("")
    P("=" * 72)

    report = "\n".join(L)
    print(report)

    if html_out:
        # cards = every labelled episode + every recorded-but-unlabelled dir, so a
        # corrupt/never-labelled episode is still reviewable by eye.
        cards = list(episodes)
        for d in no_ann:
            qpath = os.path.join(d, "qa.json")
            cards.append(dict(ep=os.path.basename(d), dir=d, outcome=None, grasps=[], places=[],
                              qa=json.load(open(qpath)) if os.path.exists(qpath) else {},
                              unlabelled=True))
        write_html(html_out, comps, succ, all_grasps, all_places, report,
                   cards, root, media_prefix)
        print(f"\nHTML report -> {html_out}")

EV_COL = {"success": "#3fb950", "slip": "#d29922", "drop": "#f85149",
          "empty": "#8b949e", "place": "#4da3ff", "flag": "#d29922"}


def episode_events(e):
    """Flatten one episode's labels into a time-ordered review list.

    This IS the review surface: each row is one thing the auto-labeler claims
    happened, at a time you can jump the video to and judge for yourself.
    `key` is the stable id a correction is filed under (see corrections.json /
    load_ann) so a verdict survives re-labelling.
    """
    evs = []
    for g in e.get("grasps") or []:
        oc = g.get("outcome") or "?"
        evs.append(dict(t=g.get("t"), col=EV_COL.get(oc, "#8b949e"),
                        key=f'grasp_attempts/{g.get("bag_id")}:{g.get("attempt")}',
                        label=f'grasp bag{g.get("bag_id")} att{g.get("attempt")}',
                        detail=oc, verdict=g.get("operator_verdict")))
    for p in e.get("places") or []:
        comp = p.get("target_compartment")
        det = p.get("detected_compartment")
        extra = "" if det is None or det == comp else f' (landed c{det})'
        inreg = p.get("in_target_region")
        evs.append(dict(t=p.get("t"),
                        col="#3fb950" if inreg else ("#f85149" if inreg is False else EV_COL["place"]),
                        key=f'place_events/{p.get("bag_id")}',
                        label=f'place bag{p.get("bag_id")} → c{comp}',
                        detail=("in region" if inreg else "OFF target" if inreg is False else "") + extra,
                        verdict=p.get("operator_verdict")))
    for f in e.get("flags") or []:
        # Flags without a time (ocr_null, geometric_targets) are episode-wide:
        # still worth showing, just not seekable.
        evs.append(dict(t=f.get("t"), col=EV_COL["flag"], key="",
                        label=f'⚑ {f.get("kind")}', detail=(f.get("detail") or "")[:90]))
    evs.sort(key=lambda x: (x["t"] is None, x["t"] or 0))
    return evs


def newest_first(cards):
    """Sort episodes newest first — dir is <root>/<YYYYMMDD>/episode_<HHMMSS>_<hash>."""
    return sorted(cards, key=lambda e: (os.path.basename(os.path.dirname(e["dir"])), e["ep"]),
                  reverse=True)


def episode_block(e, root, media_prefix):
    """The full review view for ONE episode: player, camera switcher, timeline,
    badges, and the seekable event rows with verdict buttons.

    Used by the per-episode page. The index deliberately does NOT embed this —
    sixteen preloaded players on one page is why the old grid was heavy, and the
    index's 10s auto-refresh would interrupt a review in progress.
    """
    out = []
    for e in [e]:
        qa, ep = e.get("qa") or {}, e["ep"]
        rel = os.path.relpath(e["dir"], root).replace(os.sep, "/")
        vids, cam_t0 = [], {}
        for stem, lbl, tsname in CAMS:
            if not os.path.exists(os.path.join(e["dir"], stem)):
                continue
            vids.append((f"{media_prefix}/{rel}/{stem}", lbl))
            ends = npy_endpoints(os.path.join(e["dir"], tsname))
            if ends:
                cam_t0[lbl] = ends[0]

        badges = []
        def badge(txt, col):
            badges.append(f'<span style="background:{col}22;color:{col};border:1px solid {col}55;'
                          f'border-radius:6px;padding:2px 8px;font-size:11.5px;white-space:nowrap">{txt}</span>')
        if qa.get("mcap_ok") is False:
            badge("CORRUPT mcap", "#f85149")
        elif qa.get("gripper_actuated") is False:
            badge("DEAD GRIPPER", "#f85149")
        if e.get("unlabelled"):
            badge("not labelled", "#d29922")
        if e["outcome"]:
            badge(e["outcome"], "#3fb950" if e["outcome"] == "success" else "#8b949e")
        if not e.get("unlabelled"):
            badge(f'{len(e["grasps"])} grasp', "#4da3ff")
            badge(f'{len(e["places"])} place', "#4da3ff")
        if qa.get("gripper_min_norm") is not None:
            badge(f'grip min {qa["gripper_min_norm"]:.3f}', "#8b949e")

        vid_id = "v_" + ep
        if vids:
            src0 = _html.escape(vids[0][0])
            player = (f'<video id="{vid_id}" src="{src0}" controls preload="none" playsinline '
                      f'style="width:100%;border-radius:8px;background:#000;aspect-ratio:16/9"></video>')
            if len(vids) > 1:
                btns = "".join(
                    f'<button onclick="setCam(\'{vid_id}\',\'{_html.escape(src)}\',this,\'{lbl}\')" '
                    f'style="background:#161b22;color:#9aa7b4;border:1px solid #30363d;border-radius:6px;'
                    f'padding:3px 10px;font-size:11.5px;cursor:pointer;margin-right:6px">{lbl}</button>'
                    for src, lbl in vids)
                player += f'<div style="margin-top:8px">{btns}</div>'
        else:
            player = ('<div style="width:100%;aspect-ratio:16/9;border-radius:8px;background:#0b0f16;'
                      'border:1px dashed #30363d;display:flex;align-items:center;justify-content:center;'
                      'color:#6e7681;font-size:12px">no video in this episode dir</div>')

        # ── label review: timeline + seekable event rows ────────────────────
        # Only meaningful when we know a camera's t0; without the sidecar the
        # rows still render (so you can read the labels) but cannot seek.
        evs = episode_events(e)
        meta = e.get("meta") or {}
        t_start, t_end = meta.get("t_start"), meta.get("t_end")
        base = cam_t0.get(vids[0][1]) if vids else None
        timeline = events_html = ""
        if evs and t_start and t_end and t_end > t_start:
            dur = t_end - t_start
            marks = "".join(
                f'<span title="{_html.escape(v["label"])}" style="position:absolute;'
                f'left:{max(0.0, min(100.0, (v["t"]-t_start)/dur*100)):.2f}%;top:0;bottom:0;width:2px;'
                f'background:{v["col"]}"></span>'
                for v in evs if v["t"] is not None)
            # Playhead + click-to-seek. Without a moving cursor the markers are
            # unfalsifiable — you cannot tell whether a tick sits where the event
            # actually happens. The cursor is what makes the timeline checkable.
            timeline = (
                f'<div id="tl_{vid_id}" onclick="tlSeek(\'{vid_id}\',event)" '
                f'style="position:relative;height:18px;margin-top:10px;background:#0d1117;'
                f'border:1px solid #242c37;border-radius:4px;overflow:hidden;cursor:pointer">'
                f'{marks}'
                f'<span id="cur_{vid_id}" style="position:absolute;left:0;top:-2px;bottom:-2px;'
                f'width:2px;background:#e6edf3;box-shadow:0 0 4px #e6edf3;pointer-events:none"></span>'
                f'</div>'
                f'<div style="display:flex;justify-content:space-between;font-family:monospace;'
                f'font-size:10px;color:#6e7681;margin-top:3px">'
                f'<span>0s</span>'
                f'<span id="tlab_{vid_id}" style="color:#e6edf3">0.0s</span>'
                f'<span>{dur:.0f}s</span></div>')
        if evs:
            rows = []
            for v in evs:
                seek = "" if (v["t"] is None or base is None) else f'{v["t"] - base:.2f}'
                click = (f' onclick="seekTo(\'{vid_id}\',{seek})" style="cursor:pointer"'
                         if seek else ' style="opacity:.75"')
                tlab = f'{v["t"] - t_start:6.1f}s' if (v["t"] and t_start) else '   --  '
                btns = ""
                if v["key"]:
                    # A saved verdict must come back lit after review_watch.sh
                    # regenerates the page, or corrections look like they vanished.
                    vd = v.get("verdict")
                    btns = (f'<button onclick="event.stopPropagation();mark(this,\'{_html.escape(rel)}\','
                            f'\'{v["key"]}\',\'ok\')" class="vb{" on" if vd == "ok" else ""}" '
                            f'title="label is correct">✓</button>'
                            f'<button onclick="event.stopPropagation();mark(this,\'{_html.escape(rel)}\','
                            f'\'{v["key"]}\',\'wrong\')" class="vb{" on" if vd == "wrong" else ""}" '
                            f'title="label is WRONG">✗</button>')
                rows.append(
                    f'<div class=evrow{click}>'
                    f'<span style="color:{v["col"]}">▸</span>'
                    f'<span style="color:#6e7681">{tlab}</span>'
                    f'<span style="color:#cdd6e0;flex:1">{_html.escape(v["label"])}</span>'
                    f'<span style="color:#6e7681;font-size:10.5px">{_html.escape(v["detail"])}</span>'
                    f'{btns}</div>')
            note = ("" if base is not None else
                    '<div style="color:#d29922;font-size:10.5px;margin-top:4px">'
                    'no frame-timestamp sidecar — rows are not seekable</div>')
            events_html = (f'<div style="margin-top:10px;font-family:monospace;font-size:12.5px">'
                           f'{"".join(rows)}</div>{note}')

        camjs = json.dumps(cam_t0)
        out.append(
            f'<div style="background:#0b0f16;border:1px solid #242c37;border-radius:12px;padding:16px">'
            f'<div style="font-family:monospace;font-size:11px;color:#6e7681;margin-bottom:10px">{_html.escape(rel)}</div>'
            f'{player}'
            f'<script>CAMT["{vid_id}"]={{cur:"{vids[0][1] if vids else ""}",t0:{camjs},'
            f't_start:{json.dumps(t_start)},dur:{json.dumps((t_end - t_start) if (t_start and t_end and t_end > t_start) else None)}}};</script>'
            f'{timeline}'
            f'<div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:12px">{"".join(badges)}</div>'
            f'{events_html}'
            f'</div>')
    return "".join(out)


def index_rows(cards, root):
    """The landing page: one compact row per episode, linking to its review page.

    No <video> here on purpose. Sixteen players on one page made the index heavy,
    and the index auto-refreshes every 10s — which would yank you out of a review
    you were in the middle of. Reviewing happens on the episode's own page.
    """
    rows = []
    for e in newest_first(cards):
        qa, ep = e.get("qa") or {}, e["ep"]
        rel = os.path.relpath(e["dir"], root).replace(os.sep, "/")
        meta = e.get("meta") or {}
        t0, t1 = meta.get("t_start"), meta.get("t_end")
        dur = f"{t1 - t0:.0f}s" if (t0 and t1 and t1 > t0) else "--"
        evs = episode_events(e)
        judged = sum(1 for v in evs if v["key"] and v.get("verdict"))
        total = sum(1 for v in evs if v["key"])

        if qa.get("mcap_ok") is False:
            state, col = "CORRUPT mcap", "#f85149"
        elif qa.get("gripper_actuated") is False:
            state, col = "DEAD GRIPPER", "#f85149"
        elif e.get("unlabelled"):
            state, col = "not labelled", "#d29922"
        elif not e.get("grasps"):
            state, col = "no grasps", "#d29922"
        else:
            state, col = "usable", "#3fb950"

        prog = (f'<span style="color:{"#3fb950" if judged == total and total else "#6e7681"}">'
                f'{judged}/{total} reviewed</span>' if total else
                '<span style="color:#6e7681">—</span>')
        rows.append(
            f'<a class=eprow href="ep/{_html.escape(ep)}.html">'
            f'<span style="color:{col};width:110px;flex:none">● {state}</span>'
            f'<span style="color:#e6edf3;flex:1">{ep}</span>'
            f'<span style="color:#6e7681;width:60px;text-align:right">{dur}</span>'
            f'<span style="color:#4da3ff;width:150px;text-align:right">'
            f'{len(e.get("grasps") or [])} grasp · {len(e.get("places") or [])} place</span>'
            f'<span style="width:110px;text-align:right">{prog}</span>'
            f'<span style="color:#6e7681;width:70px;text-align:right">review →</span>'
            f'</a>')
    return (f'<div style="font-family:monospace;font-size:12.5px;border:1px solid #242c37;'
            f'border-radius:12px;overflow:hidden">{"".join(rows)}</div>')


def write_episode_pages(out_dir, cards, root, media_prefix):
    """One standalone page per episode, at <out_dir>/ep/<episode_id>.html.

    Separate files rather than a modal so the browser back button works, the URL
    is shareable, and — the real reason — the index's auto-refresh cannot reload
    the page out from under a review in progress.
    """
    ep_dir = os.path.join(out_dir, "ep")
    os.makedirs(ep_dir, exist_ok=True)
    # Pages live one level deeper, so the recordings symlink is one level up.
    deep_prefix = "../" + media_prefix
    written = []
    for e in cards:
        ep = e["ep"]
        body = episode_block(e, root, deep_prefix)
        html = (f'<!doctype html><meta charset="utf-8"><title>{_html.escape(ep)} — review</title>'
                f'{_style()}<script>var CAMT={{}};</script>'
                f'<body style="background:#0d1117;color:#cdd6e0;font-family:system-ui;'
                f'max-width:1100px;margin:24px auto;padding:0 16px">'
                f'<a href="../index.html" style="color:#4da3ff;text-decoration:none;font-size:13px">'
                f'← all episodes</a>'
                f'<h1 style="color:#e6edf3;font-size:20px;font-family:monospace;margin:10px 0 4px">'
                f'{_html.escape(ep)}</h1>'
                f'<div style="color:#6e7681;font-size:12px;margin-bottom:16px">'
                f'Click an event row to jump the video there. ✓ / ✗ records whether the auto-label '
                f'is right — saved to corrections.json immediately.</div>'
                f'{body}'
                f'<div style="margin:20px 0"><a href="../index.html" style="color:#4da3ff;'
                f'text-decoration:none;font-size:13px">← all episodes</a></div>'
                # No auto-reload on this page: you are working in it.
                f'{_script(auto_reload=False)}</body>')
        with open(os.path.join(ep_dir, f"{ep}.html"), "w") as f:
            f.write(html)
        written.append(ep)
    return written


def _script(auto_reload: bool):
    """Shared page behaviour. auto_reload is for the INDEX only — an episode page
    must never reload itself out from under a review in progress."""
    js = ('<script>'
          # Each camera starts at its own wall-clock t0 (they differ by ~60ms), so a
          # raw currentTime carry-over drifts on switch. Convert the playhead to
          # wall time, switch, convert back.
          'function setCam(id,src,btn,lbl){var v=document.getElementById(id);'
          'var c=CAMT[id]||{cur:"",t0:{}};var wall=v.currentTime+(c.t0[c.cur]||0);'
          'var playing=!v.paused;v.src=src;v.load();c.cur=lbl;'
          'var nt=c.t0[lbl];v.currentTime=(nt==null?0:Math.max(0,wall-nt));'
          'if(playing)v.play();'
          'var bs=btn.parentNode.children;for(var i=0;i<bs.length;i++){bs[i].style.color="#9aa7b4";'
          'bs[i].style.borderColor="#30363d";}btn.style.color="#4da3ff";btn.style.borderColor="#4da3ff";}'
          # Rows carry an offset computed against the FIRST camera; re-derive it
          # against whichever camera is showing now.
          'function seekTo(id,off){var v=document.getElementById(id);'
          'var c=CAMT[id]||{cur:"",t0:{}};var first=null;for(var k in c.t0){first=c.t0[k];break;}'
          'var wall=(first==null?0:first)+off;var nt=c.t0[c.cur];'
          'v.currentTime=(nt==null?off:Math.max(0,wall-nt));v.pause();}'
          # Verdicts POST straight to disk. Never keep them only in the DOM:
          # review_watch.sh regenerates these pages on every saved episode.
          'function mark(btn,dir,key,verdict){'
          'btn.disabled=true;btn.textContent="…";'
          'fetch("/corrections",{method:"POST",headers:{"Content-Type":"application/json"},'
          'body:JSON.stringify({dir:dir,key:key,verdict:verdict})})'
          '.then(function(r){return r.json();}).then(function(j){'
          'btn.disabled=false;if(!j.ok){btn.textContent="!";btn.title=j.error||"failed";return;}'
          'var sib=btn.parentNode.querySelectorAll("button.vb");'
          'for(var i=0;i<sib.length;i++){sib[i].classList.remove("on");}'
          'btn.classList.add("on");btn.textContent=(verdict=="ok"?"✓":"✗");'
          '}).catch(function(){btn.disabled=false;btn.textContent="!";btn.title="server unreachable";});}'
          # Episode-relative position of the playhead, 0..1. Goes through WALL
          # time, not raw currentTime, because each camera starts at its own t0 —
          # otherwise the cursor jumps when you switch angle.
          'function frac(c,v){if(c.t_start==null||!c.dur)return null;'
          'var t0=c.t0[c.cur];if(t0==null)return null;'
          'return((v.currentTime+t0)-c.t_start)/c.dur;}'
          # Click anywhere on the bar to seek there.
          'function tlSeek(id,ev){var c=CAMT[id],v=document.getElementById(id);'
          'if(!c||!v||c.t_start==null||!c.dur)return;var t0=c.t0[c.cur];if(t0==null)return;'
          'var r=ev.currentTarget.getBoundingClientRect();'
          'var f=Math.max(0,Math.min(1,(ev.clientX-r.left)/r.width));'
          'v.currentTime=Math.max(0,(c.t_start+f*c.dur)-t0);}'
          # rAF, not the video "timeupdate" event: timeupdate fires ~4x/sec and the
          # cursor visibly stutters, which is useless for judging whether a marker
          # lines up with what you are watching.
          'function tick(){for(var id in CAMT){var v=document.getElementById(id);if(!v)continue;'
          'var f=frac(CAMT[id],v);if(f==null)continue;'
          'var cur=document.getElementById("cur_"+id);'
          'if(cur)cur.style.left=(Math.max(0,Math.min(1,f))*100)+"%";'
          'var lab=document.getElementById("tlab_"+id);'
          'if(lab)lab.textContent=(f*CAMT[id].dur).toFixed(1)+"s";}'
          'requestAnimationFrame(tick);}requestAnimationFrame(tick);')
    if auto_reload:
        js += ('setInterval(function(){'
               'if(document.querySelector("button.vb[disabled]"))return;'
               'var vs=document.getElementsByTagName("video");'
               'for(var i=0;i<vs.length;i++){if(!vs[i].paused||vs[i].currentTime>0)return;}'
               'location.reload();},10000);')
    return js + '</script>'


def _style():
    return ('<style>'
            '.evrow{display:flex;gap:10px;align-items:center;padding:5px 6px;border-radius:4px}'
            '.evrow:hover{background:#161b22}'
            'button.vb{background:#161b22;color:#6e7681;border:1px solid #30363d;border-radius:4px;'
            'width:24px;height:22px;font-size:12px;cursor:pointer;padding:0;line-height:1}'
            'button.vb:hover{color:#cdd6e0}'
            'button.vb.on{background:#1f6feb33;color:#4da3ff;border-color:#4da3ff}'
            '.eprow{display:flex;gap:12px;align-items:center;padding:9px 14px;text-decoration:none;'
            'border-bottom:1px solid #1c232c}'
            '.eprow:last-child{border-bottom:none}'
            '.eprow:hover{background:#161b22}'
            '</style>')


def write_html(path, comps, succ, all_grasps, all_places, report,
               cards=(), root="recordings", media_prefix="recordings"):
    # top-down: robot base frame, x forward (up on screen), y lateral (left on screen).
    pts = [(g["ee_pose"][0], g["ee_pose"][1], g.get("outcome")) for g in all_grasps if g.get("ee_pose")]
    plc = [(p["achieved_ee_pose"][0], p["achieved_ee_pose"][1], p.get("in_target_region"))
           for p in all_places if p.get("achieved_ee_pose")]
    allx = [x for x, y, *_ in pts+plc] + [c["x_max"] for c in comps] + [c["x_min"] for c in comps]
    ally = [y for x, y, *_ in pts+plc] + [c["y_max"] for c in comps] + [c["y_min"] for c in comps]
    if not allx: allx=[0,0.3]; ally=[-0.5,0]
    pad=0.03
    x0,x1=min(allx)-pad,max(allx)+pad; y0,y1=min(ally)-pad,max(ally)+pad
    W,H=760,520
    def sx(y): return (y1 - y)/(y1-y0)*W          # y lateral -> screen x (flip so +y is left)
    def sy(x): return (x1 - x)/(x1-x0)*H          # x forward -> screen y (up = forward)
    col={"success":"#3fb950","slip":"#d29922","drop":"#f85149","empty":"#8b949e",None:"#8b949e"}
    svg=[f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="background:#0b0f16;border:1px solid #242c37;border-radius:12px">']
    for c in comps:
        rx,ry=sx(c["y_max"]),sy(c["x_max"]); rw=sx(c["y_min"])-sx(c["y_max"]); rh=sy(c["x_min"])-sy(c["x_max"])
        svg.append(f'<rect x="{rx:.1f}" y="{ry:.1f}" width="{rw:.1f}" height="{rh:.1f}" fill="none" stroke="#4da3ff" stroke-width="1.2"/>')
        svg.append(f'<text x="{rx+rw/2:.1f}" y="{ry+rh/2:.1f}" fill="#4da3ff" font-size="13" text-anchor="middle" font-family="monospace">{c["id"]}</text>')
    for x,y,inr in plc:  # place points = hollow squares
        c = "#3fb950" if inr else "#f85149"
        svg.append(f'<rect x="{sx(y)-4:.1f}" y="{sy(x)-4:.1f}" width="8" height="8" fill="none" stroke="{c}" stroke-width="1.5"/>')
    for x,y,o in pts:    # pick points = filled dots
        svg.append(f'<circle cx="{sx(y):.1f}" cy="{sy(x):.1f}" r="4.5" fill="{col.get(o,"#8b949e")}" opacity="0.85"/>')
    svg.append('</svg>')
    legend = ('<div style="font-family:monospace;font-size:12px;color:#9aa7b4;margin:10px 0">'
              '● pick: <span style="color:#3fb950">success</span> '
              '<span style="color:#d29922">slip</span> '
              '<span style="color:#f85149">drop</span> '
              '<span style="color:#8b949e">empty</span> &nbsp;|&nbsp; '
              '▢ place: <span style="color:#3fb950">in-region</span> '
              '<span style="color:#f85149">off</span> &nbsp;|&nbsp; '
              '<span style="color:#4da3ff">▭ compartments 1–7</span> &nbsp;·&nbsp; up = +x (forward), left = +y (lateral)</div>')
    script = _script(auto_reload=True)
    ep_written = write_episode_pages(os.path.dirname(os.path.abspath(path)) or ".",
                                     newest_first(cards), root, media_prefix)
    html=(f'<!doctype html><meta charset="utf-8"><title>Kitting data review</title>{_style()}'
          f'<script>var CAMT={{}};</script>'
          f'<body style="background:#0d1117;color:#cdd6e0;font-family:system-ui;max-width:1200px;margin:24px auto;padding:0 16px">'
          f'<h1 style="color:#e6edf3">Kitting data-collection review</h1>'
          f'<div style="color:#6e7681;font-size:12px;margin:-8px 0 16px">'
          f'Click an episode to open its review page: video, label timeline, and '
          f'✓ / ✗ on every auto-label.</div>'
          f'<h2 style="color:#e6edf3;font-size:16px">Episodes ({len(cards)}) — newest first</h2>'
          f'{index_rows(cards, root)}'
          f'<h2 style="color:#e6edf3;font-size:16px;margin-top:32px">Top-down pick / place distribution</h2>'
          f'{"".join(svg)}{legend}'
          f'<h2 style="color:#e6edf3;font-size:16px">Full report</h2>'
          f'<pre style="background:#0b0f16;border:1px solid #242c37;border-radius:12px;padding:16px;overflow-x:auto;font-size:12.5px;line-height:1.5">{_html.escape(report)}</pre>'
          f'{script}</body>')
    open(path,"w").write(html)

if __name__ == "__main__":
    main()
