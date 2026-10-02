"""Tuning constants for auto-labeling.

Gripper units differ across rigs and are not known a-priori (the recorded value
is a raw motor joint position, index 6 in the 7-DoF joint vector). So the
segmentation logic NEVER hardcodes an absolute width. It normalizes the gripper
signal per-episode to [0, 1] where 0 = fully closed, 1 = fully open, then applies
these fractional thresholds. This keeps the labeler unit-agnostic and robust to
open/closed calibration drift.
"""
from __future__ import annotations

# --- gripper segmentation (all fractions of the per-episode open→closed range) -
# Hysteresis (two thresholds) so sensor jitter near the boundary can't emit a
# storm of open/close events. Enter "closing" below CLOSE_ENTER, leave it above
# CLOSE_EXIT.
GRIPPER_CLOSE_ENTER = 0.45   # below this normalized width → gripper is closing
GRIPPER_CLOSE_EXIT = 0.60    # above this → gripper is opening/open
# A grasp must actually hold something: the closed width has to sit ABOVE this
# (i.e. not fully closed), otherwise the gripper closed on empty air.
GRIPPER_EMPTY_CLOSE = 0.08   # closed to < 8% of range = closed on nothing
# Slip/drop: while holding a bag, if the width collapses by more than this
# fraction below the grasp-hold width (toward fully closed), the bag is gone.
GRIPPER_SLIP_DROP = 0.15

# --- gripper normalisation ----------------------------------------------------
# When no open_ref/closed_ref are given, segmentation.normalize_width falls back
# to episode percentiles. That fallback is only meaningful if the episode really
# spans open→closed, so it needs a floor below which it refuses to answer.
#
# The floor is RELATIVE to the signal's own magnitude, not an epsilon. Measured
# 2026-08-08 over the 29 readable episodes in recordings/ (yam_left.mcap, col 6):
#
#     live gripper   (n=17)   p98-p2 spread 0.9960 .. 0.9986
#     dead gripper   (n=12)   p98-p2 spread 0.0000 .. 0.0002   (resting ~0.96-0.999)
#
# The old guard tested `hi - lo < 1e-9`, which none of the twelve dead episodes
# satisfied — sensor noise alone is ~1e-4 — so the fallback rescaled that noise
# into a full-scale open/close trace. Any threshold in [1e-3, 0.9] separates the
# two populations; 0.05 sits far from both edges.
#
# ASSUMPTION, stated because it is the one way this can be wrong: the raw
# gripper value has no large constant offset (this rig records ~0.002 closed /
# ~0.999 open; the other calibration seen in the wild is -0.0235 / 5.2218). A rig
# that reported, say, 1000.0 closed / 1005.0 open would trip this floor on a
# healthy episode. Pass open_ref/closed_ref and none of this applies.
GRIPPER_MIN_RANGE_FRAC = 0.05

# --- operator flags -----------------------------------------------------------
# The tag written into operator_flags.json when the operator rejects a take
# live. It is "bad": tui.py:259 maps the KEY "x" to the TAG "bad",
# session.py:389 writes {"tag": "bad"}, and control_server.py rejects a literal
# "x" with a 400. export_lerobot.py used to filter on "x", so from the day the
# flag was added until 2026-08-08 the filter could not match anything and every
# take the operator rejected was exported as training data. (AUDIT.md S7.1)
#
# A set, not a string, so the exporter and review_corpus.py cannot drift apart
# again, and so a hand-edited older file that literally says "x" still counts.
OPERATOR_BAD_TAGS = frozenset({"bad", "x"})

# A grasp/release must persist at least this long to count (debounce brief
# adjustment twitches). Seconds.
#
# Raised 0.20 → 0.50 on 2026-07-27, measured against the whole recorded corpus
# (23 episodes, 192 grip intervals) with tools/diagnose_live_gate.py:
#
#     hold duration    n     p50      p75      p90
#     ─────────────────────────────────────────────
#     outcome=success  65    4.445s   6.999s   9.064s
#     outcome=empty   124    0.300s   0.427s   1.169s
#
# Real picks and twitches are separated by an order of magnitude, and the old
# 0.20 s let 124 sub-second open/close twitches into the event stream — they
# swamped the cockpit's event log and each one reset the grasp state machine
# mid-approach. Sweep of the cutoff:
#
#     0.3s → removes 62/124 twitches, loses 1/65 real grasps
#     0.5s → removes 98/124 twitches, loses 1/65 real grasps   ← chosen
#     0.6s → removes 101/124,         loses 3/65
#     0.8s → removes 106/124,         loses 7/65
#
# 0.5 s is the knee: 79 % of the noise for the same single lost grasp (that one
# is a 0.275 s pick, which is faster than a human teleop grasp can physically be
# and is itself probably a mislabel). Re-run the sweep before changing this.
MIN_HOLD_S = 0.50
# A grasp must be followed by a lift within this window to be a real grasp
# (a close with no lift is an adjustment, not a pick). Seconds.
LIFT_WINDOW_S = 2.0
# Minimum end-effector rise (metres, robot Z-up) to count as a "lift".
MIN_LIFT_M = 0.03
# Minimum horizontal (XY) distance the EE must travel between grasp and release
# for a grasp to count as a real pick-AND-place. A success grasp that closes and
# re-opens near the pick without transporting (a re-grip / fumble at the box) is
# NOT a placement — gating on this removes the "released-at-pick" false placements.
# 0.0 disables the gate (library default, back-compat); the real kitting pipeline
# passes ~0.10 (box1→box2 is always >=~0.15m apart).
MIN_TRANSPORT_M = 0.10

# --- placement ---------------------------------------------------------------
# A release counts as "in" a compartment if the end-effector's XY at release is
# within the compartment polygon expanded by this margin (metres). Small slop
# for the offset between the wrist link (link_6, what FK gives) and the actual
# release point.
IN_REGION_MARGIN_M = 0.02

# --- fusion / clock ----------------------------------------------------------
# A cockpit event whose timestamp is more than this far outside the episode's
# [start, end] window is rejected (clock desync guard). Seconds.
CLOCK_WINDOW_SLACK_S = 1.0

# --- FK ----------------------------------------------------------------------
# The arm has 6 revolute joints (joint1..joint6); the 7th recorded value is the
# gripper. FK is computed to this link. The real grasp point is offset from the
# wrist by the gripper length; callers may add a tool offset.
FK_EE_LINK = "link_6"
N_ARM_JOINTS = 6
GRIPPER_JOINT_INDEX = 6

# --- grasp workspace ---------------------------------------------------------
# Grasps whose end-effector x is below this are outside the packet mat and are
# not training data. Measured 2026-07-28 over the 81 successful grasps in the
# corpus: sorted x has a 176 mm empty band from 0.178 to 0.354, with 4 grasps
# below it. Those 4 also sit at z 0.162-0.226 against a corpus mean of 0.120 --
# near AND high, i.e. mid-air or mislabelled, not table grasps. They come from
# two different episodes (195935, 212500), so this is not one bad session.
#
# A fixed gate, not a 3-sigma rule: a statistical cut moves as the corpus grows,
# so the same recording would be in one week and out the next. Any threshold in
# [0.20, 0.30] drops exactly these 4; 0.25 sits mid-gap. Metres, robot base frame.
GRASP_WORKSPACE_X_MIN = 0.25

# Lateral gate. None = OFF, which is the default and reproduces every dataset
# exported before 2026-08-03. It exists because the x gate above cannot express
# "the far corner of the mat":
#
#     base -y  ................................................ base +y
#     -0.38                      -0.24                     -0.085
#       |  main packet layout (n=75, 7% fail)  |  corner (n=14, 21% fail)  |
#                                           y_max
#
# Measured over the 89 labelled attempts: grasps at y > -0.13 fail three times
# as often as the rest. n=14 with 3 failures, so this is a direction, NOT a
# proven threshold -- which is exactly why it defaults to None and has to be
# passed explicitly by someone who has looked at tools/review_grasps.py.
#
# NOT a claim about the tray. The tray spans y -0.487..-0.087 (results/
# tray_box.json), so it underlies BOTH bands; the corner is not "over the box".
GRASP_ZONE_Y_MAX = None
