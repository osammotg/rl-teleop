#!/usr/bin/env bash
# Bring up the RIGHT-ARM RECORDING pipeline, verified, in one command.
# (Ported from start_left_recording.sh 2026-09-02; right configs/CAN/handle.)
#
#   ./start_right_recording.sh
#
# Every check below exists because it silently failed on 2026-09-01 and cost
# real recording time. They are GATES, not advice: this script refuses to hand
# you a session that looks alive and records nothing.
#
#   0. hardware preflight after a power-on the cheapest failures are physical:
#                        CAN adapter not enumerated, camera unplugged, leader
#                        serial gone, leader handle unpowered. Checked first,
#                        before anything starts.
#                        PREFLIGHT_ONLY=1 runs just this stage and exits.
#                        The leader-chain ping is the only part that opens a
#                        device, and only when no session holds the serial
#                        (:8792/:8794 down) -- see stage -0.5.
#                        The CAN *traffic* gate runs after the session is up --
#                        with no poller, a healthy quiet bus and the 2026-08-31
#                        up-but-dead bus both read 0 rx, so it cannot be
#                        judged earlier.
#   1. control port      the cockpit hardcodes :8792 (cockpit-control.js
#                        DEFAULT_BASE). Overriding CONTROL_PORT makes the REC
#                        button post into a void -- it only prints "kein
#                        rr-session auf ..." in a status line. Five grasps were
#                        lost to this. We never override it and we verify it.
#   2. camera bridge     :8791 holds subscriptions that die with the session and
#                        never reconnect (/cam/top starts 503ing). So it is
#                        ALWAYS restarted after the session, never before.
#   3. gripper channel   a flat gripper channel cannot train a grasp policy --
#                        the exporter finds grasp windows BY the gripper closing.
#                        A leader gripper read 2290 with zero travel on 2026-09-01
#                        and 11 minutes of otherwise-perfect video was useless.
#                        (2026-09-02 correction: that flat 2290 -- and the 1966
#                        repeat -- was the WRONG PHYSICAL HANDLE being squeezed,
#                        not a broken servo. The udev names are crossed; see
#                        runbook/leader-handle-identity-crossed.md. Stage -0.5
#                        now tells the two apart in 2 s.)
#                        This script still makes you prove the gripper moves
#                        before it lets you record -- but ONCE, not daily:
#                        stage 0a verifies the attestation in
#                        configs/leader_identity.json (FTDI serial + servo
#                        fingerprint + device enumeration time + the config's
#                        leader->arm mapping) in ~2 s and skips the 45 s
#                        squeeze. With that file ABSENT nothing changes.
#                        Stage 5b then watches the recorded gripper CHANNEL
#                        during the session and warns if it goes flat.
#   4. SSH forwarding    the operator's browser is NOT on this box. A local curl
#                        proving a port healthy proves nothing about what he can
#                        reach. We print the exact ports he must forward.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

CONFIG="${CONFIG:-configs/yam/yam_right_grasp_teleop_noscan.yaml}"
PY=./.venv/bin/python3
COCKPIT_PAGE="Buehler-Kitting-Recording.html"
SKIP_GRIPPER="${SKIP_GRIPPER:-0}"

red()  { printf '\033[31m%s\033[0m\n' "$*"; }
grn()  { printf '\033[32m%s\033[0m\n' "$*"; }
ylw()  { printf '\033[33m%s\033[0m\n' "$*"; }
die()  { red "✗ $*"; exit 1; }
port_up() { ss -tlnH 2>/dev/null | grep -q ":$1 "; }
pids_on() { ss -tlnpH 2>/dev/null | grep ":$1 " | grep -oP 'pid=\K[0-9]+' | sort -u; }

echo "── RIGHT-arm recording bring-up ─────────────────────────────────────"

# ── -1. rig hardware preflight — is everything actually plugged in? ─────────
# Presence-only checks: open no device, send nothing on any bus, safe to run with
# sessions up. The one exception is stage -0.5 (leader chain ping), which opens
# the leader serial and therefore self-skips whenever :8792/:8794 is listening.
# PREFLIGHT_ONLY=1 stops after this stage.
GPORT=$(grep -A20 'name: gello_right' "$CONFIG" | grep -oP 'port:\s*\K\S+' | head -1)
GIDS=$(grep -A20 'name: gello_right' "$CONFIG" | grep -oP 'motor_ids:\s*\[\K[^\]]+' | head -1 | tr -d ' ')
GID=$(printf '%s' "$GIDS" | awk -F, '{print $NF}')
NGID=$(printf '%s' "$GIDS" | awk -F, '{print NF}')
[ -n "$GPORT" ] && [ -n "$GID" ] || die "could not read gello_right port/motor_ids from $CONFIG"

[ -e "$GPORT" ] || die "leader serial $GPORT is missing -- handle unplugged or udev symlink gone.
     ls -l /dev/serial/by-id/ to see what actually enumerated."
grn "  ✓ leader serial $GPORT present"

[ -d /sys/class/net/can_follow_r ] || die "can_follow_r does not exist -- RIGHT CAN adapter not enumerated.
     Physically replug it (usb9, port 9-1.3); software resets do not recover a
     re-enumerated gs_usb adapter (2026-08-31)."
[ "$(cat /sys/class/net/can_follow_r/operstate)" = "up" ] || \
  die "can_follow_r is DOWN -- run in a real terminal:  sudo rig-setup"
grn "  ✓ can_follow_r exists and is UP (traffic gate comes after session start)"

$PY tools/preflight_cameras.py "$CONFIG" | sed 's/^/    /' || \
  die "a configured camera is missing -- fix it physically before anything starts.
     A camera that enumerates but won't stream can be power-cycled in software:
     echo 0 > /sys/bus/usb/devices/<dev>/authorized; sleep 3; echo 1 > .../authorized"
grn "  ✓ cameras present"

# ── -0.5 leader chain alive — are all 7 leader servos powered and answering? ─
# 2026-09-02: five consecutive bring-ups each burned the full 45 s gripper gate
# and died "GRIPPER IS FLAT" (stuck at 2290, then 1966). Nothing was broken: the
# operator was squeezing the OTHER physical handle, because the udev names are
# CROSSED relative to physical position — /dev/leader-right is wired to the
# PHYSICALLY LEFT handle (verified by servo wiggle, tools/identify_leader_handle.py).
# Meanwhile the handle on /dev/leader-left had 0 of 7 servos answering at all:
# electrically dead, an unpowered 5 V/12 V brick.
# A 2 s ping scan separates those two failures instantly, so nobody ever again
# spends 45 s squeezing a handle that cannot possibly answer.
#
# This is the only preflight check that OPENS the serial device, so it runs only
# when no session holds it — the same port_up 8792 guard the gripper gate uses,
# plus 8794 (the right-arm teleop session owns the other leader).
if port_up 8792 || port_up 8794; then
  ylw "  ⚠ a session is up (:8792/:8794) and owns a leader serial — skipping the chain ping"
  ylw "    Note: DynamixelGelloLeaderAgent REPLAYS its last reading when its device"
  ylw "    dies, so a running session can publish perfectly plausible joint_pos from"
  ylw "    a handle that is unplugged. Stop the session before trusting any of it."
else
  echo "  pinging leader chain on $GPORT (ids $GIDS) ..."
  $PY - "$GPORT" "$GIDS" <<'PYEOF'
import sys, time
from dynamixel_sdk import PortHandler, PacketHandler

port = sys.argv[1]
ids = [int(x) for x in sys.argv[2].split(",") if x.strip()]
ph = PortHandler(port)
pk = PacketHandler(2.0)
if not ph.openPort():
    print("   cannot open %s" % port)
    sys.exit(4)
ph.setBaudRate(1000000)

alive, dead, models = [], [], {}
# Every id gets at least one ping; the deadline only cuts the RETRIES short, so a
# slow bus can never make an untried id look missing.
deadline = time.time() + 2.0
for mid in ids:
    ok = False
    for attempt in range(3):
        if attempt and time.time() > deadline:
            break
        model, res, err = pk.ping(ph, mid)
        if res == 0 and err == 0:
            ok, models[mid] = True, model
            break
        time.sleep(0.01)
    (alive if ok else dead).append(mid)
ph.closePort()

print("   %d/%d servos answered: %s" % (
    len(alive), len(ids),
    " ".join("id%d=model%s" % (m, models[m]) for m in alive) or "none"))
if dead:
    print("   MISSING_IDS %s" % ",".join(str(d) for d in dead))
    fam = {m: models[m] for m in alive}
    xm = [m for m, v in fam.items() if v in (1020, 1030)]
    xl = [m for m, v in fam.items() if v in (1190, 1200)]
    print("   answering XM430 (12 V rail): %s" % (xm or "none"))
    print("   answering XL330 (5 V rail):  %s" % (xl or "none"))
sys.exit(0 if not dead else (2 if not alive else 3))
PYEOF
  rc=$?
  case "$rc" in
    0) grn "  ✓ leader chain alive: all $NGID servos on $GPORT answered" ;;
    2) die "the handle on $GPORT is ELECTRICALLY DEAD (0/$NGID servos answered) --
     this is NOT a squeeze problem and no amount of squeezing will fix it.
     Its 5 V/12 V power brick is off/unplugged, or the chain's first cable is out.
     Check the brick's LED and the daisy-chain cable at motor 1 of that handle.
     If you believe you are squeezing the right handle, prove it — run
       ./.venv/bin/python3 tools/identify_leader_handle.py --port $GPORT
     and watch which PHYSICAL handle twitches. The udev names are CROSSED:
     /dev/leader-right is the physically LEFT handle (verified 2026-09-02)." ;;
    3) die "the leader chain on $GPORT is INCOMPLETE (see MISSING_IDS above).
     Rail rule: model 1020/1030 = XM430, on the 12 V rail; model 1190/1200 =
     XL330, on the 5 V rail. If every missing id is from ONE family, that
     family's power brick is off — plug it in, do not debug the servos.
     If the missing ids are a contiguous TAIL of the chain, the daisy-chain
     cable is out at the last id that answered." ;;
    *) die "leader chain ping could not open $GPORT (rc=$rc) --
     something else holds the serial, or the FTDI vanished. Check:
       ls -l /dev/serial/by-id/ ; fuser -v $GPORT" ;;
  esac
fi

if [ "${PREFLIGHT_ONLY:-0}" = "1" ]; then
  grn "── preflight only: rig hardware is connected. Nothing was started. ──"
  exit 0
fi

# ── 0a. persistent leader identity — the squeeze, done ONCE instead of daily ─
# 2026-09-15. The 45 s gate below proves four things; two of them (the port
# opens and the gripper id answers; Torque Enable is not latched) are already
# proven above by the chain ping and by clear_leader_torque.py, with no
# operator. The other two — which physical handle is on this port, and whether
# its trigger turns its servo — are STATIC facts of the wiring that change only
# when hardware changes. tools/leader_identity.py records them once, bound to a
# fingerprint (FTDI serial + servo id set + per-id model numbers + the device
# node's enumeration time) that is re-checked here for free, and refuses the
# moment the adapter is replugged or the config's leader→arm mapping is crossed.
#
# NO FILE = NO CHANGE. When configs/leader_identity.json is absent nothing below
# runs and the gate behaves exactly as it always did. SKIP_IDENTITY=1 forces the
# old path even when the file exists.
IDFILE="${LEADER_IDENTITY:-configs/leader_identity.json}"
if [ "$SKIP_GRIPPER" != "1" ] && [ "${SKIP_IDENTITY:-0}" != "1" ] && [ -f "$IDFILE" ]; then
  $PY tools/leader_identity.py verify --arm right --node gello_right --config "$CONFIG"
  case "$?" in
    0)  SKIP_GRIPPER=1 ;;   # identity + gripper travel already attested
    10) : ;;                # no attestation for this arm -- the 45 s gate runs
    *)  die "the stored leader identity is CONTRADICTED by the rig (reasons above).
     This is exactly the class of fault the squeeze gate existed to catch, and it
     is now caught in 2 s instead of 45. Do NOT record until it is resolved:
     re-attest after any replug, or SKIP_IDENTITY=1 to fall back to the squeeze." ;;
  esac
fi

# ── 0. the leader handle's gripper MUST move ────────────────────────────────
# Done FIRST, with the serial port free, because it is the one failure that
# produces hours of perfect-looking but untrainable data.
if [ "$SKIP_GRIPPER" != "1" ]; then
  if port_up 8792; then
    ylw "  a session already holds the leader serial; skipping the gripper gate"
    ylw "  (stop it and re-run, or SKIP_GRIPPER=1 to bypass deliberately)"
  else
    # After the 2026-09-01 leader swap the LEFT ARM is driven by the RIGHT-side
    # physical handle (the left handle's gripper servo is broken). Name the
    # handle by its PORT so this prompt survives future swaps.
    case "$GPORT" in
      *leader-right*) HANDLE="handle labelled IDS 8-14 (physically LEFT, verified 2026-09-02; the udev names are crossed). If unsure, run tools/identify_leader_handle.py and watch which handle twitches";;
      *leader-left*)  HANDLE="handle labelled IDS 1-7 (physically RIGHT side; verified 2026-09-02 alive 7/7 and chosen to drive the right arm)";;
      *)              HANDLE="leader on $GPORT (unknown side -- wiggle-test it)";;
    esac
    # An unclean session kill leaves Torque Enable latched on leader servos --
    # the motor then holds its position and the squeeze reads flat (2026-09-05,
    # runbook/gripper-gate-flat-torque-latched.md). Clear it before the gate.
    $PY tools/clear_leader_torque.py "$GPORT" "$GIDS" 2>/dev/null | sed 's/^/  /' || true
    echo "  gripper gate: squeeze the trigger of the $HANDLE"
    echo "               fully closed and release. Reading $GPORT motor $GID for 45 s ..."
    $PY - "$GPORT" "$GID" <<'PYEOF' || die "GRIPPER IS FLAT -- recording refused.
     SUSPECT THE WRONG HANDLE FIRST. The chain ping above proved these servos are
     alive and reading cleanly, so a constant position means the trigger you
     squeezed is not on this chain. 2026-09-02: five failures in a row (stuck at
     2290, then 1966) were all the wrong physical handle -- the udev names are
     CROSSED, /dev/leader-right is the physically LEFT handle. Settle it with
       ./.venv/bin/python3 tools/identify_leader_handle.py --port $GPORT
     and watch which handle twitches. Only if the RIGHT handle was squeezed is
     this mechanical:
     The trigger does not move that servo. A grasp dataset with a constant
     gripper channel cannot train a grasp policy: the exporter locates grasp
     windows BY the gripper closing, so it would find none.
     Fix the handle, or swap in the handle whose gripper works and revert the
     leader swap (rl-teleop commit fee09a6). SKIP_GRIPPER=1 bypasses this gate
     if you genuinely mean to record without gripper data."
import sys, time
from dynamixel_sdk import PortHandler, PacketHandler
port, gid = sys.argv[1], int(sys.argv[2])
ph = PortHandler(port); pk = PacketHandler(2.0)
if not ph.openPort():
    print("   cannot open %s" % port); sys.exit(1)
ph.setBaudRate(1000000)
lo = hi = None; n = 0
t0 = time.time()
while time.time() - t0 < 45:
    pos, res, err = pk.read4ByteTxRx(ph, gid, 132)
    if res == 0 and err == 0:
        n += 1
        lo = pos if lo is None else min(lo, pos)
        hi = pos if hi is None else max(hi, pos)
        if hi - lo > 25:
            print("   gripper travels: %d..%d ticks (span %d)" % (lo, hi, hi - lo))
            ph.closePort(); sys.exit(0)
    time.sleep(0.03)
ph.closePort()
print("   no travel: %s reads, stuck at %s" % (n, lo))
sys.exit(1)
PYEOF
    grn "  ✓ gripper moves"
  fi
fi

# ── 1. bus ──────────────────────────────────────────────────────────────────
if port_up 5555 && port_up 5556; then
  grn "  ✓ bus already up (5555/5556)"
else
  echo "  starting bus ..."
  tmux kill-session -t bus 2>/dev/null
  tmux new-session -d -s bus -c "$PWD" './.venv/bin/rr-bus 2>&1 | tee /tmp/rr_bus.log'
  sleep 4
  port_up 5555 || die "bus did not come up -- see /tmp/rr_bus.log"
  grn "  ✓ bus up"
fi

# ── 2. the recording session ────────────────────────────────────────────────
# record_kitting.sh's own CONTROL_PORT default (8792) is what the cockpit talks
# to. We deliberately do NOT export CONTROL_PORT here.
echo "  starting recording session ($CONFIG) ..."
tmux kill-session -t rec 2>/dev/null; sleep 6
tmux new-session -d -s rec -c "$PWD" \
  "ATTACH_BUS=1 COCKPIT_FILE=$COCKPIT_PAGE ./record_kitting.sh $CONFIG 2>&1 | tee /tmp/record_right.log"

for i in $(seq 1 40); do port_up 8792 && break; sleep 2; done
port_up 8792 || die "control surface :8792 never came up -- see /tmp/record_right.log
     (if the log shows a camera preflight failure, a camera is missing or a
     device path drifted; check by FRAME CONTENT, not by remembered port)"
grn "  ✓ session up, control :8792 answering (the port the cockpit drives)"

# ── 2b. the follower CAN bus must now carry motor traffic ───────────────────
# 2026-08-31: a gs_usb interface can be UP, ERROR-ACTIVE, zero errors -- and
# completely dead. The RobotNode then publishes joint_state from nothing, so
# the TUI, read-age and check_home are all meaningless, and the brakeless arm
# is NOT being held. Now that the session polls at 200 Hz, a live bus moves
# thousands of packets in 1.5 s; a dead one moves exactly 0.
ca=$(cat /sys/class/net/can_follow_r/statistics/rx_packets); sleep 1.5
cb=$(cat /sys/class/net/can_follow_r/statistics/rx_packets)
if [ $((cb-ca)) -lt 100 ]; then
  red "  ✗ can_follow_r moved $((cb-ca)) rx packets in 1.5 s -- the RIGHT ARM IS NOT CONTROLLED."
  red "    Every software readout of this arm is now lying. PUT A HAND ON THE ARM."
  red "    Either the right arm PSU is off, or the CAN adapter is in the up-but-dead"
  red "    state: physically replug it (usb9, port 9-1.3) -- ip-link bounces and"
  red "    gs_usb rebinds do NOT clear it. Then stop this session and re-run."
  die "left arm CAN carries no traffic"
fi
grn "  ✓ can_follow_r carrying traffic ($((cb-ca)) rx pkts / 1.5 s) -- RIGHT arm really on the bus"

# ── 3. camera bridge, ALWAYS restarted after the session ────────────────────
for p in $(pids_on 8791); do kill "$p" 2>/dev/null; done
sleep 2
setsid $PY -u -m robots_realtime.labeling.live_server --live --arm right \
  --host 0.0.0.0 --port 8791 --save-root recordings --episode-mode full \
  --control-url http://localhost:8792 \
  --bus-cams "default=camera_top/rgb,egocentric=camera_top/rgb,top=camera_top/rgb,wristL=camera_right/rgb,wristR=camera_right/rgb" \
  > /tmp/live_label_right.log 2>&1 &
for i in $(seq 1 15); do port_up 8791 && break; sleep 1; done
port_up 8791 || die "camera bridge :8791 did not start -- see /tmp/live_label_right.log"

# ── 4. counter + cockpit ────────────────────────────────────────────────────
port_up 8806 || ( cd ../cockpit && setsid python3 recording_counter.py \
  --target-left 200 --target-right 300 </dev/null > /tmp/recording_counter.log 2>&1 & )
port_up 8799 || ( cd ../cockpit && setsid python3 -m http.server 8799 </dev/null > /tmp/cockpit_http.log 2>&1 & )

# ── 4b. the SKU reader — started HERE because forgetting it costs the whole day ──
# 2026-09-21: reader_server.py was started by hand and ran free from 09:46 to 14:00.
# 3870 read cycles, 44 confident reads, and exactly TWO of them fell inside an active
# take. The reader was perfect and the session produced no usable data, because nothing
# tied the two together. Two things fix that and both live here:
#   --budget-s 3.0     was 1.5, which at the NATIVE 640x480 cost of ~0.42 s/frame buys
#                      only ~3 frames. Measured on episode_201623_4120d5ad: 60 % of the
#                      frames in a carry are legible at 640x480 and the label is readable
#                      1.6 s in, so 3 frames = 92 % and 4 = 96 %. 3.0 s buys ~7, which is
#                      margin not waste: the pass stops at the first confident read.
#   --gate rec-phase   OCR runs only inside a take whose phase is isolate or pick, so a
#                      make_space take costs no GPU and a quiet reader is CORRECT.
#   the sidecar        at REC-stop it writes <episode>/sku_reads.json + the evidence
#                      frames, so the take stops needing this process to be alive.
# NON-FATAL ON PURPOSE. This is an observer: it opens no camera (it HTTP-polls :8797,
# exactly like a browser <img>), never touches the arm, and a rig that records fine
# without it must still record fine when it fails to start. So no `die` — a warning,
# and the bring-up continues. SKU_READER=0 skips it entirely.
if [ "${SKU_READER:-1}" = "1" ] && ! port_up 8803; then
  # absolute, resolved BEFORE the subshell cd -- a relative $PY would resolve against
  # yam-pick-pipeline, which has no .venv (it borrows rl-teleop's; see its CLAUDE.md)
  RLPY="$PWD/.venv/bin/python3"
  ( cd ../yam-pick-pipeline && setsid "$RLPY" wrist_ocr/reader_server.py \
      --arms right --gate "${SKU_GATE:-rec-phase}" \
      --budget-s "${SKU_BUDGET_S:-3.0}" \
      </dev/null > /tmp/sku_reader_right.log 2>&1 & )
  for i in $(seq 1 20); do port_up 8803 && break; sleep 1; done
fi
if port_up 8803; then
  grn "  ✓ SKU reader :8803 (gate ${SKU_GATE:-rec-phase}: OCR only in isolate/pick takes)"
elif [ "${SKU_READER:-1}" = "1" ]; then
  ylw "  ⚠ SKU reader did NOT start — see /tmp/sku_reader_right.log."
  ylw "    Recording still works; you simply get no part numbers and no sku_reads.json."
fi

# ── 4c. the recording loop's two other services ─────────────────────────────
# The SKU page drives a 30-packet session hands-free, and it needs both of these. They
# are observers: neither opens a camera device and neither touches the arm.
#
#   :8815 phase_recorder  owns the HOME-TO-HOME CYCLES (and REC, and the bad-cycle mark).
#                         RESTARTED, not reused, when it is already recording: on
#                         2026-09-21 an orphan from a dead episode sat "recording" for
#                         3.9 h failing to write cycle_labels.json, and a session run
#                         through it would have produced ZERO cycle labels, discovered
#                         only at export.
#   :8840 segbench        owns candidate segmentation and the target choices the loop
#                         writes to targets.jsonl.
if [ "${REC_LOOP:-1}" = "1" ]; then
  if port_up 8815 && curl -s -m 3 http://127.0.0.1:8815/api/state 2>/dev/null \
       | grep -q '"recording": *true'; then
    ylw "  ⚠ phase_recorder :8815 is stuck mid-take from a previous session — restarting it"
    for pp in $(pids_on 8815); do kill "$pp" 2>/dev/null; done
    sleep 2
  fi
  if ! port_up 8815; then
    ( cd ../training_experiment && setsid "$PWD/../rl-teleop/.venv/bin/python3" \
        tools/phase_recorder.py --phase isolate --arm right \
        </dev/null > /tmp/phase_recorder.log 2>&1 & )
    for i in $(seq 1 15); do port_up 8815 && break; sleep 1; done
  fi
  if ! port_up 8840; then
    ( cd ../yam-pick-pipeline/perception/segbench && setsid python3 server.py --port 8840 \
        </dev/null > /tmp/segbench.log 2>&1 & )
    for i in $(seq 1 15); do port_up 8840 && break; sleep 1; done
  fi
  port_up 8815 && grn "  ✓ Zyklen :8815 (phase_recorder — owns home-to-home cycles)" \
                || ylw "  ⚠ phase_recorder :8815 down — see /tmp/phase_recorder.log (no cycles, no auto-segment)"
  port_up 8840 && grn "  ✓ Segmentierung :8840 (segbench — candidates + target choices)" \
                || ylw "  ⚠ segbench :8840 down — see /tmp/segbench.log (no candidates)"
fi

sleep 4

# ── 5. verify what the operator will actually see ───────────────────────────
echo "  verifying the panels ..."
fail=0
for id in top egocentric wristR; do
  code=$(curl -s -m 5 -o /dev/null -w "%{http_code}" "http://127.0.0.1:8791/cam/$id")
  [ "$code" = "200" ] && grn "    ✓ /cam/$id" || { red "    ✗ /cam/$id -> $code"; fail=1; }
done
timeout 25 $PY tools/check_streams.py --secs 5 2>&1 | grep -E "camera_(right|top)" | sed 's/^/    /'
[ "$fail" = "0" ] || die "a camera panel is dead -- do not record until it is fixed"

# ── 5b. runtime gripper watchdog ────────────────────────────────────────────
# The squeeze gate's REAL job was protecting training data: a corpus whose
# gripper channel never moves has no grasp windows, because the exporter finds
# grasps BY the gripper closing. The gate proved that on the SERVO, before the
# session existed, and then trusted it forever. This watches the thing that is
# actually recorded -- gello_right/joint_pos on the bus -- and warns in
# THIS terminal if the channel stays flat while the arm is being driven. It
# holds no device, opens no serial, and can never block or slow the session.
# WATCHDOG=0 disables it.
if [ "${WATCHDOG:-1}" = "1" ]; then
  ( $PY -u tools/gripper_watchdog.py --node gello_right --secs 900 2>&1 \
      | tee -a /tmp/gripper_watchdog_right.log ) &
  grn "  ✓ gripper-channel watchdog armed (warns here if the channel goes flat --"
  grn "    log /tmp/gripper_watchdog_right.log; WATCHDOG=0 disables)"
fi

echo
grn "── ready ───────────────────────────────────────────────────────────"
cat <<EOF
  cockpit   http://localhost:8799/$COCKPIT_PAGE      (hard-reload: Ctrl+Shift+R)
  SKU-Seite http://localhost:8799/Buehler-Kitting-Record-SKU.html
            (rechter Arm: gelesene Teilenummer + Beweis-Frame + Ziel-Fach.
             Phase mit 1-4 umschalten; OCR läuft nur in Isolieren/Greifen.)
  control   http://localhost:8792                    (the REC button drives this)
  counter   http://localhost:8806/counts

  IF YOU ARE ON SSH, forward these ports or the page will look dead even though
  every server here is healthy -- 8792, 8797, 8805 bind 127.0.0.1 only:

      8799  8791  8792  8806  8803  8797  8815  8840

  BEFORE RECORDING MORE THAN ONE TAKE:
    1. set TAKT to Auto      (each scored placement saves the take, one take = one grasp)
    2. press Aufnahme and watch the tile start counting seconds.
       If it stays "bereit", NOTHING IS BEING SAVED -- stop and fix it.

  Stop everything:  tmux kill-session -t rec
EOF
