"""Publish REAL camera frames to the ZMQ bus so the cockpit shows the actual camera view
(not the synthetic loader bars). Opens the physical D455 (camera_top -> scan+egocentric)
and the USB wrist cam (camera_left -> wristL+wristR), same params as
configs/yam/yam_left_kitting_teleop.yaml. Robot/rr-session NOT required.

Each camera runs in its own thread and publishes {"images":{"rgb": frame}} to <name>/rgb.
If a camera fails to open, the others keep going.
"""
import os, threading, time, traceback
os.environ.setdefault("RS2_USE_RSUSB_BACKEND", "true")

from robots_realtime.runtime.transport.message_bus import MessageBus
from robots_realtime.runtime.transport.publisher import Publisher

# 1) broker (publishers connect :5555, subscribers :5556)
bus = MessageBus()
try:
    bus.start()
    print("broker started on :5555/:5556", flush=True)
except Exception as e:
    print(f"broker not started ({e}); assuming one is up", flush=True)


def pump(name, make_driver):
    """Open a camera via make_driver() and publish its frames forever."""
    try:
        cam = make_driver()
        print(f"[{name}] opened: {cam!r}", flush=True)
    except Exception:
        print(f"[{name}] FAILED to open:\n{traceback.format_exc()}", flush=True)
        return
    pub = Publisher(node_name=name)      # topic prefix -> <name>/rgb
    time.sleep(0.3)
    n, t0 = 0, time.time()
    while True:
        try:
            data = cam.read()
            rgb = data.images.get("rgb") if hasattr(data, "images") else data["images"]["rgb"]
            if rgb is None:
                continue
            pub.publish("rgb", {"images": {"rgb": rgb}})
            n += 1
            if n % 60 == 0:
                dt = time.time() - t0
                print(f"[{name}] {n} frames, {n/dt:.1f} fps, shape={getattr(rgb,'shape',None)}", flush=True)
        except Exception:
            print(f"[{name}] read/publish error:\n{traceback.format_exc()}", flush=True)
            time.sleep(0.2)


def make_top():
    from robots_realtime.sensors.cameras.realsense_camera import RealSenseCamera
    return RealSenseCamera(device_id="203522250539", resolution=(1280, 720),
                           fps=30, enable_depth=False)


def make_scan():
    # D435i mounted top-down over the box — the label/OCR "scan" view.
    from robots_realtime.sensors.cameras.realsense_camera import RealSenseCamera
    return RealSenseCamera(device_id="241222077246", resolution=(1280, 720),
                           fps=30, enable_depth=False)


def _find_wrist_node():
    """The two Innomaker wrist cams share a serial and their /dev/video mapping
    DRIFTS across reboots/replugs (see kitting config comment). Probe the known
    by-path candidates and return the first that actually delivers a frame."""
    import glob, cv2
    # Only the Innomaker wrist by-paths (never raw /dev/video*, which could be a
    # RealSense node). Prefer the config's controller (74:00.0) over the other.
    all_bp = glob.glob("/dev/v4l/by-path/*usb-0:1*video-index0")
    cands = sorted(all_bp, key=lambda p: (0 if "74:00.0" in p else 1, p))
    for dev in cands:
        try:
            cap = cv2.VideoCapture(dev)
            if not cap.isOpened():
                cap.release(); continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            ok = any(cap.read()[0] for _ in range(8))
            cap.release()
            if ok:
                return dev
        except Exception:
            pass
    raise RuntimeError("no working wrist camera node found")


def make_left():
    from robots_realtime.sensors.cameras.opencv_camera import OpencvCamera
    dev = _find_wrist_node()
    print(f"[camera_left] auto-detected working wrist node: {dev}", flush=True)
    return OpencvCamera(
        device_path=dev, resolution=(640, 480), fps=30,
        v4l2_controls={"brightness": 0, "contrast": 28, "saturation": 64,
                       "white_balance_automatic": 1, "gamma": 100, "gain": 0,
                       "power_line_frequency": 1})


threads = [
    threading.Thread(target=pump, args=("camera_top", make_top), daemon=True),
    threading.Thread(target=pump, args=("camera_scan", make_scan), daemon=True),
    threading.Thread(target=pump, args=("camera_left", make_left), daemon=True),
]
for t in threads:
    t.start()
    time.sleep(0.5)

while True:
    time.sleep(1)
