"""cv-camera-map CLI — enumerate, enroll, identify, validate, live-qa."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cv_camera_mapping import __version__
from cv_camera_mapping.assignment import identify_roles
from cv_camera_mapping.enumeration import enumerate_cameras
from cv_camera_mapping.reference_capture import (
    capture_references,
    parse_role_devices,
    validate_mapping_artifact,
)
from cv_camera_mapping.schemas import FailureReason, MatcherSettings


def _print_json(data: dict, output: Path | None) -> None:
    text = json.dumps(data, indent=2)
    print(text)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)


def cmd_enumerate(args: argparse.Namespace) -> int:
    result = enumerate_cameras(include_non_capture=args.include_non_capture)
    payload = result.model_dump(mode="json")
    _print_json(payload, Path(args.output) if args.output else None)
    return 0


def cmd_capture_reference(args: argparse.Namespace) -> int:
    try:
        role_devices = parse_role_devices(args.role_device or [])
        roles = [r.strip() for r in args.roles.split(",")]
        capture_references(
            Path(args.output),
            roles,
            role_devices,
            resolution=(args.width, args.height),
            fps=args.fps,
        )
        return 0
    except ValueError as exc:
        msg = str(exc)
        if args.json:
            reason = (
                FailureReason.ROLE_DEVICE_MAPPING_REQUIRED.value
                if "role_device" in msg
                else "capture_failed"
            )
            _print_json(
                {
                    "schema_version": 1,
                    "assignments": {},
                    "ambiguous": True,
                    "failure_reason": reason,
                    "detail": msg,
                },
                Path(args.output) / "error.json" if args.output else None,
            )
        else:
            print(msg, file=sys.stderr)
        return 1


def cmd_identify(args: argparse.Namespace) -> int:
    settings = MatcherSettings(
        confidence_threshold=args.confidence_threshold,
        margin_threshold=args.margin_threshold,
        min_inliers=args.min_inliers,
        min_keypoints=args.min_keypoints,
        min_raw_matches=args.min_raw_matches,
        ratio_threshold=0.75,
    )
    result = identify_roles(
        Path(args.references),
        settings=settings,
        include_devices=args.include_device or None,
        exclude_devices=args.exclude_device or None,
        product_filter=args.product_filter,
    )
    payload = result.artifact.model_dump(mode="json")
    out = Path(args.output) if args.output else None
    if args.json or out:
        _print_json(payload, out)
    if result.exit_code != 0 and not args.json:
        print(json.dumps(payload, indent=2), file=sys.stderr)
    return result.exit_code


def cmd_validate_mapping(args: argparse.Namespace) -> int:
    path = Path(args.mapping)
    try:
        artifact = validate_mapping_artifact(path)
        if args.check_devices:
            from cv_camera_mapping.enumeration import enumerate_cameras

            enum = enumerate_cameras()
            paths = {c.device_path for c in enum.candidates if c.capture_capable}
            for role, asn in artifact.assignments.items():
                if asn.device_path not in paths:
                    raise ValueError(f"{role}: device {asn.device_path} not currently available")
        if args.json:
            _print_json({"valid": True, "artifact": artifact.model_dump(mode="json")}, None)
        else:
            print(f"Valid mapping: {path}")
        return 0
    except Exception as exc:
        reason = FailureReason.MAPPING_ARTIFACT_MISSING.value
        detail = str(exc)
        if path.exists():
            try:
                raw = json.loads(path.read_text())
                if raw.get("ambiguous") and raw.get("failure_reason"):
                    reason = raw["failure_reason"]
                    detail = (
                        "identify did not produce a valid mapping; re-run identify after fixing. "
                        f"Original: {exc}"
                    )
            except json.JSONDecodeError:
                pass
        if args.json:
            _print_json(
                {
                    "valid": False,
                    "failure_reason": reason,
                    "detail": detail,
                },
                None,
            )
        else:
            print(str(exc), file=sys.stderr)
        return 1


def cmd_live_qa(args: argparse.Namespace) -> int:
    evidence_path = Path(args.evidence)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_path.parent.mkdir(parents=True, exist_ok=True)

    enum = enumerate_cameras()
    capture_candidates = [c for c in enum.candidates if c.capture_capable]
    payload: dict = {
        "schema_version": 1,
        "mapper_version": __version__,
        "candidate_count": len(capture_candidates),
        "skipped": False,
    }

    if len(capture_candidates) < 2:
        payload["skipped"] = True
        payload["message"] = "fewer_than_two_capture_capable_cameras"
        evidence_path.write_text(json.dumps(payload, indent=2))
        return 0

    refs_dir = output_dir / "live_refs"
    try:
        capture_references(
            refs_dir,
            ["camera_left", "camera_right"],
            {
                "camera_left": capture_candidates[0].device_path,
                "camera_right": capture_candidates[1].device_path,
            },
        )
        result = identify_roles(refs_dir)
        payload["identify"] = result.artifact.model_dump(mode="json")
        payload["identify_exit_code"] = result.exit_code
    except Exception as exc:
        payload["error"] = str(exc)

    evidence_path.write_text(json.dumps(payload, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cv-camera-map")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_enum = sub.add_parser("enumerate", help="List V4L2 capture-capable cameras")
    p_enum.add_argument("--json", action="store_true", help="JSON output")
    p_enum.add_argument("--include-non-capture", action="store_true")
    p_enum.add_argument("--output", type=str, default=None)

    p_cap = sub.add_parser("capture-reference", help="Enroll role reference images")
    p_cap.add_argument("--roles", type=str, default="camera_left,camera_right")
    p_cap.add_argument("--output", type=str, required=True)
    p_cap.add_argument(
        "--role-device",
        action="append",
        default=[],
        help="e.g. camera_left=/dev/video0",
    )
    p_cap.add_argument("--json", action="store_true")
    p_cap.add_argument("--width", type=int, default=640)
    p_cap.add_argument("--height", type=int, default=480)
    p_cap.add_argument("--fps", type=int, default=30)

    p_id = sub.add_parser("identify", help="Identify camera roles from references")
    p_id.add_argument("--references", type=str, required=True)
    p_id.add_argument("--json", action="store_true")
    p_id.add_argument("--output", type=str, default=None)
    p_id.add_argument("--confidence-threshold", type=float, default=0.75)
    p_id.add_argument("--margin-threshold", type=float, default=0.15)
    p_id.add_argument("--min-inliers", type=int, default=20)
    p_id.add_argument("--min-keypoints", type=int, default=50)
    p_id.add_argument("--min-raw-matches", type=int, default=30)
    p_id.add_argument(
        "--include-device",
        action="append",
        default=[],
        help="Only consider these device paths (repeatable)",
    )
    p_id.add_argument(
        "--exclude-device",
        action="append",
        default=[],
        help="Ignore these device paths (repeatable)",
    )
    p_id.add_argument(
        "--product-filter",
        type=str,
        default=None,
        help="Only devices whose product/serial contains this substring (e.g. Innomaker)",
    )

    p_val = sub.add_parser("validate-mapping", help="Validate mapping JSON artifact")
    p_val.add_argument("--mapping", type=str, required=True)
    p_val.add_argument("--json", action="store_true")
    p_val.add_argument("--check-devices", action="store_true")

    p_qa = sub.add_parser("live-qa", help="Bounded hardware QA with safe skip")
    p_qa.add_argument("--output", type=str, required=True)
    p_qa.add_argument("--evidence", type=str, required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "enumerate": cmd_enumerate,
        "capture-reference": cmd_capture_reference,
        "identify": cmd_identify,
        "validate-mapping": cmd_validate_mapping,
        "live-qa": cmd_live_qa,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
