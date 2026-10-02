#!/usr/bin/env python3
"""Benchmark Dynamixel GELLO leader read rates.

This script separates raw Dynamixel bus timing from the full
DynamixelGelloLeaderAgent path. Use it to diagnose cases where one leader runs
much slower than the other.
"""

from __future__ import annotations

import argparse
import importlib
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from robots_realtime.agents.teleoperation.dynamixel_gello_leader_agent import _DynamixelPositionReader


@dataclass
class BenchmarkResult:
    name: str
    kind: str
    port: str
    motor_ids: list[int]
    samples: int
    successes: int
    errors: int
    hz: float
    mean_ms: float
    median_ms: float
    p95_ms: float
    max_ms: float
    error_examples: list[str]


def _parse_ids(value: str) -> list[int]:
    return [int(part.strip(), 0) for part in value.split(",") if part.strip()]


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    values_sorted = sorted(values)
    idx = int(round((len(values_sorted) - 1) * percentile))
    return values_sorted[idx]


def _summarize(
    *,
    name: str,
    kind: str,
    port: str,
    motor_ids: list[int],
    samples: int,
    durations_ms: list[float],
    success_times: list[float],
    error_examples: list[str],
) -> BenchmarkResult:
    span = success_times[-1] - success_times[0] if len(success_times) > 1 else 0.0
    hz = (len(success_times) - 1) / span if span > 0 else 0.0
    return BenchmarkResult(
        name=name,
        kind=kind,
        port=port,
        motor_ids=motor_ids,
        samples=samples,
        successes=len(success_times),
        errors=samples - len(success_times),
        hz=hz,
        mean_ms=statistics.fmean(durations_ms) if durations_ms else 0.0,
        median_ms=statistics.median(durations_ms) if durations_ms else 0.0,
        p95_ms=_percentile(durations_ms, 0.95),
        max_ms=max(durations_ms) if durations_ms else 0.0,
        error_examples=error_examples[:5],
    )


def benchmark_raw(
    *,
    name: str,
    port: str,
    motor_ids: list[int],
    samples: int,
    baudrate: int,
    protocol_version: float,
    present_position_addr: int,
    present_position_len: int,
    num_read_retries: int,
    pause_s: float,
) -> BenchmarkResult:
    durations_ms: list[float] = []
    success_times: list[float] = []
    error_examples: list[str] = []
    try:
        reader = _DynamixelPositionReader(
            port=port,
            baudrate=baudrate,
            motor_ids=motor_ids,
            protocol_version=protocol_version,
            present_position_addr=present_position_addr,
            present_position_len=present_position_len,
            num_read_retries=num_read_retries,
        )
    except Exception as exc:
        return BenchmarkResult(
            name=name,
            kind="raw",
            port=port,
            motor_ids=motor_ids,
            samples=samples,
            successes=0,
            errors=samples,
            hz=0.0,
            mean_ms=0.0,
            median_ms=0.0,
            p95_ms=0.0,
            max_ms=0.0,
            error_examples=[f"setup failed: {exc}"],
        )

    try:
        for _ in range(samples):
            start = time.perf_counter()
            try:
                reader.get_positions()
                success_times.append(time.perf_counter())
            except Exception as exc:
                if len(error_examples) < 5:
                    error_examples.append(str(exc))
            finally:
                durations_ms.append((time.perf_counter() - start) * 1e3)
            if pause_s > 0:
                time.sleep(pause_s)
    finally:
        reader.close()

    return _summarize(
        name=name,
        kind="raw",
        port=port,
        motor_ids=motor_ids,
        samples=samples,
        durations_ms=durations_ms,
        success_times=success_times,
        error_examples=error_examples,
    )


def _load_agent_kwargs(config_path: Path, node_name: str) -> tuple[str, dict[str, Any], list[int]]:
    config = yaml.safe_load(config_path.read_text())
    for node in config["nodes"]:
        if node.get("type") == "AgentNode" and node.get("name") == node_name:
            module_name, class_name = node["agent_class"].split(":", 1)
            kwargs = dict(node.get("agent_kwargs") or {})
            motor_ids = list(kwargs.get("motor_ids", []))
            return f"{module_name}:{class_name}", kwargs, motor_ids
    raise KeyError(f"AgentNode {node_name!r} not found in {config_path}")


def benchmark_agent(
    *,
    name: str,
    config_path: Path,
    node_name: str,
    port: str,
    samples: int,
    gripper_spring: bool | None,
    pause_s: float,
) -> BenchmarkResult:
    target, kwargs, motor_ids = _load_agent_kwargs(config_path, node_name)
    module_name, class_name = target.split(":", 1)
    cls = getattr(importlib.import_module(module_name), class_name)
    kwargs["port"] = port
    if gripper_spring is not None:
        kwargs["gripper_spring"] = gripper_spring

    durations_ms: list[float] = []
    success_times: list[float] = []
    error_examples: list[str] = []
    try:
        agent = cls(**kwargs)
    except Exception as exc:
        return BenchmarkResult(
            name=name,
            kind="agent",
            port=port,
            motor_ids=motor_ids,
            samples=samples,
            successes=0,
            errors=samples,
            hz=0.0,
            mean_ms=0.0,
            median_ms=0.0,
            p95_ms=0.0,
            max_ms=0.0,
            error_examples=[f"setup failed: {exc}"],
        )

    try:
        for _ in range(samples):
            start = time.perf_counter()
            try:
                action = agent.act({})
                if action:
                    success_times.append(time.perf_counter())
            except Exception as exc:
                if len(error_examples) < 5:
                    error_examples.append(str(exc))
            finally:
                durations_ms.append((time.perf_counter() - start) * 1e3)
            if pause_s > 0:
                time.sleep(pause_s)
    finally:
        agent.close()

    return _summarize(
        name=name,
        kind="agent",
        port=port,
        motor_ids=motor_ids,
        samples=samples,
        durations_ms=durations_ms,
        success_times=success_times,
        error_examples=error_examples,
    )


def print_table(results: list[BenchmarkResult]) -> None:
    print()
    print(
        f"{'name':<24} {'kind':<8} {'ids':<22} {'ok':>5} {'err':>5} "
        f"{'hz':>8} {'mean':>9} {'p95':>9} {'max':>9}"
    )
    print("-" * 104)
    for result in results:
        ids = ",".join(str(motor_id) for motor_id in result.motor_ids)
        if len(ids) > 21:
            ids = ids[:18] + "..."
        print(
            f"{result.name:<24} {result.kind:<8} {ids:<22} "
            f"{result.successes:>5} {result.errors:>5} "
            f"{result.hz:>8.1f} {result.mean_ms:>8.2f}ms "
            f"{result.p95_ms:>8.2f}ms {result.max_ms:>8.2f}ms"
        )
        if result.error_examples:
            print(f"  errors: {result.error_examples[0]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-port", default="/dev/leader-left")
    parser.add_argument("--right-port", default="/dev/leader-right")
    parser.add_argument("--left-ids", default="1,2,3,4,5,6,7")
    parser.add_argument("--right-ids", default="8,9,10,11,12,13,14")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--single-samples", type=int, default=60)
    parser.add_argument("--baudrate", type=int, default=1_000_000)
    parser.add_argument("--protocol-version", type=float, default=2.0)
    parser.add_argument("--present-position-addr", type=int, default=132)
    parser.add_argument("--present-position-len", type=int, default=4)
    parser.add_argument("--num-read-retries", type=int, default=0)
    parser.add_argument("--pause-s", type=float, default=0.0)
    parser.add_argument("--include-subsets", action="store_true")
    parser.add_argument("--include-single-motors", action="store_true")
    parser.add_argument("--include-agent", action="store_true")
    parser.add_argument("--config", default="configs/yam/yam_bimanual_dynamixel_gello_teleop.yaml")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    left_ids = _parse_ids(args.left_ids)
    right_ids = _parse_ids(args.right_ids)
    results: list[BenchmarkResult] = []

    raw_cases = [
        ("left-all", args.left_port, left_ids, args.samples),
        ("right-all", args.right_port, right_ids, args.samples),
    ]
    if args.include_subsets:
        raw_cases.extend(
            [
                ("left-arm", args.left_port, left_ids[:6], args.samples),
                ("right-arm", args.right_port, right_ids[:6], args.samples),
                ("left-gripper", args.left_port, [left_ids[-1]], args.single_samples),
                ("right-gripper", args.right_port, [right_ids[-1]], args.single_samples),
            ]
        )
    if args.include_single_motors:
        raw_cases.extend((f"left-id-{motor_id}", args.left_port, [motor_id], args.single_samples) for motor_id in left_ids)
        raw_cases.extend((f"right-id-{motor_id}", args.right_port, [motor_id], args.single_samples) for motor_id in right_ids)

    for name, port, motor_ids, samples in raw_cases:
        print(f"Running raw benchmark: {name} on {port} ids={motor_ids}")
        results.append(
            benchmark_raw(
                name=name,
                port=port,
                motor_ids=motor_ids,
                samples=samples,
                baudrate=args.baudrate,
                protocol_version=args.protocol_version,
                present_position_addr=args.present_position_addr,
                present_position_len=args.present_position_len,
                num_read_retries=args.num_read_retries,
                pause_s=args.pause_s,
            )
        )

    if args.include_agent:
        config_path = Path(args.config)
        agent_cases = [
            ("left-agent-no-spring", "gello_left", args.left_port, False),
            ("right-agent-no-spring", "gello_right", args.right_port, False),
            ("left-agent-config", "gello_left", args.left_port, None),
            ("right-agent-config", "gello_right", args.right_port, None),
        ]
        for name, node_name, port, spring in agent_cases:
            print(f"Running agent benchmark: {name} on {port}")
            results.append(
                benchmark_agent(
                    name=name,
                    config_path=config_path,
                    node_name=node_name,
                    port=port,
                    samples=args.samples,
                    gripper_spring=spring,
                    pause_s=args.pause_s,
                )
            )

    print_table(results)

    if args.output:
        payload = [asdict(result) for result in results]
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
