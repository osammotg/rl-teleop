#!/usr/bin/env bash
.venv/bin/python scripts/replay_episode.py PIXEL_IDM_INFERENCE/trajectory_actions.npz --npz-window-idx 0 --port 8080
