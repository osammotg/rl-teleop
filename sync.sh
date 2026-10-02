#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
exec scripts/sync_episodes_to_s3.py "$@"
