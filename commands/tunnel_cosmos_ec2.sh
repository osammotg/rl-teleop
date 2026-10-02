#!/usr/bin/env bash
set -euo pipefail

EC2_SSH_TARGET="${1:-${COSMOS_EC2_SSH_TARGET:-}}"
LOCAL_COSMOS_PORT=8021
REMOTE_COSMOS_HOST=127.0.0.1
REMOTE_COSMOS_PORT=8021

if [[ -z "$EC2_SSH_TARGET" ]]; then
  echo "Usage: $0 ubuntu@EC2_HOST" >&2
  echo "Or set COSMOS_EC2_SSH_TARGET=ubuntu@EC2_HOST." >&2
  exit 2
fi

echo "Forwarding http://127.0.0.1:${LOCAL_COSMOS_PORT} -> ${EC2_SSH_TARGET}:${REMOTE_COSMOS_HOST}:${REMOTE_COSMOS_PORT}"
echo "Leave this running while rl-teleop uses Cosmos."

exec ssh \
  -N \
  -T \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -L "127.0.0.1:${LOCAL_COSMOS_PORT}:${REMOTE_COSMOS_HOST}:${REMOTE_COSMOS_PORT}" \
  "$EC2_SSH_TARGET"
