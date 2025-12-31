#!/usr/bin/env bash
# 50-baton.sh - aws-devbox post-boot hook to start Baton
# Place in: ~/.config/aws-devbox/hooks/post-boot/50-baton.sh

set -euo pipefail

log() { echo "[$(date -Iseconds)] [baton] $*"; }

# Ensure log directory exists on LUKS volume
LOG_DIR="/data/baton/logs"
AUTH_DIR="/data/baton/auth"

if [[ -d "/data" ]]; then
    mkdir -p "$LOG_DIR" "$AUTH_DIR"
    chmod 700 "$AUTH_DIR"
fi

# Check if baton is available
if ! command -v baton &>/dev/null; then
    # Try pip install location
    BATON_BIN="$HOME/.local/bin/baton"
    if [[ ! -x "$BATON_BIN" ]]; then
        log "WARNING: baton not installed, skipping"
        exit 0
    fi
fi

# Restore auth state if available
if command -v baton &>/dev/null; then
    baton auth restore 2>/dev/null || true
fi

# Start baton via systemd user service
if systemctl --user is-enabled baton.service &>/dev/null; then
    log "Starting Baton service..."
    systemctl --user start baton.service
    sleep 2
    if systemctl --user is-active baton.service &>/dev/null; then
        log "Baton started successfully"
    else
        log "WARNING: Baton failed to start"
        systemctl --user status baton.service --no-pager || true
    fi
else
    log "Baton service not enabled, skipping"
fi
