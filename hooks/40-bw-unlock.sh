#!/usr/bin/env bash
# 40-bw-unlock.sh - aws-devbox post-boot hook to unlock Bitwarden
# Place in: ~/.config/aws-devbox/hooks/post-boot/40-bw-unlock.sh

set -euo pipefail

BW_SESSION_FILE="/data/baton/auth/bw_session"
BW_CONFIG_SESSION="$HOME/.config/bitwarden/session"

log() { echo "[$(date -Iseconds)] [bw-unlock] $*"; }

# Check if bw is available
if ! command -v bw &>/dev/null; then
    log "WARNING: bw not installed, skipping"
    exit 0
fi

# Try to restore saved session from LUKS volume
if [[ -f "$BW_SESSION_FILE" ]]; then
    log "Restoring BW session from LUKS volume..."
    BW_SESSION=$(cat "$BW_SESSION_FILE")
    export BW_SESSION

    # Verify session is valid
    status=$(bw status 2>/dev/null | jq -r '.status' 2>/dev/null || echo "unknown")
    if [[ "$status" == "unlocked" ]]; then
        log "Session restored successfully"
        # Copy to standard location
        mkdir -p "$(dirname "$BW_CONFIG_SESSION")"
        cp "$BW_SESSION_FILE" "$BW_CONFIG_SESSION"
        chmod 600 "$BW_CONFIG_SESSION"
        exit 0
    else
        log "Saved session is expired, will need interactive unlock"
    fi
fi

# Session needs interactive unlock
# This will be handled by user running `source bw-unlock`
log "BW session not available - run 'source bw-unlock' manually after boot"
exit 0
