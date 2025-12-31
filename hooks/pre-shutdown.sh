#!/usr/bin/env bash
# pre-shutdown.sh - aws-devbox pre-shutdown hook to save Baton state
# Place in: ~/.config/aws-devbox/hooks/pre-shutdown/50-baton.sh

set -euo pipefail

log() { echo "[$(date -Iseconds)] [baton-shutdown] $*"; }

BW_SESSION_FILE="/data/baton/auth/bw_session"
BW_CONFIG_SESSION="$HOME/.config/bitwarden/session"

# Save BW session to LUKS volume
if [[ -f "$BW_CONFIG_SESSION" ]]; then
    log "Saving BW session to LUKS volume..."
    mkdir -p "$(dirname "$BW_SESSION_FILE")"
    cp "$BW_CONFIG_SESSION" "$BW_SESSION_FILE"
    chmod 600 "$BW_SESSION_FILE"
    log "BW session saved"
fi

# Save Baton auth state
if command -v baton &>/dev/null; then
    log "Saving Baton auth state..."
    baton auth save 2>/dev/null || true
fi

# Stop Baton service gracefully
if systemctl --user is-active baton.service &>/dev/null; then
    log "Stopping Baton service..."
    systemctl --user stop baton.service
    log "Baton stopped"
fi

log "Shutdown complete"
