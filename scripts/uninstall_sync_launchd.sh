#!/usr/bin/env bash
# Remove the launchd agent installed by install_sync_launchd.sh. Synced files are left in place.
set -u
LABEL=com.hybridquant.sync
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null && echo "unloaded $LABEL" || echo "$LABEL was not loaded"
rm -f "$PLIST" && echo "removed $PLIST"
launchctl list | grep -q "$LABEL" && echo "WARNING: still listed" || echo "ok: not listed"
