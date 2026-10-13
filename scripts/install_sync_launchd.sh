#!/usr/bin/env bash
# Install a macOS launchd user agent that runs scripts/sync_results.sh every 30 minutes.
# The repo path is the directory this script is run from (its parent). Logs go to ~/Downloads/hybridquant_results/.
set -eu
REPO="$(cd "$(dirname "$0")/.." && pwd)"
LABEL=com.hybridquant.sync
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
DEST_ROOT="$HOME/Downloads/hybridquant_results"
mkdir -p "$HOME/Library/LaunchAgents" "$DEST_ROOT"

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>$REPO/scripts/sync_results.sh</string>
    <string>$REPO</string>
    <string>$DEST_ROOT</string>
  </array>
  <key>StartInterval</key><integer>1800</integer>
  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>$DEST_ROOT/launchd_stdout.log</string>
  <key>StandardErrorPath</key><string>$DEST_ROOT/launchd_stderr.log</string>
  <key>EnvironmentVariables</key>
  <dict><key>PATH</key><string>/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin</string></dict>
</dict>
</plist>
EOF

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "installed $PLIST (every 30 min, repo $REPO)"
launchctl list | grep "$LABEL" || { echo "WARNING: $LABEL not listed"; exit 1; }
