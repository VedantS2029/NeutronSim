#!/bin/bash
# Double-click to open Neutron Sim on a Mac. Nothing to install if Python 3.10+ is present.
cd "$(dirname "$0")" || exit 1

CHECK='import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)'
for py in /Library/Frameworks/Python.framework/Versions/Current/bin/python3 \
          /opt/homebrew/bin/python3 /usr/local/bin/python3 python3; do
  if command -v "$py" >/dev/null 2>&1 && "$py" -c "$CHECK" 2>/dev/null; then
    # run detached so this Terminal window can close; launcher.py shows its own error dialogs
    nohup "$py" launcher.py >/dev/null 2>&1 &
    sleep 1
    osascript -e 'tell application "Terminal" to close (every window whose name contains "Neutron Sim")' >/dev/null 2>&1 &
    exit 0
  fi
done

osascript -e 'display dialog "Neutron Sim needs Python 3.10 or newer, which was not found on this Mac.\n\nClick OK to open the download page. Install it, then double-click Neutron Sim.command again." with title "Neutron Sim" buttons {"OK"} default button "OK"' >/dev/null 2>&1
open "https://www.python.org/downloads/macos/"
