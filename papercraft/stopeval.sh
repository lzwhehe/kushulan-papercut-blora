#!/bin/bash
# Stop the evaluation watcher and any running evaluate.py (pattern kept in this file
# so that the calling shell's own command line never matches).
for p in $(pgrep -f "eval_watch[.]sh") $(pgrep -f "python evaluate[.]py"); do
  [ "$p" != "$$" ] && kill "$p" 2>/dev/null
done
exit 0
