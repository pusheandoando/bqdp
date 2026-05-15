#!/usr/bin/env bash





find . -type d -name "__pycache__" -print -exec rm -rf {} +
echo "[OK] All __pycache__ folders were deleted"