#!/bin/bash
# Mac Mini M1 Space Cleanup with Trace Logging
# Safe to run after Application Support cleanup

set -euo pipefail

LOG_FILE="$HOME/cleanup_trace_$(date +%Y%m%d_%H%M%S).log"
DISK=$(df / | tail -1 | awk '{print $1}')

log_space() {
    local label="$1"
    local avail=$(df -h / | tail -1 | awk '{print $4}')
    echo "[$(date '+%H:%M:%S')] $label: $avail available" | tee -a "$LOG_FILE"
}

echo "=== Mac Mini M1 Cleanup Trace ===" | tee "$LOG_FILE"
echo "Started: $(date)" | tee -a "$LOG_FILE"
log_space "BEFORE ALL CLEANUP"
echo "" | tee -a "$LOG_FILE"

# ──────────────────────────────────────────────
# 1. XCODE ARTIFACTS
# ──────────────────────────────────────────────
echo ">>> [1/7] Xcode DerivedData, Simulators, Archives" | tee -a "$LOG_FILE"
log_space "  Before Xcode"

rm -rf ~/Library/Developer/Xcode/DerivedData/* 2>/dev/null || true
xcrun simctl delete unavailable 2>/dev/null || true
find ~/Library/Developer/Xcode/Archives -maxdepth 1 -mtime +90 -exec rm -rf {} \; 2>/dev/null || true

log_space "  After Xcode"
echo "" | tee -a "$LOG_FILE"

# ──────────────────────────────────────────────
# 2. DOCKER / ORBSTACK / COLIMA
# ──────────────────────────────────────────────
echo ">>> [2/7] Container Runtimes" | tee -a "$LOG_FILE"
log_space "  Before Containers"

docker system prune -a --volumes -f 2>/dev/null || true
orbctl prune --all 2>/dev/null || true
# NOTE: Colima NOT included here — requires explicit confirmation first

log_space "  After Containers"
echo "" | tee -a "$LOG_FILE"

# ──────────────────────────────────────────────
# 3. HOMEBREW (includes fix for broken cask loader)
# ──────────────────────────────────────────────
echo ">>> [3/7] Homebrew Fix + Cleanup" | tee -a "$LOG_FILE"
log_space "  Before Homebrew"

brew update 2>&1 | tail -1 | tee -a "$LOG_FILE"
brew cleanup -s --prune=all 2>&1 | tail -5 | tee -a "$LOG_FILE"
brew autoremove 2>&1 | tail -3 | tee -a "$LOG_FILE"

log_space "  After Homebrew"
echo "" | tee -a "$LOG_FILE"

# ──────────────────────────────────────────────
# 4. NODE PACKAGE MANAGERS
# ──────────────────────────────────────────────
echo ">>> [4/7] Node Package Manager Caches" | tee -a "$LOG_FILE"
log_space "  Before Node PMs"

npm cache clean --force 2>/dev/null || true
yarn cache clean 2>/dev/null || true
pnpm store prune 2>/dev/null || true

log_space "  After Node PMs"
echo "" | tee -a "$LOG_FILE"

# ──────────────────────────────────────────────
# 5. TIME MACHINE LOCAL SNAPSHOTS
# ──────────────────────────────────────────────
echo ">>> [5/7] Time Machine Local Snapshots" | tee -a "$LOG_FILE"
log_space "  Before TM Snapshots"

tmutil deletelocalsnapshots / 2>/dev/null || true

log_space "  After TM Snapshots"
echo "" | tee -a "$LOG_FILE"

# ──────────────────────────────────────────────
# 6. APPLICATION CACHES (generic placeholder)
# ──────────────────────────────────────────────
echo ">>> [6/7] App Caches (~/Library/Caches)" | tee -a "$LOG_FILE"
log_space "  Before App Caches"

# ⚠️ REPLACE "AppName" BELOW with actual culprits from prior du output
# Example safe targets based on your earlier scan:
rm -rf ~/Library/Caches/com.apple.metadata.mds* 2>/dev/null || true
rm -rf ~/Library/Caches/CloudKit 2>/dev/null || true
rm -rf ~/Library/Caches/com.apple.nsurlsessiond 2>/dev/null || true
# Add specific app caches here AFTER validating with: ls -lhS ~/Library/Caches/AppName/

log_space "  After App Caches"
echo "" | tee -a "$LOG_FILE"

# ──────────────────────────────────────────────
# 7. TRASH + APFS RECLAIM
# ──────────────────────────────────────────────
echo ">>> [7/7] Empty Trash + APFS Reclaim" | tee -a "$LOG_FILE"
log_space "  Before Trash Empty"

rm -rf ~/.Trash/* 2>/dev/null || true

log_space "  After Trash Empty"
echo "" | tee -a "$LOG_FILE"

# ──────────────────────────────────────────────
# FINAL SUMMARY
# ──────────────────────────────────────────────
echo "========================================" | tee -a "$LOG_FILE"
log_space "FINAL AVAILABLE SPACE"
echo "Completed: $(date)" | tee -a "$LOG_FILE"
echo "Full log saved to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Show APFS volume details (informational, no changes)
echo ">>> APFS Volume Info:" | tee -a "$LOG_FILE"
diskutil apfs list 2>/dev/null | grep -E "Name:|Capacity|Used|Free|Purgeable" | tee -a "$LOG_FILE"