#!/bin/bash
# SSD PCIe AER error monitor
# Logs RxErr count every 10 seconds to monitor_ssd.log

AER_PATH="/sys/bus/pci/devices/0000:02:00.0/aer_dev_correctable"
LOG_FILE="$(dirname "$0")/monitor_ssd.log"
INTERVAL=10

echo "=== SSD Monitor started at $(date) ===" | tee -a "$LOG_FILE"

PREV_RXERR=0
FIRST=1

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    RXERR=$(grep RxErr "$AER_PATH" 2>/dev/null | awk '{print $2}')
    TOTAL_COR=$(grep TOTAL_ERR_COR "$AER_PATH" 2>/dev/null | awk '{print $2}')
    STATE=$(cat /sys/class/nvme/nvme0/state 2>/dev/null)

    if [ -z "$RXERR" ]; then
        echo "$TIMESTAMP  ERROR: Cannot read AER stats" | tee -a "$LOG_FILE"
        sleep "$INTERVAL"
        continue
    fi

    if [ "$FIRST" -eq 1 ]; then
        DELTA="N/A"
        FIRST=0
    else
        DELTA=$((RXERR - PREV_RXERR))
    fi

    RATE=$(echo "scale=2; $DELTA / $INTERVAL" 2>/dev/null | bc 2>/dev/null || echo "?")

    echo "$TIMESTAMP  state=$STATE  RxErr=$RXERR  (+$DELTA in ${INTERVAL}s, ~${RATE}/s)  TOTAL_COR=$TOTAL_COR" | tee -a "$LOG_FILE"

    # Alert if error rate is high (>10 per interval)
    if [ "$FIRST" -eq 0 ] && [ "$DELTA" -gt 10 ] 2>/dev/null; then
        echo "$TIMESTAMP  *** WARNING: High RxErr rate: +$DELTA errors in ${INTERVAL}s ***" | tee -a "$LOG_FILE"
    fi

    PREV_RXERR=$RXERR
    sleep "$INTERVAL"
done
