#!/usr/bin/env bash
RATE=$1
LOG=/var/log/suricata/stats.csv
OUT=~/tests/${RATE}.txt
mkdir -p ~/tests
awk -v out="$OUT" '
  /^Date: / { ts = $4; next }
  /capture\.kernel_packets/ {
    print ts "," $NF > out
  }
' "$LOG"
echo "→ Débit extrait dans $OUT"