#!/usr/bin/env bash
# Reduced runs for a further parameter set (sensitivity of N*), N = 14, 16.
set -e
cd "$(dirname "$0")/.."
P=$1
for N in 14 16; do
  python -W ignore scripts/run_exact.py --N $N --params $P
done
python -W ignore scripts/gs_prep_data.py --N 14 16 --params $P
python -W ignore scripts/trotter_scan.py --N 14 16 --params $P --order 2 --steps ${STEPS:-2 4}
