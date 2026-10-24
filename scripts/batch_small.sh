#!/usr/bin/env bash
# Runs that fit in the 4-CPU cloud workspace (N <= 18).  Larger N: see README.
set -e
cd "$(dirname "$0")/.."
P=${1:-winter2017}
for N in 14 16 18; do
  python -W ignore scripts/run_exact.py --N $N --params $P
done
python -W ignore scripts/gs_prep_data.py --N 14 16 18 --params $P
python -W ignore scripts/trotter_scan.py --N 14 16 --params $P --order 2 4
python -W ignore scripts/trotter_scan.py --N 18 --params $P --order 2 --steps 2 4
python -W ignore scripts/trotter_scan.py --N 18 --params $P --order 4 --steps 1 2
