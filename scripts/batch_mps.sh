#!/usr/bin/env bash
# MPS baseline runs that fit the 4-CPU workspace; each line < 2 h.
set -e
cd "$(dirname "$0")/.."
P=${1:-winter2017}
python -W ignore scripts/mps_scan.py --W 2 --L 4 --params $P --chi 16 32 64 --exact
python -W ignore scripts/mps_scan.py --W 3 --L 3 --params $P --chi 16 32 --exact
