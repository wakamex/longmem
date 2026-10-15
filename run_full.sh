#!/bin/bash
# Runs of the full-context arm until its comparisons with the arms next to it in score are settled
# (replicates.py), at least 3 and at most 10, back to back so that later runs find the chat cached.
set -u
cd /code/longmem
for n in $(seq 1 10); do
  /usr/bin/python3 replicate_arms.py --arms full133 --first $n --last $n --parallel 1 --jobs 6 || exit 1
  /usr/bin/python3 replicates.py --settled full133 && break
done
/usr/bin/python3 replicates.py
