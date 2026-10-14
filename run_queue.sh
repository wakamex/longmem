#!/bin/bash
# The six-arm sweep to 10 runs each, then the scouts that read with sch and with zoom.
# Each arm's runs go back to back so that every run after the first finds its views in the prompt cache.
set -u
cd /code/longmem
for arm in none133 reader133 sch133 sch_view16k zoom133 scout133; do
  /usr/bin/python3 replicate_arms.py --arms $arm --first 2 --last 10 --parallel 1 || exit 1
done
/usr/bin/python3 replicate_arms.py --arms scout_sch --first 1 --last 10 --parallel 1 &&
/usr/bin/python3 replicate_arms.py --arms scout_zoom --first 1 --last 10 --parallel 1
/usr/bin/python3 replicates.py
