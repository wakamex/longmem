#!/bin/bash
# Runs 1 to 10 of the scout reading with sch, then with zoom, then the search scout's runs 7 to 10.
# Each arm's runs go back to back so that every run after the first finds its views in the prompt cache.
set -u
cd /code/longmem
/usr/bin/python3 replicate_arms.py --arms scout_sch --first 1 --last 10 --parallel 1 &&
/usr/bin/python3 replicate_arms.py --arms scout_zoom --first 1 --last 10 --parallel 1 &&
/usr/bin/python3 replicate_arms.py --arms scout133 --first 7 --last 10 --parallel 1
/usr/bin/python3 replicates.py
