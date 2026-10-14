#!/bin/bash
# The six-arm sweep to 10 runs each, the scouts that read with sch and with zoom, and sch without the view, in three streams.
# Within a stream each arm's runs go back to back so that every run after the first finds its views
# in the prompt cache. Usage: run_queue.sh a|b|c [questions at once]
set -u
cd /code/longmem
case $1 in
  a) arms="none133:2 reader133:2 sch133:2 sch_view16k:2 scout_sch:1" ;;
  b) arms="zoom133:1 scout133:2 scout_zoom:1" ;;
  c) arms="sch_noview:1:10 sch_noview_reworded:1:5 sch_reworded:2:5" ;;
esac
for spec in $arms; do
  IFS=: read -r arm first last <<< "$spec"
  /usr/bin/python3 replicate_arms.py --arms $arm --first $first --last ${last:-10} --parallel 1 --jobs ${2:-6} || exit 1
done
