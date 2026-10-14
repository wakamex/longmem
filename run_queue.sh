#!/bin/bash
# The six-arm sweep to 10 runs each, then the scouts that read with sch and with zoom, in two streams.
# Within a stream each arm's runs go back to back so that every run after the first finds its views
# in the prompt cache. Usage: run_queue.sh a|b [questions at once]
set -u
cd /code/longmem
case $1 in
  a) arms="none133:2 reader133:2 sch133:2 sch_view16k:2 scout_sch:1" ;;
  b) arms="zoom133:1 scout133:2 scout_zoom:1" ;;
esac
for spec in $arms; do
  /usr/bin/python3 replicate_arms.py --arms ${spec%%:*} --first ${spec##*:} --last 10 --parallel 1 --jobs ${2:-6} || exit 1
done
