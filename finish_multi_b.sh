#!/usr/bin/env bash
# Wait for the second multi-session endchat run (zoom prompt), judge it, and email the comparison.
set -uo pipefail
cd /code/longmem
while systemctl --user is-active lme-multib lme-multib-redo >/dev/null; do sleep 60; done
set -a; . ./.env; set +a
cp results/hypotheses_endchat_multi133b.jsonl results/hypotheses_endchat_multi133b_j38.jsonl
python3 eval_retrieval.py --hypotheses results/hypotheses_endchat_multi133b_j38.jsonl --judge openai > /nvme/claude-tmp/multib_judge.out 2>&1
python3 analyze_zoom.py --tag multi133b --workdir /nvme/claude-tmp/lme_multib > /nvme/claude-tmp/multib_analysis.txt 2>&1
python3 multib_mail.py > /nvme/claude-tmp/multib_mail.txt
python3 /code/email/send_notification.py --from-name endchat "endchat zoom-prompt run finished" < /nvme/claude-tmp/multib_mail.txt > /nvme/claude-tmp/multib_mailsent.txt 2>&1
