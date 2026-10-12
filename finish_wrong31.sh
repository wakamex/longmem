#!/usr/bin/env bash
set -uo pipefail
cd /code/longmem
while systemctl --user is-active lme-wrong31 >/dev/null; do sleep 60; done
set -a; . ./.env; set +a
cp results/hypotheses_endchat_wrong31.jsonl results/hypotheses_endchat_wrong31_j38.jsonl
python3 eval_retrieval.py --hypotheses results/hypotheses_endchat_wrong31_j38.jsonl --judge openai > /nvme/claude-tmp/wrong31_judge.out 2>&1
cp results/hypotheses_endchat_wrong31_j38.eval.jsonl results/hypotheses_endchat_wrong31.eval.jsonl
python3 analyze_zoom.py --tag wrong31 --workdir /nvme/claude-tmp/lme_wrong31 > /nvme/claude-tmp/wrong31_analysis.txt 2>&1
{ echo "Rerun of the 31 wrong multi-session questions with zoom calls logged."; tail -6 /nvme/claude-tmp/wrong31_analysis.txt; echo "Full table: /nvme/claude-tmp/wrong31_analysis.txt"; } > /nvme/claude-tmp/wrong31_mail.txt
python3 /code/email/send_notification.py --from-name endchat "endchat wrong-answer zoom analysis finished" < /nvme/claude-tmp/wrong31_mail.txt > /nvme/claude-tmp/wrong31_mailsent.txt 2>&1
