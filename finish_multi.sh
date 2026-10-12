#!/usr/bin/env bash
# Wait for the multi-session endchat run, judge it, and email the comparison with the retrieval baselines.
set -uo pipefail
cd /code/longmem
while systemctl --user is-active lme-multi2 >/dev/null; do sleep 60; done
set -a; . ./.env; set +a
cp results/hypotheses_endchat_multi133.jsonl results/hypotheses_endchat_multi133_j38.jsonl
python3 eval_retrieval.py --hypotheses results/hypotheses_endchat_multi133_j38.jsonl --judge openai > /nvme/claude-tmp/multi133_judge.out 2>&1
python3 - > /nvme/claude-tmp/multi133_mail.txt <<'PY'
import json
ids=[json.loads(l)['question_id'] for l in open('results/hypotheses_endchat_multi133_j38.jsonl')]
e={json.loads(l)['question_id']:json.loads(l)['label'] for l in open('results/hypotheses_endchat_multi133_j38.eval.jsonl')}
st=[json.loads(l) for l in open('results/endchat_stats_multi133.jsonl')]
i=sum(x['import']['input_tokens'] for x in st); c=sum(x['import']['cached_input_tokens'] for x in st); o=sum(x['import']['output_tokens'] for x in st)
print(f"endchat on LongMemEval multi-session: {sum(e.values())}/{len(e)} correct ({100*sum(e.values())/len(e):.1f}%), judged by gemini-3.8-flash.")
print(f"{len(ids)} of 133 questions have answers; Haiku 5.5 summaries, Sonnet 5.5 answers, import paced at 50 messages.")
for name,p in [("ColBERT-Zero k50 mr1 obs0","results/hypotheses_colbert_lightonai_ColBERT-Zero_k50_mr1_obs0_j38.eval.jsonl"),("OAI small k50 mr1 obs0","results/hypotheses_small_k50_mr1_obs0_j38.eval.jsonl")]:
    b={json.loads(l)['question_id']:json.loads(l)['label'] for l in open(p)}
    cm=[q for q in e if q in b]
    print(f"{name} on the same {len(cm)} questions: {sum(b[q] for q in cm)}/{len(cm)} ({100*sum(b[q] for q in cm)/len(cm):.1f}%); endchat {sum(e[q] for q in cm)}/{len(cm)}")
print(f"cost estimate: ${((i-c)*0.1+c*0.01+o*0.5)/1e6:.0f} API-equivalent for summaries, cache reads {100*c/i:.1f}%, {len(st)} questions")
print("details: /code/longmem/results/hypotheses_endchat_multi133_j38.eval.jsonl")
PY
python3 /code/email/send_notification.py --from-name endchat "endchat multi-session run finished" < /nvme/claude-tmp/multi133_mail.txt > /nvme/claude-tmp/multi133_mailsent.txt 2>&1
