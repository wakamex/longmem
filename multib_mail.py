"""Summary of the zoom-prompt multi-session run against the first run and the retrieval baselines."""
import json


def labels(path):
    return {json.loads(l)["question_id"]: json.loads(l)["label"] for l in open(path)}


new = labels("results/hypotheses_endchat_multi133b_j38.eval.jsonl")
old = labels("results/hypotheses_endchat_multi133_j38.eval.jsonl")
common = [q for q in new if q in old]
print(f"endchat with the zoom prompt: {sum(new.values())}/{len(new)} ({100 * sum(new.values()) / len(new):.1f}%), judged by gemini-3.8-flash.")
print(f"first run on the same {len(common)} questions: {sum(old[q] for q in common)}; this run: {sum(new[q] for q in common)}")
for name, path in [
    ("ColBERT-Zero k50", "results/hypotheses_colbert_lightonai_ColBERT-Zero_k50_mr1_obs0_j38.eval.jsonl"),
    ("OpenAI small k50", "results/hypotheses_small_k50_mr1_obs0_j38.eval.jsonl"),
]:
    base = labels(path)
    print(f"{name}: {sum(base[q] for q in common)}/{len(common)}")
print("zoom table: /nvme/claude-tmp/multib_analysis.txt")
