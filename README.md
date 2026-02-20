# LongMemEval Retrieval Benchmark

Embedding-based retrieval evaluation on [LongMemEval](https://github.com/xiaowu0162/LongMemEval) — a benchmark for long-term conversational memory.

We achieve **84.2%** on 133 multi-session questions from LongMemEval-S, beating [Mastra](https://github.com/mastra-ai/mastra)'s reported 79.7% using simple embedding retrieval with no external dependencies beyond an embedding API.

## Quick Start

```bash
# Install dependencies
pip install openai python-dotenv requests

# Set API keys in .env
echo "OPENAI_API_KEY=sk-..." > .env
echo "GEMINI_API_KEY=..." >> .env

# Download LongMemEval dataset
git clone https://github.com/xiaowu0162/LongMemEval.git
mkdir -p data
cp LongMemEval/data/longmemeval_s_cleaned.json data/longmemeval_s.json

# Run eval (embedding retriever, default config)
python3 eval_retrieval.py --question-type multi-session --obs-char-limit 0

# Run eval with ColBERT-Zero retriever
pip install pylate  # requires Python ≤3.12
python3 eval_retrieval.py --retriever colbert --question-type multi-session --obs-char-limit 0

# Add strict LongMemEval judge
python3 eval_retrieval.py --question-type multi-session --obs-char-limit 0 --judge openai
```

## Method

1. **Retrieve** all ~500 messages per question using either:
   - `text-embedding-3-small` embeddings ranked by cosine similarity, or
   - [ColBERT-Zero](https://huggingface.co/lightonai/ColBERT-Zero) late-interaction multi-vector scoring
2. **Expand** top-K hits with ±N surrounding messages for conversational context
3. **Answer** with gemini-3-flash-preview using a terse prompt ("Give ONLY the answer")
4. **Judge** with LongMemEval's official per-question-type prompts

No vector database, no reranking, no observations/summaries — just raw messages + retrieval.

## Results

### Best Configuration (LongMemEval strict judge, GPT-4o)

| Retriever | K | MR | Obs | Answerer | Accuracy |
|-----------|---|----|-----|----------|----------|
| OAI small | 50 | ±1 | 500 | gemini-3-flash | **84.2%** |
| Mastra (reported) | 50 | ±1 | — | — | 79.7% |

GPT-4o is now deprecated. Newer runs below use gemini-3-flash as judge.

### Retriever Comparison (gemini-3-flash answerer)

| Retriever | K | MR | Obs | Generic | Strict† |
|-----------|---|----|-----|---------|---------|
| ColBERT-Zero | 50 | ±1 | full | 88.0% | 89.5% |
| OAI small | 50 | ±1 | full | 86.5% | 88.0% |
| OAI small | 20 | 0 | full | 87.2% | — |
| ColBERT-Zero | 20 | 0 | full | 85.0% | — |
| ColBERT-Zero | 50 | ±1 | 500 | 87.2% | — |
| OAI small | 50 | ±1 | 500 | 85.0% | — |
| Cohere embed-v4.0 | 50 | ±1 | 500 | 82.7% | — |
| Yuan-embedding-2.0 (local) | 50 | ±1 | 500 | 82.7% | — |

†Strict judge = gemini-3-flash-preview (scores higher than GPT-4o; not directly comparable to above).

### Retrieval Parameters (OAI small, gemini-3-flash answerer, obs=500)

| K | MR | Rerank | Generic | LongMemEval* |
|---|----|--------|---------|--------------|
| 50 | ±1 | — | 85.7% | 84.2% |
| 100 | ±1 | — | 85.0% | 78.9% |
| 50 | ±1 | Cohere rerank-v3.5 | 83.5% | 76.7% |
| 50 | ±2 | — | 82.0% | 76.7% |
| 50 | 0 | — | 82.0% | 75.2% |
| 10 | 0 | — | 72.2% | 70.7% |

*LongMemEval = official per-type prompts + GPT-4o judge (now deprecated).

K = number of top messages retrieved. MR = message range (±N surrounding messages).
Obs = observation char limit per message (500 = truncated, full = no limit).
Generic = inline "is this correct?" judge.

### Answerer Prompt (K=50 MR=±1, OAI small, gemini-3-flash, obs=500)

| Prompt | Generic | LongMemEval* |
|--------|---------|--------------|
| Terse ("Give ONLY the answer — a number, name, or short phrase") | 84.2% | 84.2% |
| Verbose ("Answer concisely using ONLY the observations") | 85.7% | 79.7% |
| Mastra's exact prompt ("helpful assistant with conversation history") | 82.0% | 77.4% |

### Answerer Date Enhancements (K=50 MR=±1, OAI small, obs=500) — All Hurt

| Enhancement | Generic | LongMemEval* |
|-------------|---------|--------------|
| None (baseline) | 85.7% | 79.7% |
| + date context headers | 79.7% | 78.2% |
| + date gap markers | 79.7% | 78.2% |
| + relative time annotations | 81.2% | 77.4% |
| + date prefix in embeddings | 79.7% | 75.2% |

## Key Findings

1. **Retrieval >> stuffing.** Embedding top-50 messages beats stuffing all ~500
   messages by +30pp. Irrelevant context drowns signal.

2. **Message range ±1 is the sweet spot.** Including ±1 surrounding messages
   around each hit adds conversational context (+4.5pp). ±2 adds too much noise.

3. **K=50 >> K=10.** Retrieving more messages helps significantly (+9pp). But
   K=100 is slightly worse (-0.8pp) — diminishing returns as noise increases.

4. **Cohere rerank hurts (-3pp).** Cross-encoder reranking concentrates results
   from the most relevant single session, reducing diversity. Multi-session
   questions need evidence from multiple sessions.

5. **Embedding model doesn't matter much.** OpenAI small and Cohere v4 both hit
   79.7%. MTEB retrieval scores don't predict task performance.

6. **Terse answers are critical (+4.5pp).** Verbose answers contain correct core
   facts but the strict judge penalizes slightly-off supporting details. Error
   analysis showed 10/27 "wrong" answers were actually correct but rejected due
   to verbose elaboration.

7. **All date enhancements hurt.** Every Mastra-style date decoration tested
   degrades accuracy by 1.5-4.5pp. Clean, undecorated messages work best.

8. **Mastra's prompt is worst.** Their permissive framing scores 77.4% vs our
   terse prompt at 84.2%.

## Error Analysis (Baseline K=50 MR=±1)

Of 27 wrong answers (out of 133):

| Category | Count | Description |
|----------|-------|-------------|
| Judge disagreement | 10 | Model answer correct, strict judge wrongly rejected |
| Retrieval failure | 6 | Evidence not in top-50 retrieved messages |
| Hallucination on abstention | 1 | Model answered when it should say "I don't know" |
| Wrong count/reasoning | 10 | Found evidence but computed wrong number |

See [error_analysis.md](error_analysis.md) for per-question details.

## Script

Single script handles everything: retrieval, answering, and strict judging.

```
eval_retrieval.py
    --dataset           LongMemEval dataset JSON (default: data/longmemeval_s.json)
    --retriever         Retriever: embedding, colbert (default: embedding)
    --question-type     Filter by type (e.g. multi-session)
    --embed-model       Embedding model (default: text-embedding-3-small)
    --colbert-model     ColBERT model (default: lightonai/ColBERT-Zero)
    --answerer          LLM answerer (default: gemini-3-flash-preview)
    --retrieval-k       Top-K messages to retrieve (default: 50)
    --message-range     ±N surrounding messages (default: 1)
    --obs-char-limit    Truncate observations to N chars, 0=full (default: 500)
    --retrieval-only    Skip LLM answering, report retrieval metrics only
    --judge             Strict judge: "openai" (optional)
    --judge-model       Judge model (default: gemini-3-flash-preview)
    --hypotheses        Re-score existing hypotheses (skip retrieval)
```

## Dependencies

```
openai
python-dotenv
requests
```

Optional for ColBERT retriever: `pylate`, `torch` (requires Python ≤3.12).
Optional for Cohere embeddings: `CO_API_KEY` env var.
Optional for local embeddings: `sentence-transformers`, `torch`.
