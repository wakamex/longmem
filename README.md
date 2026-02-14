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

# Clone LongMemEval dataset
git clone https://github.com/xiaowu0162/LongMemEval.git

# Run best config + strict LongMemEval judge in one command
python3 eval_retrieval.py \
    --dataset LongMemEval/data/longmemeval_s_cleaned.json \
    --question-type multi-session \
    --judge openai --judge-model gpt-4o

# Or re-score existing hypotheses with strict judge only
python3 eval_retrieval.py \
    --dataset LongMemEval/data/longmemeval_s_cleaned.json \
    --hypotheses results/hypotheses_small_k50_mr1.jsonl \
    --judge openai --judge-model gpt-4o
```

## Method

1. **Embed** all ~500 messages per question with `text-embedding-3-small`
2. **Rank** by cosine similarity to the question
3. **Expand** top-50 hits with ±1 surrounding messages for conversational context
4. **Answer** with gemini-3-flash-preview using a terse prompt ("Give ONLY the answer")
5. **Judge** with GPT-4o using LongMemEval's official per-question-type prompts

No vector database, no reranking, no observations/summaries — just raw messages + embeddings.

## Results

### Best Configuration

| Embed Model | K | MR | Answerer | LongMemEval Judge |
|-------------|---|----|----------|-------------------|
| **OAI text-embedding-3-small** | **50** | **±1** | **gemini-3-flash** | **84.2%** |
| Mastra (reported) | ? | ? | gemini-3-flash | 79.7% |

### Retrieval Parameters

| Embed Model | K | MR | Rerank | Answerer | Generic | LongMemEval |
|-------------|---|----|--------|----------|---------|-------------|
| OAI small | 50 | ±1 | — | gemini-3-flash | 85.7% | 79.7%* |
| OAI small | 100 | ±1 | — | gemini-3-flash | 85.0% | 78.9% |
| OAI small | 50 | ±1 | Cohere rerank-v3.5 | gemini-3-flash | 83.5% | 76.7% |
| OAI small | 50 | ±2 | — | gemini-3-flash | 82.0% | 76.7% |
| OAI small | 50 | 0 | — | gemini-3-flash | 82.0% | 75.2% |
| OAI small | 10 | 0 | — | gemini-3-flash | 72.2% | 70.7% |

*With verbose answerer prompt. Terse prompt raises this to 84.2%.

K = number of top messages retrieved. MR = message range (±N surrounding messages).
Generic = inline "is this correct?" judge. LongMemEval = official per-type prompts + GPT-4o.

### Embedding Models (K=50 MR=±1, gemini-3-flash answerer)

| Embed Model | Generic | LongMemEval |
|-------------|---------|-------------|
| OAI text-embedding-3-small | 85.7% | 79.7% |
| Cohere embed-v4.0 | 82.7% | 79.7% |
| Yuan-embedding-2.0-en (local) | 82.7% | 75.2% |

### Answerer Prompt (K=50 MR=±1, OAI small, gemini-3-flash)

| Prompt | Generic | LongMemEval |
|--------|---------|-------------|
| **Terse** ("Give ONLY the answer — a number, name, or short phrase") | **84.2%** | **84.2%** |
| Verbose ("Answer concisely using ONLY the observations") | 85.7% | 79.7% |
| Mastra's exact prompt ("helpful assistant with conversation history") | 82.0% | 77.4% |

### Answerer Date Enhancements (K=50 MR=±1, OAI small) — All Hurt

| Enhancement | Generic | LongMemEval |
|-------------|---------|-------------|
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

See `results/error_analysis.md` for per-question details.

## Script

Single script handles everything: retrieval, answering, and strict judging.

```
eval_retrieval.py
    --dataset           LongMemEval dataset JSON (required)
    --question-type     Filter by type (e.g. multi-session)
    --embed-model       Embedding model (default: text-embedding-3-small)
    --answerer          LLM answerer (default: gemini-3-flash-preview)
    --retrieval-k       Top-K messages to retrieve (default: 50)
    --message-range     ±N surrounding messages (default: 1)
    --judge             Strict judge: "openai" (optional)
    --judge-model       Judge model (default: gpt-4o)
    --hypotheses        Re-score existing hypotheses (skip retrieval)
```

## Dependencies

```
openai
python-dotenv
requests
```

Optional for Cohere embeddings: `CO_API_KEY` env var.
Optional for local embeddings: `sentence-transformers`, `torch`.
