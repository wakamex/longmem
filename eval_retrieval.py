#!/usr/bin/env python3
"""Evaluate embedding retrieval on LongMemEval.

For each question:
  1. Load raw messages from oracle haystack_sessions
  2. Rank by cosine similarity to question using embedding model
  3. Give top-K messages (with ±N surrounding context) to an LLM answerer
  4. Judge answer correctness (generic inline judge + optional strict LongMemEval judge)

Usage:
    # Best config: K=50, message-range=±1, terse prompt
    python3 scripts/eval_retrieval.py \
        --dataset data/longmemeval_s.json \
        --question-type multi-session

    # Also run strict LongMemEval judge (GPT-4o) in one command
    python3 scripts/eval_retrieval.py \
        --dataset data/longmemeval_s.json \
        --question-type multi-session \
        --judge openai --judge-model gpt-4o

    # Re-score existing hypotheses with strict judge only (no retrieval)
    python3 scripts/eval_retrieval.py \
        --dataset data/longmemeval_s.json \
        --hypotheses results/hypotheses_small_k50_mr1.jsonl \
        --judge openai --judge-model gpt-4o
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import re
import sys
import threading
import time
import types
from dataclasses import dataclass
from pathlib import Path

import requests
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Message:
    id: int
    text: str
    session_id: str | None = None


# ---------------------------------------------------------------------------
# Load messages from oracle
# ---------------------------------------------------------------------------

def load_messages(oracle_entry: dict) -> list[Message]:
    """Load raw chat messages from oracle haystack_sessions."""
    messages: list[Message] = []
    session_ids = oracle_entry.get("haystack_session_ids", [])
    msg_id = 0
    for session_idx, session in enumerate(oracle_entry.get("haystack_sessions", [])):
        sid = session_ids[session_idx] if session_idx < len(session_ids) else None
        for turn in session:
            text = f"[{turn['role']}] {turn['content']}"
            messages.append(Message(id=msg_id, text=text, session_id=sid))
            msg_id += 1
    return messages


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _truncate_for_embedding(text: str, max_chars: int = 28000) -> str:
    """Truncate text to fit within embedding model token limit (~7K tokens)."""
    return text[:max_chars] if len(text) > max_chars else text


def _embed_texts_openai(
    texts: list[str], client, model: str = "text-embedding-3-small",
    batch_size: int = 512, max_retries: int = 5,
) -> list[list[float]]:
    """Embed texts using OpenAI embeddings API."""
    truncated = [_truncate_for_embedding(t) for t in texts]
    all_embeddings: list[list[float]] = []
    for i in range(0, len(truncated), batch_size):
        batch = truncated[i:i + batch_size]
        for attempt in range(max_retries):
            try:
                resp = client.embeddings.create(model=model, input=batch)
                sorted_data = sorted(resp.data, key=lambda x: x.index)
                all_embeddings.extend([d.embedding for d in sorted_data])
                break
            except Exception as e:
                if attempt < max_retries - 1 and ("429" in str(e) or "rate" in str(e).lower() or "500" in str(e) or "server_error" in str(e).lower()):
                    time.sleep(2 ** attempt + 1)
                else:
                    raise
        else:
            raise RuntimeError(f"Embedding failed after {max_retries} retries")
    return all_embeddings


def _embed_texts_cohere(
    texts: list[str], model: str = "embed-v4.0",
    input_type: str = "search_document", batch_size: int = 96,
    max_retries: int = 5,
) -> list[list[float]]:
    """Embed texts using Cohere embeddings API."""
    api_key = os.environ.get("CO_API_KEY") or os.environ.get("COHERE_API_KEY")
    if not api_key:
        raise RuntimeError("CO_API_KEY or COHERE_API_KEY not set")
    truncated = [_truncate_for_embedding(t) for t in texts]
    all_embeddings: list[list[float]] = []
    for i in range(0, len(truncated), batch_size):
        batch = truncated[i:i + batch_size]
        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    "https://api.cohere.com/v2/embed",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={"texts": batch, "model": model,
                          "input_type": input_type, "embedding_types": ["float"]},
                    timeout=60,
                )
                resp.raise_for_status()
                all_embeddings.extend(resp.json()["embeddings"]["float"])
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt + 1)
                else:
                    raise
        else:
            raise RuntimeError(f"Cohere embedding failed after {max_retries} retries")
    return all_embeddings


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------

def _cache_path(cache_dir: Path, embed_model: str, question_id: str) -> Path:
    model_short = embed_model.replace("text-embedding-3-", "").replace("/", "_")
    return cache_dir / model_short / f"{question_id}.json"


def _load_cache(path: Path, expected_count: int) -> list[list[float]] | None:
    if not path.exists():
        return None
    try:
        with path.open() as f:
            data = json.load(f)
        return data if len(data) == expected_count else None
    except Exception:
        return None


def _save_cache(path: Path, embeddings: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(embeddings, f)


# ---------------------------------------------------------------------------
# Cosine similarity ranking
# ---------------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def embedding_rank(
    messages: list[Message], query: str, client,
    embed_model: str, cache_dir: Path | None = None, question_id: str = "",
) -> list[int]:
    """Rank messages by cosine similarity to query. Returns message IDs."""
    is_cohere = embed_model.startswith("embed-")

    # Get message embeddings (cached)
    msg_embeddings = None
    cp = None
    if cache_dir and question_id:
        cp = _cache_path(cache_dir, embed_model, question_id)
        msg_embeddings = _load_cache(cp, len(messages))

    if msg_embeddings is None:
        texts = [m.text for m in messages]
        if is_cohere:
            msg_embeddings = _embed_texts_cohere(texts, model=embed_model, input_type="search_document")
        else:
            msg_embeddings = _embed_texts_openai(texts, client, embed_model)
        if cp:
            _save_cache(cp, msg_embeddings)

    # Embed query
    if is_cohere:
        q_emb = _embed_texts_cohere([query], model=embed_model, input_type="search_query")[0]
    else:
        q_emb = _embed_texts_openai([query], client, embed_model)[0]

    # Score and sort
    scores = [(m.id, _cosine_similarity(q_emb, emb)) for m, emb in zip(messages, msg_embeddings)]
    scores.sort(key=lambda x: x[1], reverse=True)
    return [mid for mid, _ in scores]


# ---------------------------------------------------------------------------
# Exact attention feasibility retrievers (non-CUDA)
# ---------------------------------------------------------------------------

class ExactAttentionRetriever:
    """Reference retriever that scores messages from attention mass.

    Methods:
      - exact_attn_oracle: global top-k over each selected query row.
      - exact_attn_tiled_ref: blockwise running top-k merge over key tiles.
    """

    def __init__(
        self,
        method: str,
        model_name: str,
        layer: int,
        head: int,
        query_tokens: int,
        topk_per_query: int,
        tile_size: int,
        max_seq_len: int,
        device: str,
        dtype: str,
        cache_dir: str | None,
        scan_mode: str,
        micro_batch_size: int,
    ):
        if method not in {"exact_attn_oracle", "exact_attn_tiled_ref"}:
            raise ValueError(f"Unsupported exact attention method: {method}")
        if scan_mode not in {"packed", "per_message"}:
            raise ValueError(f"Unsupported --attn-scan-mode={scan_mode}")

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Exact attention retriever requires `torch` and `transformers`."
            ) from exc

        self.torch = torch
        self.method = method
        self.model_name = model_name
        self.layer = layer
        self.head = head
        self.query_tokens = max(1, query_tokens)
        self.topk_per_query = max(1, topk_per_query)
        self.tile_size = max(1, tile_size)
        self.max_seq_len = max(128, max_seq_len)
        self.cache_dir = cache_dir
        self.scan_mode = scan_mode
        self.micro_batch_size = max(1, micro_batch_size)

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        if dtype == "auto":
            dtype = "float16" if self.device.type == "cuda" else "float32"
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if dtype not in dtype_map:
            raise ValueError(f"Unsupported --attn-dtype={dtype}")
        torch_dtype = dtype_map[dtype]

        cache_kwargs = {}
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            cache_kwargs["cache_dir"] = self.cache_dir

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            **cache_kwargs,
        )
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": True,
            **cache_kwargs,
        }
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                attn_implementation="eager",
                **model_kwargs,
            )
        except TypeError:
            # Older Transformers may not accept attn_implementation in from_pretrained.
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs,
            )
        self.model.to(self.device)
        self.model.eval()
        if hasattr(self.model.config, "_attn_implementation"):
            self.model.config._attn_implementation = "eager"

        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        max_pos = getattr(self.model.config, "max_position_embeddings", None)
        if isinstance(max_pos, int) and max_pos > 0:
            self.max_seq_len = min(self.max_seq_len, max_pos)

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False).input_ids

    def _build_packed_inputs(
        self,
        messages: list[Message],
        query: str,
    ) -> tuple["torch.Tensor", "torch.Tensor", list[int], set[int], int]:
        torch = self.torch

        input_ids: list[int] = []
        token_to_message: list[int] = []
        included_message_ids: set[int] = set()

        if self.tokenizer.bos_token_id is not None:
            input_ids.append(self.tokenizer.bos_token_id)
            token_to_message.append(-1)

        header_ids = self._encode("Conversation memory:\n")
        input_ids.extend(header_ids)
        token_to_message.extend([-1] * len(header_ids))

        question_prefix_ids = self._encode("\n\nQuestion:\n")
        question_ids = self._encode(query)
        if not question_ids:
            eos = self.tokenizer.eos_token_id
            if eos is not None:
                question_ids = [eos]
            else:
                question_ids = [0]
        question_suffix_ids = self._encode("\nAnswer:")

        base_len = len(input_ids) + len(question_prefix_ids) + len(question_suffix_ids)
        max_q_tokens = max(1, self.max_seq_len - base_len)
        if len(question_ids) > max_q_tokens:
            question_ids = question_ids[-max_q_tokens:]

        message_budget = self.max_seq_len - (
            len(input_ids) + len(question_prefix_ids) + len(question_ids) + len(question_suffix_ids)
        )
        message_budget = max(0, message_budget)

        packed: list[tuple[Message, list[int], list[int]]] = []
        for message in reversed(messages):
            message_prefix = self._encode(f"\n[{message.id}] ")
            message_ids = self._encode(message.text)
            if not message_ids:
                continue

            need = len(message_prefix) + len(message_ids)
            if need <= message_budget:
                packed.append((message, message_prefix, message_ids))
                message_budget -= need
                continue

            room_for_message = message_budget - len(message_prefix)
            if room_for_message <= 0:
                continue

            # Keep the message tail if only a partial fit is available.
            message_ids = message_ids[-room_for_message:]
            packed.append((message, message_prefix, message_ids))
            message_budget = 0
            break

        packed.reverse()
        for message, message_prefix, message_ids in packed:
            input_ids.extend(message_prefix)
            token_to_message.extend([-1] * len(message_prefix))
            input_ids.extend(message_ids)
            token_to_message.extend([message.id] * len(message_ids))
            included_message_ids.add(message.id)

        input_ids.extend(question_prefix_ids)
        token_to_message.extend([-1] * len(question_prefix_ids))
        query_start = len(input_ids)
        input_ids.extend(question_ids)
        token_to_message.extend([-1] * len(question_ids))
        query_end = len(input_ids)
        input_ids.extend(question_suffix_ids)
        token_to_message.extend([-1] * len(question_suffix_ids))

        query_positions = list(range(query_start, query_end))
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        mapping_tensor = torch.tensor(token_to_message, dtype=torch.long, device=self.device)
        dropped_messages = len(messages) - len(included_message_ids)
        return input_tensor, mapping_tensor, query_positions, included_message_ids, dropped_messages

    def _build_single_message_inputs(
        self,
        message: Message,
        query: str,
    ) -> tuple[list[int], list[int], list[int]]:
        """Build one candidate prompt so every message is scored consistently."""
        input_ids: list[int] = []
        token_to_message: list[int] = []

        if self.tokenizer.bos_token_id is not None:
            input_ids.append(self.tokenizer.bos_token_id)
            token_to_message.append(-1)

        memory_header = self._encode("Memory:\n")
        message_prefix = self._encode(f"[{message.id}] ")
        message_ids = self._encode(message.text)

        question_prefix = self._encode("\n\nQuestion:\n")
        question_ids = self._encode(query)
        if not question_ids:
            eos = self.tokenizer.eos_token_id
            if eos is not None:
                question_ids = [eos]
            else:
                question_ids = [0]
        question_suffix = self._encode("\nAnswer:")

        # Reserve at least one message token so each candidate remains scorable.
        fixed = (
            len(input_ids)
            + len(memory_header)
            + len(message_prefix)
            + len(question_prefix)
            + len(question_suffix)
            + 1
        )
        max_q_tokens = max(1, self.max_seq_len - fixed)
        if len(question_ids) > max_q_tokens:
            question_ids = question_ids[-max_q_tokens:]

        message_budget = self.max_seq_len - (
            len(input_ids)
            + len(memory_header)
            + len(message_prefix)
            + len(question_prefix)
            + len(question_ids)
            + len(question_suffix)
        )
        if message_budget < 1:
            trim = 1 - message_budget
            if trim < len(question_ids):
                question_ids = question_ids[trim:]
            else:
                question_ids = question_ids[-1:]
            message_budget = self.max_seq_len - (
                len(input_ids)
                + len(memory_header)
                + len(message_prefix)
                + len(question_prefix)
                + len(question_ids)
                + len(question_suffix)
            )

        if message_budget > 0 and len(message_ids) > message_budget:
            message_ids = message_ids[-message_budget:]
        if not message_ids:
            message_ids = [self.tokenizer.eos_token_id or 0]

        input_ids.extend(memory_header)
        token_to_message.extend([-1] * len(memory_header))
        input_ids.extend(message_prefix)
        token_to_message.extend([-1] * len(message_prefix))
        input_ids.extend(message_ids)
        token_to_message.extend([message.id] * len(message_ids))

        input_ids.extend(question_prefix)
        token_to_message.extend([-1] * len(question_prefix))
        query_start = len(input_ids)
        input_ids.extend(question_ids)
        token_to_message.extend([-1] * len(question_ids))
        query_end = len(input_ids)
        input_ids.extend(question_suffix)
        token_to_message.extend([-1] * len(question_suffix))

        query_positions = list(range(query_start, query_end))
        return input_ids, token_to_message, query_positions

    def _topk_oracle(self, row: "torch.Tensor", k: int) -> tuple["torch.Tensor", "torch.Tensor"]:
        torch = self.torch
        k = min(k, row.numel())
        values, indices = torch.topk(row, k=k, dim=-1)
        return indices, values

    def _topk_tiled(self, row: "torch.Tensor", k: int) -> tuple["torch.Tensor", "torch.Tensor"]:
        torch = self.torch
        k = min(k, row.numel())
        best_values = torch.empty(0, device=row.device, dtype=row.dtype)
        best_indices = torch.empty(0, device=row.device, dtype=torch.long)

        for start in range(0, row.numel(), self.tile_size):
            end = min(start + self.tile_size, row.numel())
            tile = row[start:end]
            tk = min(k, tile.numel())
            tile_values, tile_indices = torch.topk(tile, k=tk, dim=-1)
            tile_indices = tile_indices + start

            if best_values.numel() == 0:
                best_values = tile_values
                best_indices = tile_indices
                continue

            merged_values = torch.cat([best_values, tile_values], dim=0)
            merged_indices = torch.cat([best_indices, tile_indices], dim=0)
            keep_values, keep_pos = torch.topk(merged_values, k=min(k, merged_values.numel()), dim=-1)
            best_values = keep_values
            best_indices = merged_indices.index_select(0, keep_pos)

        return best_indices, best_values

    def _resolve_layer_index(self, num_layers: int) -> int:
        layer_idx = self.layer if self.layer >= 0 else num_layers + self.layer
        if layer_idx < 0:
            return 0
        if layer_idx >= num_layers:
            return num_layers - 1
        return layer_idx

    def _capture_layer_attention(
        self,
        input_ids: "torch.Tensor",
        layer_idx: int,
        attention_mask: "torch.Tensor | None" = None,
    ) -> tuple["torch.Tensor", bool]:
        """Capture only one layer's attention if possible.

        Returns:
          (layer_attention, used_full_materialization)
          layer_attention shape: [batch, heads, q_len, k_len]
        """
        torch = self.torch

        model_core = getattr(self.model, "model", None)
        model_layers = getattr(model_core, "layers", None) if model_core is not None else None
        if model_layers is None:
            model_layers = getattr(self.model, "layers", None)

        if model_layers is None or layer_idx >= len(model_layers):
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True,
                    use_cache=False,
                )
            attentions = outputs.attentions
            if attentions is None:
                raise RuntimeError("Model did not return attentions. Enable output_attentions support.")
            return attentions[layer_idx], True

        target_layer = model_layers[layer_idx]
        self_attn = getattr(target_layer, "self_attn", None)
        if self_attn is None:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True,
                    use_cache=False,
                )
            attentions = outputs.attentions
            if attentions is None:
                raise RuntimeError("Model did not return attentions. Enable output_attentions support.")
            return attentions[layer_idx], True

        captured: dict[str, "torch.Tensor"] = {}
        original_forward = self_attn.forward

        def wrapped_forward(module_self, *args, **kwargs):
            forced_kwargs = dict(kwargs)
            forced_kwargs["output_attentions"] = True
            out = original_forward(*args, **forced_kwargs)
            if isinstance(out, tuple) and len(out) >= 2:
                captured["attn"] = out[1]
                if not kwargs.get("output_attentions", False):
                    if len(out) > 2:
                        out = (out[0], None, *out[2:])
                    else:
                        out = (out[0], None)
            return out

        self_attn.forward = types.MethodType(wrapped_forward, self_attn)
        try:
            with torch.no_grad():
                _ = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=False,
                    use_cache=False,
                )
        finally:
            self_attn.forward = original_forward

        layer_attn = captured.get("attn")
        if layer_attn is None:
            raise RuntimeError("Failed to capture target-layer attention via hook.")
        return layer_attn, False

    def _get_num_layers(self) -> int:
        model_core = getattr(self.model, "model", None)
        model_layers = getattr(model_core, "layers", None) if model_core is not None else None
        if model_layers is None:
            model_layers = getattr(self.model, "layers", None)
        if model_layers is not None:
            return len(model_layers)
        return max(1, int(getattr(self.model.config, "num_hidden_layers", 1)))

    def _select_head(self, layer_attn: "torch.Tensor") -> tuple["torch.Tensor", str]:
        if self.head >= 0:
            head_idx = min(self.head, int(layer_attn.shape[0]) - 1)
            return layer_attn[head_idx], f"head_{head_idx}"
        return layer_attn.mean(dim=0), "mean_heads"

    def _score_query_rows(
        self,
        attn_2d: "torch.Tensor",
        token_to_message: "torch.Tensor",
        query_positions: list[int],
        message_scores: dict[int, float],
    ) -> int:
        torch = self.torch
        if len(query_positions) > self.query_tokens:
            query_positions = query_positions[-self.query_tokens:]
        if not query_positions:
            query_positions = [int(attn_2d.shape[0]) - 1]

        query_rows = attn_2d.index_select(
            0, torch.tensor(query_positions, device=attn_2d.device, dtype=torch.long)
        )
        non_message_mask = token_to_message < 0
        token_to_message_cpu = token_to_message.detach().cpu().tolist()

        topk_fn = self._topk_oracle if self.method == "exact_attn_oracle" else self._topk_tiled
        for row in query_rows:
            masked_row = row.masked_fill(non_message_mask, float("-inf"))
            key_indices, key_values = topk_fn(masked_row, self.topk_per_query)
            key_indices_list = key_indices.detach().cpu().tolist()
            key_values_list = key_values.detach().cpu().tolist()
            for token_idx, score in zip(key_indices_list, key_values_list):
                if not math.isfinite(score):
                    continue
                mid = token_to_message_cpu[token_idx]
                if mid >= 0 and mid in message_scores:
                    message_scores[mid] += float(score)

        n_rows = int(query_rows.shape[0])
        if n_rows > 0:
            for mid in list(message_scores.keys()):
                message_scores[mid] = message_scores[mid] / float(n_rows)
        return n_rows

    def _rank_packed(self, messages: list[Message], query: str) -> tuple[list[int], dict]:
        (
            input_ids,
            token_to_message,
            query_positions,
            included_message_ids,
            dropped_messages,
        ) = self._build_packed_inputs(messages, query)

        layer_idx = self._resolve_layer_index(self._get_num_layers())
        layer_attn_batch, used_full_materialization = self._capture_layer_attention(input_ids, layer_idx)
        layer_attn = layer_attn_batch[0]
        attn_2d, head_mode = self._select_head(layer_attn)

        message_scores = {m.id: float("-inf") for m in messages}
        for mid in included_message_ids:
            message_scores[mid] = 0.0

        included_scores = {mid: message_scores[mid] for mid in included_message_ids}
        query_used = self._score_query_rows(attn_2d, token_to_message, query_positions, included_scores)
        for mid, score in included_scores.items():
            message_scores[mid] = score

        ranking = [mid for mid, _ in sorted(message_scores.items(), key=lambda kv: kv[1], reverse=True)]
        diag = {
            "retriever": self.method,
            "scan_mode": "packed",
            "attn_model": self.model_name,
            "attn_layer": layer_idx,
            "attn_head": head_mode,
            "seq_len": int(input_ids.shape[1]),
            "query_tokens_used": query_used,
            "topk_per_query": self.topk_per_query,
            "included_messages": len(included_message_ids),
            "dropped_messages": dropped_messages,
            "attention_capture": (
                "all_layers_output_attentions"
                if used_full_materialization else "single_layer_hook"
            ),
        }
        return ranking, diag

    def _rank_per_message(self, messages: list[Message], query: str) -> tuple[list[int], dict]:
        torch = self.torch
        layer_idx = self._resolve_layer_index(self._get_num_layers())
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0

        message_scores = {m.id: 0.0 for m in messages}
        used_full_materialization = False
        head_mode = "mean_heads"
        total_query_tokens = 0
        total_seq_len = 0

        for start in range(0, len(messages), self.micro_batch_size):
            batch_messages = messages[start:start + self.micro_batch_size]
            built: list[tuple[int, list[int], list[int], list[int]]] = []
            max_len = 0
            for message in batch_messages:
                ids, mapping, query_pos = self._build_single_message_inputs(message, query)
                built.append((message.id, ids, mapping, query_pos))
                max_len = max(max_len, len(ids))

            batch_size = len(built)
            input_batch = torch.full(
                (batch_size, max_len), pad_id, dtype=torch.long, device=self.device
            )
            attention_mask = torch.zeros(
                (batch_size, max_len), dtype=torch.long, device=self.device
            )
            mapping_batch = torch.full(
                (batch_size, max_len), -1, dtype=torch.long, device=self.device
            )

            for bi, (_, ids, mapping, _) in enumerate(built):
                seq_len = len(ids)
                total_seq_len += seq_len
                input_batch[bi, :seq_len] = torch.tensor(ids, dtype=torch.long, device=self.device)
                attention_mask[bi, :seq_len] = 1
                mapping_batch[bi, :seq_len] = torch.tensor(
                    mapping, dtype=torch.long, device=self.device
                )

            layer_attn_batch, used_full = self._capture_layer_attention(
                input_batch, layer_idx, attention_mask=attention_mask
            )
            used_full_materialization = used_full_materialization or used_full

            for bi, (mid, _, _, query_positions) in enumerate(built):
                layer_attn = layer_attn_batch[bi]
                attn_2d, head_mode = self._select_head(layer_attn)
                local_scores = {mid: 0.0}
                used = self._score_query_rows(attn_2d, mapping_batch[bi], query_positions, local_scores)
                total_query_tokens += used
                message_scores[mid] = local_scores[mid]

        ranking = [mid for mid, _ in sorted(message_scores.items(), key=lambda kv: kv[1], reverse=True)]
        avg_seq_len = int(round(total_seq_len / len(messages))) if messages else 0
        avg_query = int(round(total_query_tokens / len(messages))) if messages else 0
        diag = {
            "retriever": self.method,
            "scan_mode": "per_message",
            "attn_model": self.model_name,
            "attn_layer": layer_idx,
            "attn_head": head_mode,
            "seq_len": avg_seq_len,
            "query_tokens_used": avg_query,
            "topk_per_query": self.topk_per_query,
            "included_messages": len(messages),
            "dropped_messages": 0,
            "micro_batch_size": self.micro_batch_size,
            "attention_capture": (
                "all_layers_output_attentions"
                if used_full_materialization else "single_layer_hook"
            ),
        }
        return ranking, diag

    def rank(self, messages: list[Message], query: str) -> tuple[list[int], dict]:
        if not messages:
            return [], {}
        if self.scan_mode == "per_message":
            return self._rank_per_message(messages, query)
        return self._rank_packed(messages, query)


class ColBERTRetriever:
    """Late-interaction retriever using ColBERT multi-vector scoring."""

    def __init__(self, model_name: str, device: str, batch_size: int):
        from pylate import models, rank

        self.rank_module = rank
        self.model_name = model_name
        self.batch_size = batch_size

        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        self.model = models.ColBERT(model_name_or_path=model_name, device=device)
        self.model.half()

    def rank(self, messages: list[Message], query: str) -> tuple[list[int], dict]:
        if not messages:
            return [], {}

        doc_texts = [m.text for m in messages]
        doc_ids = [m.id for m in messages]

        query_embeddings = self.model.encode([query], is_query=True, batch_size=1)
        doc_embeddings = self.model.encode(
            doc_texts, is_query=False, batch_size=self.batch_size
        )

        reranked = self.rank_module.rerank(
            documents_ids=[doc_ids],
            queries_embeddings=query_embeddings,
            documents_embeddings=[doc_embeddings],
        )
        ranking = [entry["id"] for entry in reranked[0]]
        diag = {"retriever": "colbert", "model": self.model_name}
        return ranking, diag


def _session_hit(
    msg_by_id: dict[int, Message],
    ranked_ids: list[int],
    answer_session_ids: list[str],
) -> bool:
    if not answer_session_ids:
        return False
    answer_sessions = set(answer_session_ids)
    for mid in ranked_ids:
        message = msg_by_id.get(mid)
        if message and message.session_id in answer_sessions:
            return True
    return False


# ---------------------------------------------------------------------------
# LLM answerer + inline judge
# ---------------------------------------------------------------------------

ANSWER_PROMPT = """\
You are answering a question using ONLY the observations below from a chat \
memory system. These observations are retrieved memories from past conversations.

Observations:
{observations}

Question: {question}

Give ONLY the answer — a number, name, or short phrase. \
Do not explain your reasoning or list supporting evidence. \
If the observations don't contain enough information, say "I don't know"."""

JUDGE_PROMPT = """\
You are judging whether a model's answer is correct given a ground truth answer.

Question: {question}
Ground truth answer: {answer}
Model answer: {hypothesis}

Is the model's answer correct? It doesn't need to be word-for-word identical, \
but it must convey the same essential information. Answer "yes" or "no" only."""


# ---------------------------------------------------------------------------
# LongMemEval strict judge prompts (from official evaluate_qa.py)
# ---------------------------------------------------------------------------

_STRICT_BASE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
    "If the response is equivalent to the correct answer or contains all the intermediate "
    "steps to get the correct answer, you should also answer yes. If the response only "
    "contains a subset of the information required by the answer, answer no. "
    "\n\nQuestion: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {hypothesis}"
    "\n\nIs the model response correct? Answer yes or no only."
)

STRICT_JUDGE_PROMPTS = {
    "single-session-user": _STRICT_BASE,
    "single-session-assistant": _STRICT_BASE,
    "multi-session": _STRICT_BASE,
    "temporal-reasoning": (
        "I will give you a question, a correct answer, and a response from a model. "
        "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
        "If the response is equivalent to the correct answer or contains all the intermediate "
        "steps to get the correct answer, you should also answer yes. If the response only "
        "contains a subset of the information required by the answer, answer no. "
        "In addition, do not penalize off-by-one errors for the number of days. "
        "If the question asks for the number of days/weeks/months, etc., and the model makes "
        "off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's "
        "response is still correct. "
        "\n\nQuestion: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {hypothesis}"
        "\n\nIs the model response correct? Answer yes or no only."
    ),
    "knowledge-update": (
        "I will give you a question, a correct answer, and a response from a model. "
        "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
        "If the response contains some previous information along with an updated answer, "
        "the response should be considered as correct as long as the updated answer is the "
        "required answer."
        "\n\nQuestion: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {hypothesis}"
        "\n\nIs the model response correct? Answer yes or no only."
    ),
    "single-session-preference": (
        "I will give you a question, a rubric for desired personalized response, and a response "
        "from a model. Please answer yes if the response satisfies the desired response. "
        "Otherwise, answer no. The model does not need to reflect all the points in the rubric. "
        "The response is correct as long as it recalls and utilizes the user's personal "
        "information correctly."
        "\n\nQuestion: {question}\n\nRubric: {answer}\n\nModel Response: {hypothesis}"
        "\n\nIs the model response correct? Answer yes or no only."
    ),
    "abstention": (
        "I will give you an unanswerable question, an explanation, and a response from a model. "
        "Please answer yes if the model correctly identifies the question as unanswerable. "
        "The model could say that the information is incomplete, or some other information is "
        "given but the asked information is not."
        "\n\nQuestion: {question}\n\nExplanation: {answer}\n\nModel Response: {hypothesis}"
        "\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
    ),
}


def _get_strict_judge_prompt(question_type: str, question_id: str,
                             question: str, answer: str, hypothesis: str) -> str:
    if "_abs" in question_id:
        template = STRICT_JUDGE_PROMPTS["abstention"]
    else:
        template = STRICT_JUDGE_PROMPTS.get(question_type, _STRICT_BASE)
    return template.format(question=question, answer=answer, hypothesis=hypothesis)


class RateLimiter:
    def __init__(self, rpm: int):
        self.interval = 60.0 / rpm if rpm > 0 else 0
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self):
        if self.interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            earliest = self._last + self.interval
            if now < earliest:
                time.sleep(earliest - now)
            self._last = time.monotonic()


_rate_limiter: RateLimiter | None = None


def _llm_call(client, model, messages, max_tokens, max_retries=8):
    for attempt in range(max_retries):
        if _rate_limiter:
            _rate_limiter.wait()
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages,
                max_completion_tokens=max_tokens,
                temperature=0 if "gpt-4" in model or "gemini" in model else 1,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            err = str(e).lower()
            if ("429" in str(e) or "rate" in err or "quota" in err
                    or "resource_exhausted" in err) and attempt < max_retries - 1:
                time.sleep(2 ** attempt + 5)
            else:
                raise
    raise RuntimeError(f"LLM call failed after {max_retries} retries")


def answer_and_judge(
    top_messages: list[Message], oracle_entry: dict, client, model: str,
    obs_char_limit: int = 500,
) -> tuple[str, bool]:
    """Give retrieved messages to LLM, get answer, judge correctness."""
    question = oracle_entry["question"]
    answer = str(oracle_entry["answer"])

    if obs_char_limit > 0:
        obs_text = "\n".join(
            f"[{i+1}] {m.text[:obs_char_limit]}" for i, m in enumerate(top_messages)
        )
    else:
        obs_text = "\n".join(
            f"[{i+1}] {m.text}" for i, m in enumerate(top_messages)
        )
    prompt = ANSWER_PROMPT.format(observations=obs_text, question=question)

    try:
        llm_answer = _llm_call(client, model, [{"role": "user", "content": prompt}], 1024)
    except Exception as e:
        return f"ERROR: {e}", False

    judge_prompt = JUDGE_PROMPT.format(question=question, answer=answer, hypothesis=llm_answer)
    try:
        judge_resp = _llm_call(client, model, [{"role": "user", "content": judge_prompt}], 256)
        correct = "yes" in judge_resp.lower()
    except Exception:
        return llm_answer, False

    return llm_answer, correct


# ---------------------------------------------------------------------------
# Strict judge (LongMemEval per-question-type prompts)
# ---------------------------------------------------------------------------

BAR_WIDTH = 20


def _bar(fraction: float) -> str:
    filled = round(fraction * BAR_WIDTH)
    return "\u2588" * filled + "\u2591" * (BAR_WIDTH - filled)


def _display_strict_results(
    results: dict[str, bool],
    ref_lookup: dict[str, dict],
    judge_label: str,
) -> None:
    type_counts: dict[str, tuple[int, int]] = {}
    for qid, correct in results.items():
        ref = ref_lookup.get(qid)
        if ref is None:
            continue
        qtype = ref["question_type"]
        c, t = type_counts.get(qtype, (0, 0))
        type_counts[qtype] = (c + (1 if correct else 0), t + 1)

    total_correct = sum(c for c, _ in type_counts.values())
    total_count = sum(t for _, t in type_counts.values())
    overall = total_correct / total_count * 100 if total_count else 0

    print(f"\nStrict LongMemEval Judge: {judge_label}")
    print("\u2500" * 68)
    max_type_len = max(len(t) for t in type_counts) if type_counts else 0
    for qtype in sorted(type_counts):
        c, t = type_counts[qtype]
        pct = c / t * 100 if t else 0
        print(f"  {qtype:<{max_type_len}} : {pct:5.1f}% [{_bar(pct / 100)}] ({c}/{t})")
    print(f"\n  LongMemEval Accuracy: {overall:.1f}% ({total_correct}/{total_count})")


def _run_strict_judge(
    hypotheses: list[dict],
    oracle_lookup: dict[str, dict],
    judge_model: str,
    eval_cache_path: Path,
    max_workers: int = 4,
) -> dict[str, bool]:
    """Run strict LongMemEval judge on hypotheses. Returns {qid: correct}."""
    from openai import OpenAI

    if "gemini" in judge_model:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("GEMINI_API_KEY not set (needed for strict judge).", file=sys.stderr)
            sys.exit(1)
        client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("OPENAI_API_KEY not set (needed for strict judge).", file=sys.stderr)
            sys.exit(1)
        client = OpenAI(api_key=api_key)

    # Load eval cache
    cache: dict[str, bool] = {}
    if eval_cache_path.exists():
        with eval_cache_path.open() as fh:
            for raw in fh:
                try:
                    entry = json.loads(raw.strip())
                    cache[entry["question_id"]] = entry["label"]
                except (json.JSONDecodeError, KeyError):
                    pass

    to_judge = [h for h in hypotheses
                if h["question_id"] not in cache and h["question_id"] in oracle_lookup]

    if to_judge:
        print(f"\nStrict judging {len(to_judge)} questions ({len(cache)} cached) "
              f"with {judge_model}...", flush=True)

        eval_fh = eval_cache_path.open("a")
        done = [0]

        def judge(hyp: dict) -> dict:
            ref = oracle_lookup[hyp["question_id"]]
            prompt = _get_strict_judge_prompt(
                ref["question_type"], hyp["question_id"],
                ref["question"], str(ref["answer"]), hyp["hypothesis"],
            )
            try:
                resp = client.chat.completions.create(
                    model=judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=256, temperature=0,
                )
                response = resp.choices[0].message.content or ""
                response = response.strip()
                label = "yes" in response.lower()
            except Exception as exc:
                response = str(exc)
                label = False
            return {"question_id": hyp["question_id"], "label": label,
                    "judge_response": response}

        workers = max(1, min(max_workers, len(to_judge)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_qid = {pool.submit(judge, h): h["question_id"] for h in to_judge}
            for future in concurrent.futures.as_completed(future_to_qid):
                qid = future_to_qid[future]
                try:
                    result = future.result()
                    cache[result["question_id"]] = result["label"]
                    eval_fh.write(json.dumps(result, ensure_ascii=False) + "\n")
                    eval_fh.flush()
                    done[0] += 1
                    status = "correct" if result["label"] else "wrong"
                    print(f"  [{done[0]}/{len(to_judge)}] {qid}: {status}", flush=True)
                except Exception as exc:
                    print(f"  [{qid}] ERROR: {exc}", file=sys.stderr, flush=True)
        eval_fh.close()

    return cache


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Evaluate embedding retrieval on LongMemEval",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--dataset", type=Path, default=Path("data/longmemeval_s.json"),
                        help="LongMemEval dataset JSON (default: data/longmemeval_s.json)")
    parser.add_argument("--retriever", default="embedding",
                        choices=["embedding", "exact_attn_oracle", "exact_attn_tiled_ref", "colbert"],
                        help="Retriever backend (default: embedding)")
    parser.add_argument("--embed-model", default="text-embedding-3-small",
                        help="Embedding model (OpenAI text-embedding-3-* or Cohere embed-*)")
    parser.add_argument("--attn-model", default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="HF causal LM used by exact_attn_* retrievers")
    parser.add_argument("--attn-layer", type=int, default=-4,
                        help="Attention layer index for exact_attn_* (default: -4)")
    parser.add_argument("--attn-head", type=int, default=-1,
                        help="Attention head index, -1 means mean over heads (default: -1)")
    parser.add_argument("--attn-query-tokens", type=int, default=32,
                        help="Use last N question tokens as query rows (default: 32)")
    parser.add_argument("--attn-topk-per-query", type=int, default=64,
                        help="Top-k key tokens kept per query row (default: 64)")
    parser.add_argument("--attn-tile-size", type=int, default=256,
                        help="Tile size for exact_attn_tiled_ref running top-k (default: 256)")
    parser.add_argument("--attn-max-seq-len", type=int, default=2048,
                        help="Max packed sequence length for exact_attn_* (default: 2048)")
    parser.add_argument("--attn-device", default="auto",
                        help='Device for exact_attn_* ("auto", "cuda", or "cpu")')
    parser.add_argument("--attn-dtype", default="auto",
                        choices=["auto", "float16", "bfloat16", "float32"],
                        help="Model dtype for exact_attn_* (default: auto)")
    parser.add_argument("--attn-cache-dir", default="/tmp/hf_cache",
                        help="Hugging Face cache directory for exact_attn_* models")
    parser.add_argument("--attn-scan-mode", default="per_message",
                        choices=["per_message", "packed"],
                        help="Exact attention scoring mode: full-coverage per_message or packed context")
    parser.add_argument("--attn-micro-batch", type=int, default=8,
                        help="Micro-batch size for per_message exact attention scoring")
    parser.add_argument("--colbert-model", default="lightonai/ColBERT-Zero",
                        help="ColBERT model name or path (default: lightonai/ColBERT-Zero)")
    parser.add_argument("--colbert-device", default="auto",
                        help='Device for ColBERT retriever ("auto", "cuda", or "cpu")')
    parser.add_argument("--colbert-batch-size", type=int, default=64,
                        help="Batch size for ColBERT document encoding (default: 64)")
    parser.add_argument("--answerer", default="gemini-3-flash-preview",
                        help="LLM model for answering + inline judging")
    parser.add_argument("--retrieval-k", type=int, default=50,
                        help="Number of top messages to retrieve (default: 50)")
    parser.add_argument("--message-range", type=int, default=1,
                        help="Include ±N surrounding messages around each hit (default: 1)")
    parser.add_argument("--question-type", type=str, default=None,
                        help="Filter to question type (e.g. multi-session)")
    parser.add_argument("--question-id", type=str, default=None,
                        help="Run a single question")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--obs-char-limit", type=int, default=500,
                        help="Truncate each observation to N chars (0=no limit, default: 500)")
    parser.add_argument("--scorer-rpm", type=int, default=0,
                        help="Rate limit LLM calls (requests/min, 0=unlimited)")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=None,
                        help="Override output path")
    parser.add_argument("--judge", default=None,
                        help='Strict LongMemEval judge: "openai" for OpenAI API')
    parser.add_argument("--judge-model", default="gemini-3-flash-preview",
                        help="Model for strict judge (default: gemini-3-flash-preview)")
    parser.add_argument("--hypotheses", type=Path, default=None,
                        help="Re-score existing hypotheses file (skip retrieval)")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="Skip LLM answer/judge calls and report retrieval oracle-hit metrics only")
    args = parser.parse_args()

    # Load dataset
    with args.dataset.open() as f:
        oracle_data = json.load(f)
    oracle_lookup = {q["question_id"]: q for q in oracle_data}

    # Mode: re-score existing hypotheses
    if args.hypotheses:
        if not args.judge:
            parser.error("--judge is required when using --hypotheses")
        hypotheses = []
        with args.hypotheses.open() as fh:
            for raw in fh:
                line = raw.strip()
                if line:
                    hypotheses.append(json.loads(line))
        if not hypotheses:
            print("No hypotheses found.", file=sys.stderr)
            sys.exit(1)
        eval_cache_path = args.hypotheses.with_suffix(".eval.jsonl")
        cache = _run_strict_judge(
            hypotheses, oracle_lookup, args.judge_model,
            eval_cache_path, max_workers=args.max_workers,
        )
        judge_label = f"openai/{args.judge_model}" if args.judge == "openai" else args.judge
        _display_strict_results(cache, oracle_lookup, judge_label)
        return

    if args.retrieval_only and args.judge:
        parser.error("--judge cannot be used with --retrieval-only")

    # Output paths
    if args.retriever == "embedding":
        run_tag = args.embed_model.replace("text-embedding-3-", "").replace("/", "_")
    elif args.retriever == "colbert":
        run_tag = f"colbert_{args.colbert_model.replace('/', '_')}"
    else:
        run_tag = f"{args.retriever}_{args.attn_model.replace('/', '_')}"

    if args.output:
        out_jsonl = args.output
    else:
        args.results_dir.mkdir(parents=True, exist_ok=True)
        obs_tag = f"_obs{args.obs_char_limit}" if args.obs_char_limit != 500 else ""
        out_jsonl = args.results_dir / f"eval_{run_tag}_k{args.retrieval_k}_mr{args.message_range}{obs_tag}.jsonl"
    hyp_stem = out_jsonl.stem.replace("eval_", "hypotheses_")
    if hyp_stem == out_jsonl.stem:
        hyp_stem = out_jsonl.stem + "_hypotheses"
    out_hypotheses = out_jsonl.with_name(hyp_stem + ".jsonl")

    question_ids = sorted(oracle_lookup.keys())
    if args.question_id:
        question_ids = [args.question_id]
    if args.question_type:
        question_ids = [
            qid for qid in question_ids
            if oracle_lookup.get(qid, {}).get("question_type") == args.question_type
        ]
    if args.limit:
        question_ids = question_ids[:args.limit]

    # API clients
    embed_client = None
    scorer_client = None
    is_cohere = args.embed_model.startswith("embed-")
    openai_key = os.environ.get("OPENAI_API_KEY")

    openai_cls = None
    needs_openai = (
        (args.retriever == "embedding" and not is_cohere)
        or (not args.retrieval_only and "gemini" not in args.answerer)
    )
    if needs_openai:
        if not openai_key:
            print("OPENAI_API_KEY not set.", file=sys.stderr)
            sys.exit(1)
        from openai import OpenAI
        openai_cls = OpenAI

    if args.retriever == "embedding" and not is_cohere:
        embed_client = openai_cls(api_key=openai_key)

    if not args.retrieval_only:
        if "gemini" in args.answerer:
            gemini_key = os.environ.get("GEMINI_API_KEY")
            if not gemini_key:
                print("GEMINI_API_KEY not set.", file=sys.stderr)
                sys.exit(1)
            from openai import OpenAI
            scorer_client = OpenAI(
                api_key=gemini_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
        else:
            scorer_client = embed_client or openai_cls(api_key=openai_key)

    embed_cache_dir = args.results_dir / "embed_cache"

    exact_retriever = None
    colbert_retriever = None
    if args.retriever == "colbert":
        if args.max_workers > 1:
            print("ColBERT retriever uses GPU; forcing --max-workers=1.")
            args.max_workers = 1
        colbert_retriever = ColBERTRetriever(
            model_name=args.colbert_model,
            device=args.colbert_device,
            batch_size=args.colbert_batch_size,
        )
    elif args.retriever != "embedding":
        if args.max_workers > 1:
            print("Exact attention retrievers are single-worker; forcing --max-workers=1.")
            args.max_workers = 1
        exact_retriever = ExactAttentionRetriever(
            method=args.retriever,
            model_name=args.attn_model,
            layer=args.attn_layer,
            head=args.attn_head,
            query_tokens=args.attn_query_tokens,
            topk_per_query=args.attn_topk_per_query,
            tile_size=args.attn_tile_size,
            max_seq_len=args.attn_max_seq_len,
            device=args.attn_device,
            dtype=args.attn_dtype,
            cache_dir=args.attn_cache_dir,
            scan_mode=args.attn_scan_mode,
            micro_batch_size=args.attn_micro_batch,
        )

    global _rate_limiter
    if not args.retrieval_only and args.scorer_rpm > 0:
        _rate_limiter = RateLimiter(args.scorer_rpm)

    print(f"Retriever: {args.retriever}")
    if args.retriever == "embedding":
        print(f"Embedding model: {args.embed_model}")
    elif args.retriever == "colbert":
        print(f"ColBERT model: {args.colbert_model}")
    else:
        print(
            "Attention model: "
            f"{args.attn_model} | layer={args.attn_layer} head={args.attn_head} "
            f"q_tokens={args.attn_query_tokens} topk={args.attn_topk_per_query} "
            f"scan={args.attn_scan_mode} mb={args.attn_micro_batch}"
        )
    if args.retrieval_only:
        print("Mode: retrieval-only (no LLM answer/judge calls)")
    else:
        print(f"Answerer: {args.answerer}")
    print(f"Retrieval: K={args.retrieval_k}, MR=±{args.message_range}")
    print(f"Questions: {len(question_ids)}")

    t_start = time.monotonic()
    write_lock = threading.Lock()
    counter = [0]

    def process_question(qid: str) -> dict | None:
        oracle = oracle_lookup.get(qid)
        if oracle is None:
            return None

        messages = load_messages(oracle)
        if not messages:
            return None
        msg_by_id = {m.id: m for m in messages}

        if args.retriever == "embedding":
            ranking = embedding_rank(
                messages, oracle["question"], embed_client,
                embed_model=args.embed_model,
                cache_dir=embed_cache_dir, question_id=qid,
            )
            retriever_diag = {"retriever": "embedding"}
        elif args.retriever == "colbert":
            ranking, retriever_diag = colbert_retriever.rank(messages, oracle["question"])
        else:
            ranking, retriever_diag = exact_retriever.rank(messages, oracle["question"])

        topk_ids = ranking[:args.retrieval_k]
        topk_session_hit = _session_hit(
            msg_by_id, topk_ids, oracle.get("answer_session_ids", [])
        )

        # Oracle message precision/recall
        answer_sessions = set(oracle.get("answer_session_ids", []))
        oracle_msg_ids = {m.id for m in messages if m.session_id in answer_sessions}
        n_oracle_msgs = len(oracle_msg_ids)
        topk_oracle_hits = sum(1 for mid in topk_ids if mid in oracle_msg_ids)
        topk_precision = topk_oracle_hits / len(topk_ids) if topk_ids else 0.0
        topk_recall = topk_oracle_hits / n_oracle_msgs if n_oracle_msgs else 0.0
        max_recall = min(len(topk_ids), n_oracle_msgs) / n_oracle_msgs if n_oracle_msgs else 0.0
        norm_recall = (topk_recall / max_recall) if max_recall > 0 else 0.0

        # Expand with message range.
        expanded_ids = topk_ids
        if args.message_range > 0:
            max_id = max(m.id for m in messages)
            expanded = set()
            for oid in topk_ids:
                for offset in range(-args.message_range, args.message_range + 1):
                    nid = oid + offset
                    if 0 <= nid <= max_id:
                        expanded.add(nid)
            expanded_ids = sorted(expanded)

        expanded_session_hit = _session_hit(
            msg_by_id, expanded_ids, oracle.get("answer_session_ids", [])
        )
        top_messages = [msg_by_id[oid] for oid in expanded_ids if oid in msg_by_id]

        llm_answer = ""
        correct = None
        if not args.retrieval_only:
            llm_answer, correct = answer_and_judge(
                top_messages, oracle, scorer_client, model=args.answerer,
                obs_char_limit=args.obs_char_limit,
            )

        with write_lock:
            counter[0] += 1
            if args.retrieval_only:
                hit = "HIT" if expanded_session_hit else "MISS"
                print(
                    f"[{counter[0]}/{len(question_ids)}] {qid}  {hit}"
                    f"  P={topk_precision:.2f} R={topk_recall:.2f} MaxR={max_recall:.2f} R/MaxR={norm_recall:.2f}"
                    f"  ({topk_oracle_hits}/{n_oracle_msgs} oracle msgs)",
                    flush=True,
                )
            else:
                status = "OK" if bool(correct) else "WRONG"
                print(f"[{counter[0]}/{len(question_ids)}] {qid}  {status}", flush=True)

        result = {
            "question_id": qid,
            "question_type": oracle["question_type"],
            "question": oracle["question"],
            "answer": str(oracle["answer"]),
            "n_messages": len(messages),
            "topk_ids": topk_ids,
            "n_oracle_msgs": n_oracle_msgs,
            "topk_oracle_hits": topk_oracle_hits,
            "topk_precision": round(topk_precision, 4),
            "topk_recall": round(topk_recall, 4),
            "max_recall": round(max_recall, 4),
            "norm_recall": round(norm_recall, 4),
            "topk_session_hit": topk_session_hit,
            "expanded_session_hit": expanded_session_hit,
            "retriever_diag": retriever_diag,
        }
        if not args.retrieval_only:
            result["llm_answer"] = llm_answer
            result["correct"] = bool(correct)
        return result

    # Process
    if args.max_workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(process_question, qid): qid for qid in question_ids}
            results = []
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
    else:
        results = [r for qid in question_ids if (r := process_question(qid))]

    results.sort(key=lambda r: r["question_id"])

    # Write JSONL
    with out_jsonl.open("w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Write hypotheses (LLM mode only)
    if not args.retrieval_only:
        with out_hypotheses.open("w") as f:
            for r in results:
                f.write(
                    json.dumps(
                        {"question_id": r["question_id"], "hypothesis": r["llm_answer"]}
                    ) + "\n"
                )

    # Summary
    elapsed = time.monotonic() - t_start
    n_topk_hit = sum(1 for r in results if r.get("topk_session_hit"))
    n_expanded_hit = sum(1 for r in results if r.get("expanded_session_hit"))
    topk_hit_pct = n_topk_hit / len(results) * 100 if results else 0
    expanded_hit_pct = n_expanded_hit / len(results) * 100 if results else 0

    avg_precision = sum(r.get("topk_precision", 0) for r in results) / len(results) if results else 0
    avg_recall = sum(r.get("topk_recall", 0) for r in results) / len(results) if results else 0
    avg_max_recall = sum(r.get("max_recall", 0) for r in results) / len(results) if results else 0
    avg_norm_recall = sum(r.get("norm_recall", 0) for r in results) / len(results) if results else 0

    print(f"\nOracle session-hit@{args.retrieval_k}: {n_topk_hit}/{len(results)} = {topk_hit_pct:.1f}%")
    print(
        "Oracle session-hit after message-range expansion: "
        f"{n_expanded_hit}/{len(results)} = {expanded_hit_pct:.1f}%"
    )
    print(f"Oracle msg precision@{args.retrieval_k}: {avg_precision:.3f}")
    print(f"Oracle msg recall@{args.retrieval_k}: {avg_recall:.3f}  (max possible: {avg_max_recall:.3f})")
    print(f"Oracle msg R/MaxR@{args.retrieval_k}: {avg_norm_recall:.3f}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Results: {out_jsonl}")

    if not args.retrieval_only:
        print(f"Hypotheses: {out_hypotheses}")
        n_correct = sum(1 for r in results if r["correct"])
        pct = n_correct / len(results) * 100 if results else 0
        print(f"\nGeneric accuracy: {n_correct}/{len(results)} = {pct:.1f}%")

    # Strict judge (optional)
    if not args.retrieval_only and args.judge:
        hypotheses = [
            {"question_id": r["question_id"], "hypothesis": r["llm_answer"]}
            for r in results
        ]
        eval_cache_path = out_hypotheses.with_suffix(".eval.jsonl")
        cache = _run_strict_judge(
            hypotheses, oracle_lookup, args.judge_model,
            eval_cache_path, max_workers=args.max_workers,
        )
        judge_label = f"openai/{args.judge_model}" if args.judge == "openai" else args.judge
        _display_strict_results(cache, oracle_lookup, judge_label)
    elif not args.retrieval_only:
        print("\nTo also run strict LongMemEval judge, add: --judge openai")


if __name__ == "__main__":
    main()
